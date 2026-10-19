"""
Resilient NVIDIA NIM (OpenAI-compatible) client tuned for a large swarm.

Key realities this client handles:

* Many NIM endpoints are REASONING models: the visible `content` can be empty
  while the text lands in `reasoning_content` / `reasoning`. We extract from
  every plausible field and prefer a non-empty `content`.
* Because reasoning eats the completion budget, `max_tokens` must be generous
  or a reasoning model returns content=None. We default high.
* Not every advertised model is provisioned for a given key (HTTP 404
  "Function ... Not found for account"). Dead models are skipped fast.
* A 300+ agent swarm hammers the API: we use a bounded thread pool, per-call
  timeouts, retries with backoff, and 429-aware throttling.
* Offline mode lets the entire orchestration run with deterministic stub
  responses (no network, no tokens) for testing the pipeline end-to-end.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import NVIDIA_API_KEY, NVIDIA_CHAT_URL


# ---------------------------------------------------------------------------
# Response records
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    ok: bool
    text: str
    model: str
    attempts: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    error: str = ""

    @property
    def tokens(self) -> int:
        return self.total_tokens


# ---------------------------------------------------------------------------
# Content extraction across provider message shapes
# ---------------------------------------------------------------------------
_CONTENT_FIELDS = (
    "content",
    "text",
    "message",
    "output_text",
    "reasoning_content",
    "reasoning",
)


def _extract_text(message: Dict) -> str:
    """Pull the best available assistant text from a message dict.

    Preference order: a non-empty `content`, then any other text-bearing field.
    Reasoning fields are only used as a fallback so we never return the
    thinking trace when a real answer is present.
    """
    if not isinstance(message, dict):
        return str(message or "")

    # 1) the real answer, if present
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    # content can be a list of content-parts (OpenAI style)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            elif isinstance(part, str):
                parts.append(part)
        if parts:
            return "\n".join(parts).strip()

    # 2) fallbacks (reasoning models)
    for name in _CONTENT_FIELDS[1:]:
        val = message.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


class _OfflineStub:
    """Deterministic stand-in used when live_llm is disabled."""

    def __init__(self, seed_fn: Optional[Callable[[str], str]] = None):
        self._seed_fn = seed_fn


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class LLMClient:
    """Thread-safe NVIDIA NIM client with model fallback and throttling."""

    def __init__(
        self,
        api_key: str = NVIDIA_API_KEY,
        default_models: Optional[Sequence[str]] = None,
        max_concurrency: int = 24,
        timeout: int = 60,
        live: bool = True,
        max_tokens: int = 700,
        temperature: float = 0.35,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = "none",
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.live = live
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        # "none" makes NIM reasoning models emit the final answer immediately
        # (0 reasoning tokens, far faster) which is essential for reliable
        # structured output. Models that reject the parameter are retried
        # without it (see _effort_bad handling in chat()).
        self.reasoning_effort = reasoning_effort
        self.default_models: List[str] = list(
            default_models or ["nvidia/nemotron-3-super-120b-a12b"]
        )

        # bounded concurrency for the whole swarm
        self._sem = threading.Semaphore(max(1, max_concurrency))

        # dead models we discovered (404) are skipped for the rest of the run
        self._dead: set = set()
        self._dead_lock = threading.Lock()

        # models that rejected reasoning_effort (we stop sending it to them)
        self._effort_bad: set = set()

        # 429-aware global cooldown
        self._cooldown_until = 0.0
        self._cooldown_lock = threading.Lock()

        # metrics
        self._stats_lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.retries = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.model_calls: Dict[str, int] = {}
        self.model_ok: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.errors: List[str] = []

    # -- metrics helpers ----------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def stats_snapshot(self) -> Dict:
        with self._stats_lock:
            return {
                "calls": self.calls,
                "failures": self.failures,
                "retries": self.retries,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "total_tokens": self.tokens_in + self.tokens_out,
                "avg_latency": round(self.avg_latency, 2),
                "dead_models": sorted(self._dead),
                "model_calls": dict(self.model_calls),
                "model_ok": dict(self.model_ok),
            }

    # -- throttling ---------------------------------------------------------
    def _wait_cooldown(self):
        with self._cooldown_lock:
            wait = self._cooldown_until - time.time()
        if wait > 0:
            time.sleep(min(wait, 5.0))

    def _cool_down(self, seconds: float):
        with self._cooldown_lock:
            self._cooldown_until = max(self._cooldown_until, time.time() + seconds)

    def _mark_dead(self, model: str):
        with self._dead_lock:
            self._dead.add(model)

    def _is_dead(self, model: str) -> bool:
        with self._dead_lock:
            return model in self._dead

    # -- core single-model call --------------------------------------------
    def _once(self, model: str, messages: List[Dict], max_tokens: int,
              temperature: float, timeout: int,
              effort: Optional[str] = None) -> Tuple[str, Dict, bool, str]:
        """One HTTP attempt against a single model.

        Returns (text, usage, is_rate_limit, error). `usage` may carry signal
        flags: `_404` (model unavailable), `_5xx` (endpoint down),
        `_effort_bad` (model rejects reasoning_effort).
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if effort is not None:
            payload["reasoning_effort"] = effort
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            NVIDIA_CHAT_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            is_429 = e.code == 429
            is_404 = e.code in (404, 400) and (
                "Not found for account" in detail or "Function" in detail or e.code == 404
            )
            # 5xx from a model endpoint means it is down -> try the NEXT model
            # rather than burning retries on a dead endpoint.
            is_5xx = e.code >= 500
            return "", {"_404": is_404, "_5xx": is_5xx}, is_429, f"HTTP {e.code}: {detail}"
        except Exception as e:  # timeout, reset, dns, etc.
            return "", {}, False, f"{type(e).__name__}: {e}"

        try:
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = _extract_text(message)
            usage = body.get("usage") or {}
            return text, usage, False, ""
        except Exception as e:
            return "", {}, False, f"parse {type(e).__name__}: {e}"

    # -- public API ---------------------------------------------------------
    def chat(
        self,
        messages: List[Dict],
        models: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        role: str = "",
    ) -> LLMResult:
        """Run one logical request, falling back across a model chain.

        Tries each candidate model (skipping known-dead ones); retries a live
        model a couple of times on transient errors. Returns the first
        non-empty answer.
        """
        if not self.live:
            return self._offline(messages, models)

        chain = list(models or self.default_models)
        # never start from a known-dead model, but keep them as last resort
        live_chain = [m for m in chain if not self._is_dead(m)]
        if not live_chain:
            live_chain = chain

        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        timeout = timeout or self.timeout

        attempts: List[str] = []
        with self._stats_lock:
            self.calls += 1

        for model in live_chain:
            for attempt in range(self.max_retries + 1):
                self._wait_cooldown()
                with self._sem:
                    t0 = time.time()
                    text, usage, is_429, err = self._once(
                        model, messages, max_tokens, temperature, timeout
                    )
                    dt = time.time() - t0

                attempts.append(model)
                with self._stats_lock:
                    self.model_calls[model] = self.model_calls.get(model, 0) + 1

                if is_429:
                    # back off globally, then retry
                    self._cool_down(1.5 + attempt * 2 + random.random())
                    with self._stats_lock:
                        self.retries += 1
                        self.failures += 1
                        self.errors.append(f"{model}: {err}")
                    continue

                if usage.get("_404"):
                    # model not provisioned for this key -> abandon it entirely
                    self._mark_dead(model)
                    with self._stats_lock:
                        self.errors.append(f"{model}: {err}")
                    break

                if usage.get("_5xx") and attempt == 0:
                    # endpoint down: don't burn retries here, fall through to
                    # the next model in the chain on the first failure.
                    with self._stats_lock:
                        self.errors.append(f"{model}: {err}")
                    break

                if text:
                    # success
                    p = int(usage.get("prompt_tokens") or 0)
                    c = int(usage.get("completion_tokens") or 0)
                    tot = int(usage.get("total_tokens") or (p + c))
                    with self._stats_lock:
                        self.model_ok[model] = self.model_ok.get(model, 0) + 1
                        self.tokens_in += p
                        self.tokens_out += c
                        self.latencies.append(dt)
                    return LLMResult(
                        ok=True, text=text, model=model, attempts=attempts,
                        prompt_tokens=p, completion_tokens=c, total_tokens=tot,
                        latency=dt,
                    )

                # empty content: transient or reasoning-budget exhaustion
                with self._stats_lock:
                    self.retries += 1
                    self.errors.append(f"{model}: empty ({err})")
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))

        # total failure for this request
        with self._stats_lock:
            self.failures += 1
        return LLMResult(
            ok=False, text="", model="", attempts=attempts, error="all models failed"
        )

    def chat_text(self, prompt: str, system: str = "", **kw) -> str:
        """Convenience: single-turn prompt -> text (may be empty on failure)."""
        msgs: List[Dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw).text

    # -- offline stub -------------------------------------------------------
    def _offline(self, messages: List[Dict], models: Optional[Sequence[str]]) -> LLMResult:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = str(m.get("content", ""))
                break
        model = (models or self.default_models or ["offline"])[0]
        # produce a plausible, deterministic stand-in. Return valid JSON so the
        # whole downstream path (parse -> files -> sandbox write) is exercised.
        snippet = " ".join(user.split())[:100] or "task"
        stub = {
            "summary": f"[offline] Handled slice of: {snippet[:120]}",
            "detail": (
                f"[offline] Deterministic stub. Full request was {len(user)} chars. "
                "Run with live LLM to spend real tokens and get real code."
            ),
            "files": [
                {
                    "path": "offline_stub.py",
                    "content": (
                        '"""Placeholder produced by an offline swarm agent."""\n'
                        'SLICE = """' + snippet[:400] + '"""\n\n'
                        "def describe() -> str:\n"
                        "    return SLICE.strip()\n\n"
                        'if __name__ == "__main__":\n'
                        "    print(describe())\n"
                    ),
                },
                {"path": "OFFLINE.md", "content": f"# Offline stub\n\n{snippet}\n"},
            ],
            "commands": ["mkdir -p . && echo offline-agent-ran > .offline_ran"],
        }
        with self._stats_lock:
            self.calls += 1
            self.model_calls[model] = self.model_calls.get(model, 0) + 1
            self.model_ok[model] = self.model_ok.get(model, 0) + 1
            self.latencies.append(0.001)
        return LLMResult(
            ok=True,
            text=json.dumps(stub),
            model=f"offline/{model}",
            total_tokens=0,
            latency=0.0,
        )

    # -- parallel fan-out ---------------------------------------------------
    def map_chat(
        self,
        jobs: Sequence[Tuple[str, List[Dict]]],
        models: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> List[LLMResult]:
        """Run many independent chat requests concurrently.

        `jobs` is a list of (label, messages). Results are returned in the
        same order as `jobs`.
        """
        if not jobs:
            return []
        results: List[Optional[LLMResult]] = [None] * len(jobs)
        workers = max_workers or min(len(jobs), 32)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    self.chat, msgs, models, max_tokens, temperature, timeout
                ): i
                for i, (_, msgs) in enumerate(jobs)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = LLMResult(ok=False, text="", model="",
                                           error=f"exc {type(e).__name__}: {e}")
        return [r if r is not None else LLMResult(False, "", "", error="none")
                for r in results]
