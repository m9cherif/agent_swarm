"""
Worker agents.

Each squad has exactly 5 worker agents sharing ONE role. They receive the
prompt the curator council dispatched, then run concurrently. An agent:

  * reads the shared blackboard + discussion board for context,
  * asks the LLM to decide what IT specifically produces (a sub-task slice),
  * writes real files into the Daytona sandbox when the role calls for it,
  * posts a SHORT digest of its work to the public discussion board so every
    other agent can see a less-detailed description of what it did,
  * returns a work record the peer-review stage scores.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import WORKER_MODEL_CHAIN, Role, Scale
from .llm import LLMClient
from .sandbox import Sandbox
from .state import Blackboard, DiscussionBoard, EventLog, Metrics


@dataclass
class WorkUnit:
    agent_id: str
    squad: int
    role: str
    ok: bool
    summary: str              # condensed, board-visible description
    detail: str               # full output, kept locally
    artifacts: Dict[str, str] = field(default_factory=dict)  # relpath -> content
    commands: List[str] = field(default_factory=list)        # shell side effects
    score: float = 0.0
    error: str = ""
    tokens: int = 0
    model: str = ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_worker_prompt(
    role: Role,
    dispatched_prompt: str,
    blackboard_ctx: str,
    board_ctx: str,
    goal: str,
    shard_index: int,
    total_shards: int,
    run_dir: str,
) -> Tuple[str, str]:
    """Return (system, user) for one worker agent.

    The council's winning prompt is the core instruction; we add the
    swarm-wide context and slice the squad's work across the 5 agents.
    """
    system = (
        "You are one worker agent in a large collaborative software swarm running inside a "
        "Linux sandbox. You share a role with other agents; you handle ONE slice of it. "
        "Always output STRICT JSON with these keys:\n"
        '{"summary": "<one short sentence, visible to all other agents>",\n'
        ' "detail": "<full explanation of what you did>",\n'
        ' "files": [{"path": "<relative path>", "content": "<full file content>"}],\n'
        ' "commands": ["<shell command to run in the sandbox>"]}\n'
        "Rules:\n"
        "- Only include files your slice actually owns. Paths are relative to the project "
        f"root `{run_dir}` (the swarm will prefix it). Use REAL filenames with an extension "
        "(e.g. `src/core.py`, `ARCHITECTURE.md`) - never placeholders like `<path>`.\n"
        "- Real content or real code only, never a description of code.\n"
        "- Keep file contents tight and complete; prefer a few well-formed files over "
        "many partial ones.\n"
        "- Keep 'summary' to ONE sentence under 200 characters; it is broadcast to everyone.\n"
        "- Output ONLY the JSON object."
    )
    user = (
        f"PROJECT GOAL:\n{goal}\n\n"
        f"YOUR ROLE: {role.name} - {role.directive}\n"
        f"REQUIRED DELIVERABLE: {role.deliverable}\n\n"
        f"THE PROMPT SELECTED BY THE CURATOR COUNCIL:\n{dispatched_prompt}\n\n"
        f"WHAT EARLIER SQUADS BUILT (blackboard):\n{blackboard_ctx}\n\n"
        f"RECENT PUBLIC DISCUSSION:\n{board_ctx}\n\n"
        f"You are worker {shard_index + 1} of {total_shards} for this squad.\n"
        "Focus on your slice (worker 1 = structure/skeleton, worker 2 = core logic, "
        "worker 3 = integration/glue, worker 4 = tests/validation, worker 5 = polish/"
        "docs/fallback). Coordinate through the shared context above. Produce the JSON."
    )
    return system, user


def _parse_json_reply(text: str) -> Dict:
    """Best-effort extraction of a JSON object from a model reply."""
    if not text:
        return {}
    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # first balanced brace
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return {}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        agent_id: str,
        squad: int,
        role: Role,
        llm: LLMClient,
        sandbox: Sandbox,
        board: DiscussionBoard,
        blackboard: Blackboard,
        metrics: Metrics,
        log: EventLog,
        scale: Scale,
        goal: str,
        run_dir: str,
        board_context_limit: int = 40,
    ):
        self.agent_id = agent_id
        self.squad = squad
        self.role = role
        self.llm = llm
        self.sandbox = sandbox
        self.board = board
        self.blackboard = blackboard
        self.metrics = metrics
        self.log = log
        self.scale = scale
        self.goal = goal
        self.run_dir = run_dir
        self.board_context_limit = board_context_limit

    # -- main ---------------------------------------------------------------
    def work(self, dispatched_prompt: str, shard_index: int) -> WorkUnit:
        """Execute one worker's slice of the squad's role."""
        total = self.scale.agents_per_squad
        bb_ctx = self.blackboard.context_block()
        board_ctx = self.board.context_block(self.board_context_limit)

        system, user = _build_worker_prompt(
            self.role, dispatched_prompt, bb_ctx, board_ctx, self.goal,
            shard_index, total, self.run_dir,
        )

        res = self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            models=WORKER_MODEL_CHAIN,
            max_tokens=1600,
            temperature=0.5,
            timeout=self.scale.llm_timeout,
        )

        obj = _parse_json_reply(res.text)
        if not obj and res.ok:
            # model returned prose instead of JSON: treat it as the summary
            obj = {
                "summary": " ".join(res.text.split())[:200],
                "detail": res.text,
                "files": [],
                "commands": [],
            }

        summary = " ".join(str(obj.get("summary", "")).split())[:240] or \
            f"{self.role.name} worker {shard_index + 1} produced its slice"
        detail = str(obj.get("detail", "")) or res.text
        files = self._normalize_files(obj.get("files"))
        commands = [
            str(c) for c in (obj.get("commands") or [])
            if isinstance(c, str) and c.strip()
        ][:5]

        # Side effects: push files + run commands in the shared sandbox.
        executed: List[str] = []
        if self.role.writes_code and files and self.sandbox.enabled:
            for rel, content in files.items():
                full = f"{self.run_dir}/{rel}"
                r = self.sandbox.write_file(full, content)
                if r.success:
                    executed.append(f"wrote {rel}")
            if commands:
                for cmd in commands[:3]:
                    scoped = f"cd {shlex.quote(self.run_dir)} && {cmd}"
                    r = self.sandbox.exec(scoped, timeout=120)
                    executed.append(f"$ {cmd} -> exit {r.exit_code}")
                    if r.output.strip():
                        executed.append(r.output.strip()[:300])

        # Broadcast the condensed digest so every agent can see it.
        self.board.post(
            squad=self.squad, role=self.role.name, author=self.agent_id,
            kind="digest", text=summary, detail=detail,
        )

        work = WorkUnit(
            agent_id=self.agent_id,
            squad=self.squad,
            role=self.role.name,
            ok=bool(res.ok and (summary or files)),
            summary=summary,
            detail=detail,
            artifacts=files,
            commands=executed,
            tokens=res.total_tokens,
            model=res.model,
            error="" if res.ok else res.error,
        )
        self.log.emit(
            "work",
            f"{self.agent_id} finished",
            squad=self.squad, files=list(files.keys()),
            tokens=res.total_tokens, model=res.model, ok=work.ok,
        )
        return work

    # -- helpers ------------------------------------------------------------
    def _normalize_files(self, raw) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, str):
                    out[self._clean_path(str(k), self.role.name)] = v
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    p = item.get("path")
                    c = item.get("content", "")
                    if isinstance(p, str) and isinstance(c, str):
                        out[self._clean_path(p, self.role.name)] = c
        return out

    @staticmethod
    def _clean_path(p: str, role_name: str = "") -> str:
        """Sanitise an agent-supplied path; replace placeholders with a real name.

        Models sometimes return literal placeholders like `<relative path>` or
        `path/to/file`. Those must never become the on-disk filename.
        """
        p = (p or "").strip().lstrip("/")
        # refuse anything that could escape the run dir or the box
        if ".." in p.split("/"):
            p = p.replace("../", "")
        p = re.sub(r"[^A-Za-z0-9._/\-]", "_", p)
        p = re.sub(r"_{2,}", "_", p).strip("/_")

        lowered = p.lower()
        looks_placeholder = (
            not p
            or p.startswith("path")
            or "relative" in lowered
            or "<" in (p or "")
            or lowered in {"file", "filename", "name", "path", "output", "your_file"}
            or "to_file" in lowered
            or "here" == lowered
        )
        if looks_placeholder:
            ext = "md" if not role_name else "md"
            safe_role = re.sub(r"[^a-z0-9]+", "_", (role_name or "agent").lower()).strip("_")
            p = f"notes/{safe_role}.{ext}"
        return p or "notes/untitled.md"


# ---------------------------------------------------------------------------
# Peer review
# ---------------------------------------------------------------------------
def peer_review(
    works: List[WorkUnit],
    llm: LLMClient,
    metrics: Metrics,
    log: EventLog,
    reviewers: int = 3,
) -> List[WorkUnit]:
    """Have peer agents score each work unit 0-10 and stamp reputation.

    Scores are averaged across reviewers and stored on the work unit; the
    metrics ledger accumulates per-agent reputation for the leaderboard.
    """
    from .config import VOTER_MODEL_CHAIN

    if not works:
        return works

    # Reviews run concurrently: for each work unit, `reviewers` judgments.
    jobs = []
    targets: List[WorkUnit] = []
    for w in works:
        for r in range(reviewers):
            reviewer = f"reviewer-{(r % max(1, len(works))) + 1}"
            system = (
                "You are a peer reviewer in a multi-agent software swarm. Score ONE worker's "
                "output from 0 to 10 for quality, correctness and usefulness to the project. "
                "Reply ONLY as JSON: {\"score\": <number>, \"note\": \"<short reason>\"}"
            )
            user = (
                f"ROLE: {w.role}\n"
                f"OUTPUT SUMMARY: {w.summary}\n"
                f"DETAIL: {w.detail[:1500]}\n"
                f"FILES: {', '.join(w.artifacts.keys()) or '(none)'}\n"
                f"SIDE EFFECTS: {' | '.join(w.commands[:4]) or '(none)'}"
            )
            jobs.append((f"{w.agent_id}#{r}", [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]))
            targets.append(w)

    results = llm.map_chat(
        jobs, models=VOTER_MODEL_CHAIN, max_tokens=200, temperature=0.2
    )

    # Aggregate: mean score per work unit.
    buckets: Dict[str, List[float]] = {}
    for w, res in zip(targets, results):
        obj = _parse_json_reply(res.text)
        score = None
        if isinstance(obj, dict) and "score" in obj:
            try:
                score = max(0.0, min(10.0, float(obj["score"])))
            except (TypeError, ValueError):
                score = None
        if score is None:
            m = re.search(r"([0-9]{1,2}(?:\.[0-9]+)?)", res.text or "")
            if m and float(m.group(1)) <= 10:
                score = float(m.group(1))
        if score is None:
            score = 6.0  # neutral default for unparseable reviews
        buckets.setdefault(w.agent_id, []).append(score)

    for w in works:
        scores = buckets.get(w.agent_id, [6.0])
        avg = sum(scores) / len(scores)
        w.score = avg
        metrics.record_agent_score(w.squad, w.agent_id, avg)

    metrics.note_phase("peer_review")
    log.emit("review", "peer review complete",
             scores={w.agent_id: round(w.score, 2) for w in works})
    return works
