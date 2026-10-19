"""
Daytona sandbox client for the `agent_swarm` sandbox.

Everything the swarm builds is written and executed inside a single shared
Daytona sandbox (`agent_swarm`) so every agent collaborates in the same
filesystem and shell.

Verified working primitives (discovered against the live API):

* Shell exec:  POST {toolbox}/{sandboxId}/process/execute   {"command": "..."}
               -> {"exitCode": N, "result": "<stdout+stderr>"}
* Auth:        Authorization: Bearer <DAYTONA_API_KEY>  (proxy requires it)
* File write:  base64-encode content and pipe through the shell (robust for
               any bytes / any quoting, no FS endpoint needed).
* The sandbox is Ubuntu 22.04 as root; bash + tar present; python/node/gits
   are NOT preinstalled, so `bootstrap()` installs what agent code needs.
"""
from __future__ import annotations

import base64
import json
import shlex
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import DAYTONA_API_BASE, DAYTONA_API_KEY, DAYTONA_SANDBOX_ID, DAYTONA_SANDBOX_NAME, DAYTONA_TOOLBOX_BASE, SANDBOX_WORKSPACE


@dataclass
class ExecResult:
    exit_code: int
    output: str
    ok: bool
    error: str = ""

    @property
    def success(self) -> bool:
        return self.ok and self.exit_code == 0


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """A thin, thread-safe client to one Daytona sandbox."""

    def __init__(
        self,
        sandbox_id: str = DAYTONA_SANDBOX_ID,
        api_key: str = DAYTONA_API_KEY,
        workspace: str = SANDBOX_WORKSPACE,
        timeout: int = 180,
        enabled: bool = True,
    ):
        self.sandbox_id = sandbox_id
        self.api_key = api_key
        self.workspace = workspace
        self.timeout = timeout
        self.enabled = enabled
        self._exec_url = f"{DAYTONA_TOOLBOX_BASE}/{sandbox_id}/process/execute"
        self._lock = threading.Lock()
        self._bootstrapped = False
        self.exec_count = 0
        self.history: List[Tuple[str, int]] = []

    # -- low level ----------------------------------------------------------
    def _post(self, url: str, payload: Dict, timeout: int) -> Tuple[int, Dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
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
                raw = resp.read().decode("utf-8", "replace")
                try:
                    return resp.status, json.loads(raw)
                except Exception:
                    return resp.status, {"result": raw}
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            return e.code, {"error": detail or str(e)}
        except Exception as e:
            return -1, {"error": f"{type(e).__name__}: {e}"}

    def exec(self, command: str, timeout: Optional[int] = None) -> ExecResult:
        """Run a shell command as root in the sandbox."""
        if not self.enabled:
            return ExecResult(exit_code=0, output=f"[dry-run] {command}", ok=True)

        timeout = timeout or self.timeout
        for attempt in range(3):
            code, body = self._post(
                self._exec_url, {"command": command}, timeout
            )
            if code == 200:
                out = str(body.get("result", ""))
                ec = body.get("exitCode", 0)
                if not isinstance(ec, int):
                    ec = 0
                with self._lock:
                    self.exec_count += 1
                    self.history.append((command[:80], ec))
                return ExecResult(exit_code=ec, output=out, ok=True)
            if code in (-1, 502, 503, 504, 429):
                time.sleep(1.5 * (attempt + 1))
                continue
            return ExecResult(
                exit_code=code if isinstance(code, int) else -1,
                output="",
                ok=False,
                error=str(body.get("error", ""))[:500],
            )
        return ExecResult(exit_code=-1, output="", ok=False, error="unreachable")

    def sh(self, command: str, timeout: Optional[int] = None) -> str:
        """Run a command and return its combined output as a string."""
        r = self.exec(command, timeout)
        return (r.output or "") + ("" if r.ok else f"\n[error] {r.error}")

    # -- files --------------------------------------------------------------
    @staticmethod
    def _write_cmd(path: str, content: str) -> str:
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        return (
            "mkdir -p $(dirname {p}) && printf '%s' '{b}' | base64 -d > {p} && "
            "printf '%s\\n' \"$(wc -c < {p})\""
        ).format(p=shlex.quote(path), b=b64)

    def write_file(self, path: str, content: str) -> ExecResult:
        """Write text content to a file in the sandbox (base64-safe)."""
        return self.exec(self._write_cmd(path, content), timeout=60)

    def write_files(self, files: Dict[str, str]) -> Dict[str, bool]:
        """Write many files. Returns path -> success."""
        results = {}
        for path, content in files.items():
            results[path] = self.write_file(path, content).success
        return results

    def read_file(self, path: str, max_bytes: int = 200_000) -> str:
        r = self.exec(f"cat {shlex.quote(path)} 2>/dev/null || true", timeout=60)
        return r.output[:max_bytes]

    def append_file(self, path: str, content: str) -> ExecResult:
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        return self.exec(
            f"mkdir -p $(dirname {shlex.quote(path)}) && printf '%s' '{b64}' | base64 -d >> {shlex.quote(path)}",
            timeout=60,
        )

    def list_dir(self, path: Optional[str] = None, depth: int = 3) -> str:
        p = path or self.workspace
        return self.sh(
            f"find {shlex.quote(p)} -maxdepth {depth} -not -path '*/.git/*' 2>/dev/null | sort | head -200",
            timeout=60,
        )

    def run_python(self, code: str, timeout: Optional[int] = None) -> ExecResult:
        """Run a python snippet in the sandbox (requires bootstrap())."""
        b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        cmd = (
            "mkdir -p /root/swarm/_tmp && printf '%s' '{b}' | base64 -d > /root/swarm/_tmp/_snippet.py && "
            "python3 /root/swarm/_tmp/_snippet.py"
        ).format(b=b64)
        return self.exec(cmd, timeout=timeout or self.timeout)

    def run_script(self, path: str, timeout: Optional[int] = None) -> ExecResult:
        """Execute a script file already present in the sandbox."""
        return self.exec(f"python3 {shlex.quote(path)}", timeout=timeout or self.timeout)

    # -- lifecycle ----------------------------------------------------------
    def probe(self) -> Dict:
        """Confirm reachability and report basics."""
        r = self.exec("echo OK; whoami; uname -sr; python3 --version 2>&1 | head -1")
        return {
            "reachable": r.ok and "OK" in r.output,
            "output": r.output.strip(),
            "sandbox_id": self.sandbox_id,
            "workspace": self.workspace,
        }

    def ensure_dirs(self, *subpaths: str) -> None:
        parts = " ".join(shlex.quote(p) for p in subpaths)
        self.exec(f"mkdir -p {parts}", timeout=60)

    def bootstrap(self, force: bool = False) -> bool:
        """Install the runtimes agent code needs (python3, git, etc.).

        The base sandbox ships with bash + tar only. This is idempotent and
        memoized so we pay the apt cost at most once per run.
        """
        if self._bootstrapped and not force:
            return True
        if not self.enabled:
            self._bootstrapped = True
            return True

        self.ensure_dirs(self.workspace, "/root/swarm/_tmp", "/root/swarm/bin")
        # Only install if python3 is actually missing.
        check = self.exec("command -v python3 >/dev/null 2>&1 && echo HAVE || echo NEED", timeout=60)
        if "HAVE" in check.output:
            self._bootstrapped = True
            return True

        self.exec("export DEBIAN_FRONTEND=noninteractive", timeout=30)
        # apt can be slow; give it a generous budget and tolerate failure.
        cmd = (
            "export DEBIAN_FRONTEND=noninteractive; "
            "(apt-get update -y >/root/swarm/apt.log 2>&1 || true); "
            "(apt-get install -y python3 python3-pip git ca-certificates "
            ">>/root/swarm/apt.log 2>&1 || true); "
            "command -v python3 >/dev/null 2>&1 && echo BOOTSTRAP_OK || echo BOOTSTRAP_FAIL"
        )
        r = self.exec(cmd, timeout=600)
        self._bootstrapped = "BOOTSTRAP_OK" in r.output
        return self._bootstrapped

    def summary(self) -> Dict:
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_name": DAYTONA_SANDBOX_NAME,
            "workspace": self.workspace,
            "enabled": self.enabled,
            "exec_count": self.exec_count,
            "bootstrapped": self._bootstrapped,
        }
