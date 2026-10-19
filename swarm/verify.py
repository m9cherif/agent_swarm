"""
Post-squad verification: actually RUN what the agents wrote.

After each squad persists files to the Daytona sandbox, this module finds a
runnable entry point (python files, tests) and executes it in the sandbox,
then broadcasts the REAL pass/fail result to the discussion board. Later
squads see concrete execution feedback instead of only prose summaries.

That closes the loop: agents don't just claim they wrote code - the swarm
proves it runs and tells everyone.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sandbox import Sandbox
from .state import DiscussionBoard, EventLog


@dataclass
class Verification:
    ran: bool
    target: str = ""
    exit_code: int = 0
    passed: bool = False
    output: str = ""
    files_checked: int = 0
    error: str = ""


# cheap, high-signal probes we try in order
_RUNNERS = [
    # (label, command template)  -- {file} is substituted with a sandbox path
    ("unittest", "python3 -m pytest -q {file} 2>&1 || python3 {file} 2>&1"),
    ("script", "python3 {file} 2>&1"),
]


def verify_squad_files(
    sandbox: Sandbox,
    run_dir: str,
    artifacts: Dict[str, str],
    board: DiscussionBoard,
    log: EventLog,
    squad: int,
    role: str,
) -> Verification:
    """Run agent-written Python in the sandbox and broadcast the result."""
    if not sandbox.enabled or not artifacts:
        return Verification(ran=False)

    py_files = sorted(
        rel for rel in artifacts if rel.endswith(".py") and not rel.startswith("_")
    )
    if not py_files:
        # nothing executable: still record what exists
        return Verification(ran=False, files_checked=len(artifacts))

    # Prefer test-looking files, else the first module.
    test_files = [f for f in py_files if "test" in f.lower()]
    order = test_files + [f for f in py_files if f not in test_files]

    last = Verification(ran=False, files_checked=len(artifacts))
    # Bound the cost tightly: verification runs on the critical path alongside
    # review/discussion, so cap it at 2 files with short timeouts.
    for rel in order[:2]:
        full = f"{run_dir}/{rel}"
        # Compile-check first: cheap syntax gate.
        comp = sandbox.exec(f"python3 -m py_compile {shlex.quote(full)}", timeout=45)
        if not comp.success:
            last = Verification(
                ran=True, target=rel, exit_code=comp.exit_code, passed=False,
                output=(comp.output or comp.error or "")[:800],
                files_checked=len(artifacts), error="syntax-error",
            )
            _broadcast(board, log, squad, role, last, rel)
            return last  # a broken file is worth reporting immediately

        # Execute it (self-running scripts / tests). Timeout guards hangs.
        run = sandbox.exec(f"cd {shlex.quote(run_dir)} && python3 {shlex.quote(rel)}",
                           timeout=60)
        passed = run.success
        last = Verification(
            ran=True, target=rel, exit_code=run.exit_code, passed=passed,
            output=(run.output or "")[:800], files_checked=len(artifacts),
        )
        if passed:
            _broadcast(board, log, squad, role, last, rel)
            return last  # stop at the first clean execution

    _broadcast(board, log, squad, role, last, last.target)
    return last


def _broadcast(board: DiscussionBoard, log: EventLog, squad: int, role: str,
               v: Verification, rel: str) -> None:
    if not v.ran:
        return
    if v.passed:
        text = f"VERIFIED: ran `{rel}` in sandbox - exited 0. Real execution confirmed."
        kind = "verification"
    else:
        snippet = " ".join((v.output or v.error or "").split())[:160]
        text = f"VERIFY FAILED: `{rel}` exit {v.exit_code}. {snippet}"
        kind = "verification"
    board.post(squad=squad, role=role, author="verifier", kind=kind, text=text,
               detail=v.output)
    log.emit("verify", text, squad=squad, target=rel, passed=v.passed,
             exit_code=v.exit_code)


def summarise(verifications: List[Verification]) -> str:
    ran = [v for v in verifications if v.ran]
    if not ran:
        return "no executable verification ran"
    ok = sum(1 for v in ran if v.passed)
    return f"{ok}/{len(ran)} executions passed in sandbox"
