"""
Post-squad verification: actually RUN what the agents wrote.

After each squad persists files to the Daytona sandbox, this module finds a
runnable entry point (python files, tests) and executes it in the sandbox,
then broadcasts the REAL pass/fail result to the discussion board. Later
squads see concrete execution feedback instead of only prose summaries.

That closes the loop: agents don't just claim they wrote code - the swarm
proves it runs and tells everyone.

Test files deserve special care: running `tests/test_suite.py` as a plain
script only DEFINES the test cases and exits 0 without executing a single
assertion. That used to be reported as "VERIFIED". We now run tests through
pytest/unittest and only accept the result if the output proves cases ran.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Dict, List

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


# Candidate ways to run an artefact, in preference order. `{file}` is a path
# relative to the run dir, `{dir}` its directory, `{name}` its basename.
_RUNNERS = [
    ("pytest", "python3 -m pytest -q {file} 2>&1"),
    ("unittest", "python3 -m unittest discover -s {dir} -p {name} 2>&1"),
    ("script", "python3 {file} 2>&1"),
]

# output markers proving test cases actually executed
_EFFECTIVE = (
    r"\bRan \d+ tests?\b",
    r"\b\d+ passed\b",
    r"\b\d+ failed\b",
    r"\b\d+ errors?\b",
    r"\b\d+ skipped\b",
    r"\bcollected \d+ item",
)


def _is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1].lower()
    return "test" in name or rel.startswith("tests/")


def _commands_for(rel: str) -> List[tuple]:
    """(label, command) candidates for one artefact, in preference order."""
    quote = shlex.quote
    fields = {
        "file": rel,
        "dir": quote(rel.rsplit("/", 1)[0]) if "/" in rel else ".",
        "name": quote(rel.rsplit("/", 1)[-1]),
    }
    if _is_test_file(rel):
        return [(label, tmpl.format(**fields)) for label, tmpl in _RUNNERS]
    return [("script", f"python3 {quote(rel)} 2>&1")]


def _ran_real_tests(output: str) -> bool:
    text = output or ""
    return any(re.search(p, text, re.I) for p in _EFFECTIVE)


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
    test_files = [f for f in py_files if _is_test_file(f)]
    order = test_files + [f for f in py_files if f not in test_files]

    last = Verification(ran=False, files_checked=len(artifacts))
    # Bound the cost tightly: verification runs on the critical path alongside
    # review/discussion, so cap it at 2 files with short timeouts.
    for rel in order[:2]:
        # Compile-check first: cheap syntax gate.
        comp = sandbox.exec(
            f"cd {shlex.quote(run_dir)} && python3 -m py_compile {shlex.quote(rel)}",
            timeout=45,
        )
        if not comp.success:
            last = Verification(
                ran=True, target=rel, exit_code=comp.exit_code, passed=False,
                output=(comp.output or comp.error or "")[:800],
                files_checked=len(artifacts), error="syntax-error",
            )
            _broadcast(board, log, squad, role, last, rel)
            return last  # a broken file is worth reporting immediately

        for label, cmd in _commands_for(rel):
            run = sandbox.exec(f"cd {shlex.quote(run_dir)} && {cmd}", timeout=60)
            combined = (run.output or "") + (run.error or "")
            if "No module named" in combined:
                continue  # that runner isn't installed here (e.g. pytest)
            if _is_test_file(rel):
                # ANY runner (including the plain-script fallback) must prove
                # test cases executed - exit 0 alone means nothing, since a
                # file that only defines tests also exits 0.
                if not _ran_real_tests(run.output):
                    continue
            last = Verification(
                ran=True, target=rel, exit_code=run.exit_code,
                passed=run.success, output=(run.output or "")[:800],
                files_checked=len(artifacts),
                error="" if run.success else f"{label} exit {run.exit_code}",
            )
            break
        else:
            # every runner fell through: for a test file that means no
            # assertion ever ran - say so instead of claiming a pass.
            if _is_test_file(rel):
                last = Verification(
                    ran=True, target=rel, exit_code=0, passed=False,
                    output="test file executed but no test cases ran",
                    files_checked=len(artifacts), error="no-tests-executed",
                )
            else:
                last = Verification(
                    ran=True, target=rel, exit_code=1, passed=False,
                    output="no runner could execute this artefact",
                    files_checked=len(artifacts), error="no-runner",
                )

        _broadcast(board, log, squad, role, last, last.target)
        if last.passed:
            return last  # stop at the first clean execution

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
