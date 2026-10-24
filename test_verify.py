"""
Tests for post-squad verification.

Run:  python -m unittest test_verify -v
      SWARM_LIVE=1 python -m unittest test_verify -v   # + real Daytona calls

The point of these tests is the regression they guard: a test file that merely
DEFINES unittest/pytest cases exits 0 when executed as a script, which used to
be broadcast to the whole swarm as "VERIFIED". Verification must only claim a
pass when the output proves test cases actually ran.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swarm.sandbox import ExecResult, Sandbox  # noqa: E402
from swarm.state import DiscussionBoard, EventLog  # noqa: E402
from swarm.verify import (  # noqa: E402
    _commands_for,
    _is_test_file,
    _ran_real_tests,
    summarise,
    verify_squad_files,
)

LIVE = bool(os.environ.get("SWARM_LIVE"))


class FakeSandbox:
    """Executes nothing: replays canned results keyed on the runner used."""

    def __init__(self, script_out="hello\n", script_code=0,
                 pytest_out="", pytest_code=0, pytest_missing=False,
                 unittest_out="Ran 2 tests in 0.001s\n\nOK\n"):
        self.enabled = True
        self.calls = []
        self.script_out, self.script_code = script_out, script_code
        self.pytest_out, self.pytest_code = pytest_out, pytest_code
        self.pytest_missing = pytest_missing
        self.unittest_out = unittest_out

    def exec(self, command, timeout=None):
        self.calls.append(command)
        if "py_compile" in command:
            if "broken" in command:
                return ExecResult(1, "", False, error="SyntaxError")
            return ExecResult(0, "", True)
        if "pytest" in command:
            if self.pytest_missing:
                return ExecResult(1, "No module named pytest", False)
            return ExecResult(self.pytest_code, self.pytest_out, self.pytest_code == 0)
        if "unittest" in command:
            return ExecResult(0, self.unittest_out, True)
        return ExecResult(self.script_code, self.script_out, self.script_code == 0)


def _verify(sandbox, artifacts, board=None, log=None):
    board = board or DiscussionBoard()
    log = log or EventLog()
    v = verify_squad_files(sandbox, "/root/swarm/workspace/run-test", artifacts,
                           board, log, squad=3, role="test-engineer")
    return v, board


class TestFileDetection(unittest.TestCase):
    def test_test_paths(self):
        self.assertTrue(_is_test_file("tests/test_suite.py"))
        self.assertTrue(_is_test_file("src/test_core.py"))
        self.assertTrue(_is_test_file("test_vote.py"))

    def test_non_test_paths(self):
        self.assertFalse(_is_test_file("src/core.py"))
        self.assertFalse(_is_test_file("ARCHITECTURE.md"))

    def test_commands_prefer_test_runners_for_tests(self):
        labels = [label for label, _ in _commands_for("tests/test_suite.py")]
        self.assertEqual(labels, ["pytest", "unittest", "script"])

    def test_plain_modules_run_as_scripts(self):
        labels = [label for label, _ in _commands_for("src/core.py")]
        self.assertEqual(labels, ["script"])


class EffectiveRunDetection(unittest.TestCase):
    def test_unittest_and_pytest_markers_count(self):
        self.assertTrue(_ran_real_tests("Ran 3 tests in 0.01s\nOK"))
        self.assertTrue(_ran_real_tests("4 passed in 0.20s"))
        self.assertTrue(_ran_real_tests("collected 2 items\n1 failed"))
        self.assertTrue(_ran_real_tests("2 errors"))

    def test_silent_or_irrelevant_output_does_not_count(self):
        self.assertFalse(_ran_real_tests(""))
        self.assertFalse(_ran_real_tests("imported ok\n"))
        self.assertFalse(_ran_real_tests("help output for pytest"))


class VerificationChain(unittest.TestCase):
    """The full verify_squad_files decision tree, with no network."""

    def test_passing_test_file_is_verified(self):
        sb = FakeSandbox(pytest_out="2 passed in 0.05s", pytest_code=0)
        v, board = _verify(sb, {"tests/test_suite.py": "print('x')"})
        self.assertTrue(v.ran)
        self.assertTrue(v.passed)
        self.assertTrue(any("VERIFIED" in p.text for p in board.all()))

    def test_defines_only_test_file_is_not_verified(self):
        # the regression: every runner exits 0 but no test case ever ran
        sb = FakeSandbox(pytest_out="", pytest_code=0,
                         unittest_out="", script_out="")
        v, board = _verify(sb, {"tests/test_suite.py": "class T: pass"})
        self.assertTrue(v.ran)
        self.assertFalse(v.passed)
        self.assertEqual(v.error, "no-tests-executed")
        self.assertTrue(any("VERIFY FAILED" in p.text for p in board.all()))

    def test_falls_back_when_pytest_is_missing(self):
        sb = FakeSandbox(pytest_missing=True,
                         unittest_out="Ran 2 tests in 0.001s\n\nOK\n")
        v, _ = _verify(sb, {"tests/test_suite.py": "pass"})
        self.assertTrue(v.passed)
        self.assertIn("unittest", "".join(sb.calls))

    def test_failing_tests_report_failure(self):
        sb = FakeSandbox(pytest_out="1 failed, 1 passed in 0.02s", pytest_code=1)
        v, _ = _verify(sb, {"tests/test_suite.py": "pass"})
        self.assertTrue(v.ran)
        self.assertFalse(v.passed)

    def test_syntax_error_is_reported_immediately(self):
        sb = FakeSandbox()
        v, board = _verify(sb, {"broken.py": "def oops(:\n    pass\n"})
        self.assertTrue(v.ran)
        self.assertFalse(v.passed)
        self.assertEqual(v.error, "syntax-error")
        self.assertTrue(any("VERIFY FAILED" in p.text for p in board.all()))

    def test_plain_script_that_exits_zero_passes(self):
        sb = FakeSandbox(script_out="all good\n", script_code=0)
        v, _ = _verify(sb, {"src/core.py": "print('all good')"})
        self.assertTrue(v.ran)
        self.assertTrue(v.passed)

    def test_failing_script_reports_failure(self):
        sb = FakeSandbox(script_out="boom", script_code=3)
        v, _ = _verify(sb, {"src/core.py": "import sys; sys.exit(3)"})
        self.assertFalse(v.passed)

    def test_non_python_artefacts_are_skipped(self):
        sb = FakeSandbox()
        v, board = _verify(sb, {"ARCHITECTURE.md": "# hi"})
        self.assertFalse(v.ran)
        self.assertEqual(board.count(), 0)

    def test_disabled_sandbox_runs_nothing(self):
        sb = FakeSandbox()
        sb.enabled = False
        v, _ = _verify(sb, {"src/core.py": "pass"})
        self.assertFalse(v.ran)

    def test_summarise_counts_passes(self):
        from swarm.verify import Verification
        self.assertEqual(
            summarise([Verification(True, passed=True),
                       Verification(True, passed=False),
                       Verification(False)]),
            "1/2 executions passed in sandbox",
        )
        self.assertEqual(summarise([]), "no executable verification ran")


@unittest.skipUnless(LIVE, "set SWARM_LIVE=1 to reach the Daytona sandbox")
class LiveSandbox(unittest.TestCase):
    """End-to-end against the real sandbox (writes under the run dir)."""

    def test_real_execution(self):
        sb = Sandbox(enabled=True, timeout=120)
        probe = sb.probe()
        self.assertTrue(probe["reachable"], probe)

        run_dir = "/root/swarm/workspace/_verify_test"
        sb.ensure_dirs(run_dir)
        board, log = DiscussionBoard(), EventLog()

        good = {"ok_script.py": "print('verify-ok')\nprint(2+2)\n"}
        sb.write_files({f"{run_dir}/{k}": v for k, v in good.items()})
        v, _ = _verify_with(sb, run_dir, good, board, log)
        self.assertTrue(v.ran)
        self.assertTrue(v.passed)

        bad = {"broken.py": "def oops(:\n    pass\n"}
        sb.write_files({f"{run_dir}/{k}": v for k, v in bad.items()})
        v2, _ = _verify_with(sb, run_dir, bad, board, log)
        self.assertFalse(v2.passed)
        self.assertEqual(v2.error, "syntax-error")

        fails = {"fails.py": "import sys; print('about to fail'); sys.exit(3)"}
        sb.write_files({f"{run_dir}/{k}": v for k, v in fails.items()})
        v3, _ = _verify_with(sb, run_dir, fails, board, log)
        self.assertTrue(v3.ran)
        self.assertFalse(v3.passed)


def _verify_with(sb, run_dir, artifacts, board, log):
    return (verify_squad_files(sb, run_dir, artifacts, board, log,
                               squad=99, role="tester"), board)


if __name__ == "__main__":
    unittest.main(verbosity=2)
