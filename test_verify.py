import sys, os
sys.path.insert(0, r"C:\Users\Labo info\Documents\Projet par défaut")
from swarm.sandbox import Sandbox
from swarm.state import DiscussionBoard, EventLog
from swarm.verify import verify_squad_files

sb = Sandbox(enabled=True, timeout=120)
print("probe:", sb.probe())

run_dir = "/root/swarm/workspace/_verify_test"
sb.ensure_dirs(run_dir)

board = DiscussionBoard()
log = EventLog()

# case 1: valid self-running python
good = {
    "ok_script.py": "print('verify-ok')\nprint(2+2)\n",
    "NOTES.md": "# notes\n",
}
sb.write_files({f"{run_dir}/{k}": v for k, v in good.items()})
v = verify_squad_files(sb, run_dir, good, board, log, squad=99, role="tester")
print("\nCASE GOOD -> ran=%s target=%s exit=%s passed=%s out=%r" % (
    v.ran, v.target, v.exit_code, v.passed, (v.output or "")[:80]))

# case 2: syntax error python
bad = {"broken.py": "def oops(:\n    pass\n"}
sb.write_files({f"{run_dir}/{k}": v for k, v in bad.items()})
v2 = verify_squad_files(sb, run_dir, bad, board, log, squad=99, role="tester")
print("CASE BAD  -> ran=%s target=%s exit=%s passed=%s err=%s out=%r" % (
    v2.ran, v2.target, v2.exit_code, v2.passed, v2.error, (v2.output or "")[:80]))

# case 3: a failing (nonzero exit) script
fail = {"fails.py": "import sys; print('about to fail'); sys.exit(3)\n"}
sb.write_files({f"{run_dir}/{k}": v for k, v in fail.items()})
v3 = verify_squad_files(sb, run_dir, fail, board, log, squad=99, role="tester")
print("CASE FAIL -> ran=%s target=%s exit=%s passed=%s" % (
    v3.ran, v3.target, v3.exit_code, v3.passed))

print("\nBOARD POSTS:")
for p in board.all():
    print("  [%s] %s" % (p.kind, p.text))
