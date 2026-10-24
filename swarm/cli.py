"""
Command-line entry point for the agent swarm.

Examples
--------
    python -m swarm run                 # default sized run
    python -m swarm run --squads 60     # 300 agents (60 x 5), sequential
    python -m swarm run --squads 4 --offline   # pipeline test, no tokens
    python -m swarm run --squads 4 --no-sandbox # no sandbox writes
    python -m swarm probe               # verify NVIDIA + Daytona reachability
    python -m swarm models              # list usable NVIDIA models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _print(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def cmd_probe(args) -> int:
    from .config import NVIDIA_API_KEY, DAYTONA_API_KEY, DAYTONA_SANDBOX_NAME
    from .llm import LLMClient
    from .sandbox import Sandbox

    _print("=" * 70)
    _print("PROBE")
    _print("=" * 70)

    # NVIDIA
    _print("\n[NVIDIA NIM]")
    llm = LLMClient(api_key=NVIDIA_API_KEY, live=True, timeout=45)
    res = llm.chat_text(
        "Reply with exactly: OK",
        system="You are a connectivity check.",
        models=[
            "poolside/laguna-xs-2.1",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
        ],
        max_tokens=400,
    )
    _print(f"  chat -> {res[:120]!r}" if res else "  chat -> FAILED")
    _print(f"  stats: {json.dumps(llm.stats_snapshot()['model_calls'])}")

    # Daytona
    _print(f"\n[Daytona] sandbox={DAYTONA_SANDBOX_NAME}")
    sb = Sandbox(enabled=True)
    probe = sb.probe()
    _print(f"  reachable={probe['reachable']}")
    _print(f"  {probe['output'][:200]}")
    if probe["reachable"]:
        _print(f"  exec_count={sb.exec_count}")
        # confirm python3 / bootstrap state
        r = sb.exec("command -v python3 && python3 --version || echo 'python3 not installed yet'")
        _print(f"  python3: {r.output.strip()[:120]}")

    _print("")
    return 0 if (res and probe["reachable"]) else 1


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def cmd_models(args) -> int:
    import urllib.request
    from .config import NVIDIA_API_KEY, NVIDIA_MODELS_URL

    req = urllib.request.Request(
        NVIDIA_MODELS_URL, headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        _print(f"failed to list models: {e}")
        return 1
    ids = [m["id"] for m in data.get("data", [])]
    _print(f"{len(ids)} models available on this key:\n")
    for i in sorted(ids):
        _print(f"  {i}")
    return 0


# ---------------------------------------------------------------------------
# resume helper
# ---------------------------------------------------------------------------
def _read_state(path: str):
    """Read a checkpoint (or exported report) file into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resume(orch, checkpoint_path: str):
    """Restore a previous run and return the squad index to continue from.

    Returns None when the file cannot be used, so the caller can abort
    instead of silently restarting from squad 0.
    """
    try:
        state = _read_state(checkpoint_path)
    except Exception as e:
        _print(f"! could not read checkpoint {checkpoint_path}: {e}")
        return None
    if not isinstance(state, dict):
        _print(f"! {checkpoint_path} is not a swarm checkpoint/report")
        return None

    shape = "checkpoint" if ("version" in state or
                              "completed_squads" in state) else "report"
    try:
        completed = orch.restore(state)
    except Exception as e:
        _print(f"! {checkpoint_path} could not be restored: "
               f"{type(e).__name__}: {e}")
        return None
    if completed == 0 and shape == "report" and not state.get("metrics"):
        _print(f"! {checkpoint_path} carries no squad progress")
        return None

    _print(f"resuming from {shape} {checkpoint_path}: "
           f"{completed}/{orch.scale.squads} squad(s) already done, "
           f"run_dir={orch.run_dir}")
    if completed >= orch.scale.squads:
        _print("! nothing left to do: every squad in this run is already "
               "complete (raise --squads to continue past it)")
    return min(completed, orch.scale.squads)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    from .config import Scale, Settings, SETTINGS
    from .orchestrator import Orchestrator

    base = Scale()
    scale = Scale(
        squads=args.squads if args.squads is not None else base.squads,
        agents_per_squad=(
            args.agents_per_squad if args.agents_per_squad is not None
            else base.agents_per_squad
        ),
        curators=args.curators if args.curators is not None else base.curators,
        concurrency=(
            args.concurrency if args.concurrency is not None else base.concurrency
        ),
        llm_timeout=(
            args.llm_timeout if args.llm_timeout is not None else base.llm_timeout
        ),
    )

    settings = Settings(
        scale=scale,
        goal=args.goal or SETTINGS.goal,
        use_sandbox=not args.no_sandbox,
        live_llm=not args.offline,
        verbose=not args.quiet,
        board_context_limit=args.board_context,
        peer_reviews_per_work=args.reviews,
        discussants=args.discussants,
        max_total_tokens=args.budget,
        run_prefix=args.run_prefix,
    )

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "out")
    out_dir = os.path.abspath(out_dir)
    orch = Orchestrator(settings, out_dir=out_dir)

    # Resume: pick up where a previous run left off (same sandbox folder).
    start_squad = 0
    if args.resume_from:
        resumed = _resume(orch, args.resume_from)
        if resumed is None:
            return 2
        start_squad = resumed

    t0 = time.time()
    interrupted = False
    try:
        orch.run(start_squad=start_squad)
    except KeyboardInterrupt:
        interrupted = True
        _print("\n! interrupted -> writing partial report")
        orch.stop()
    finally:
        # Reports are written even if the run blew up: losing the artefact
        # because of an exception would waste the whole run.
        _write_outputs(orch, out_dir, interrupted, t0)
    return 130 if interrupted else 0


def _write_outputs(orch, out_dir: str, interrupted: bool, t0: float) -> None:
    """Persist the JSON/txt/HTML artefacts for a finished (or stopped) run."""
    try:
        report = orch.final_report()
        _print(report)
    except Exception as e:
        _print(f"! final report failed: {type(e).__name__}: {e}")
        report = ""

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"swarm-report-{stamp}.json")
    txt_path = os.path.join(out_dir, f"swarm-report-{stamp}.txt")
    html_path = os.path.join(out_dir, f"swarm-dashboard-{stamp}.html")

    try:
        exported = orch.export_report()
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(exported, fh, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(report)
    except Exception as e:
        _print(f"! could not write reports: {type(e).__name__}: {e}")

    html_written = False
    try:
        from .dashboard import write as write_dashboard
        write_dashboard(exported, html_path)
        html_written = True
    except Exception as e:
        _print(f"! dashboard failed: {e}")

    _print(f"\nwall time : {time.time() - t0:.1f}s")
    _print(f"json report: {json_path}")
    _print(f"txt report : {txt_path}")
    _print(f"dashboard  : {html_path if html_written else '(not written)'}")
    _print(f"checkpoint : {orch.checkpoint_path}")
    if orch.log.path:
        _print(f"event log  : {orch.log.path}")


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from .config import Scale, Settings

    base = Scale()
    defaults = Settings()

    p = argparse.ArgumentParser(
        prog="swarm",
        description="A large multi-agent swarm: sequential 5-agent role squads, "
                    "a 15-AI curator council that authors and votes on prompts, "
                    "a shared discussion board, peer review, all executing in one "
                    "Daytona sandbox.",
    )
    sub = p.add_subparsers(dest="cmd")

    # run - numeric defaults come from swarm/config.py so there is a single
    # source of truth (argparse, config and README can never drift apart).
    r = sub.add_parser("run", help="run the swarm")
    r.add_argument("--squads", type=int, default=None,
                   help=f"number of sequential 5-agent squads (default {base.squads})")
    r.add_argument("--agents-per-squad", type=int, default=None,
                   help=f"agents per squad (default {base.agents_per_squad}, per design)")
    r.add_argument("--curators", type=int, default=None,
                   help=f"number of curator AIs that author+vote on prompts (default {base.curators})")
    r.add_argument("--concurrency", type=int, default=None,
                   help=f"max concurrent LLM calls (default {base.concurrency})")
    r.add_argument("--llm-timeout", type=int, default=None,
                   help=f"per-LLM-call timeout seconds (default {base.llm_timeout})")
    r.add_argument("--goal", type=str, default=None,
                   help="the project goal the swarm builds toward")
    r.add_argument("--offline", action="store_true",
                   help="use deterministic stubs instead of live LLM calls")
    r.add_argument("--no-sandbox", action="store_true",
                   help="skip all Daytona sandbox interaction")
    r.add_argument("--board-context", type=int, default=defaults.board_context_limit,
                   help=f"how many discussion posts each agent sees (default {defaults.board_context_limit})")
    r.add_argument("--reviews", type=int, default=defaults.peer_reviews_per_work,
                   help=f"peer reviews per work unit (default {defaults.peer_reviews_per_work})")
    r.add_argument("--discussants", type=int, default=defaults.discussants,
                   help=f"agents commenting publicly after each squad (default {defaults.discussants})")
    r.add_argument("--budget", type=int, default=defaults.max_total_tokens,
                   help="stop cleanly after ~N total tokens (0 = unlimited)")
    r.add_argument("--resume-from", type=str, default=None,
                   help="path to out/_checkpoint.json (or a swarm-report JSON) to continue")
    r.add_argument("--run-prefix", type=str, default=defaults.run_prefix,
                   help="sandbox workspace folder prefix")
    r.add_argument("--out", type=str, default=None, help="output dir for reports")
    r.add_argument("--quiet", action="store_true", help="suppress live transcript")
    r.set_defaults(func=cmd_run)

    # probe
    pr = sub.add_parser("probe", help="verify NVIDIA + Daytona connectivity")
    pr.set_defaults(func=cmd_probe)

    # models
    m = sub.add_parser("models", help="list NVIDIA models on this key")
    m.set_defaults(func=cmd_models)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
