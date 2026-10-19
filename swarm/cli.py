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
def _resume(orch, checkpoint_path: str) -> int:
    """Load a previous run's checkpoint and return the next squad index.

    Also re-points the orchestrator at the SAME sandbox folder so artefacts
    accumulate in one place instead of scattering across dated runs.
    """
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as e:
        _print(f"! could not read checkpoint {checkpoint_path}: {e}")
        return 0

    completed = int(state.get("completed_squads", 0))
    run_dir = state.get("run_dir")
    if run_dir:
        orch.run_dir = run_dir
    # keep the blackboard/board counts for reporting continuity
    _print(f"resuming from checkpoint: {completed} squad(s) already done "
           f"(run_dir={run_dir})")
    return max(0, min(completed, orch.scale.squads))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    from .config import Scale, Settings, SETTINGS
    from .orchestrator import Orchestrator

    scale = Scale(
        squads=args.squads,
        agents_per_squad=args.agents_per_squad,
        curators=args.curators,
        concurrency=args.concurrency,
        llm_timeout=args.llm_timeout,
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

    orch = Orchestrator(settings)

    # Resume: pick up where a previous run left off (same sandbox folder).
    start_squad = 0
    if args.resume_from:
        start_squad = _resume(orch, args.resume_from)

    t0 = time.time()
    try:
        orch.run(start_squad=start_squad)
    except KeyboardInterrupt:
        _print("\n! interrupted -> writing partial report")
        orch.stop()

    report = orch.final_report()
    _print(report)

    # persist exports locally
    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "out")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"swarm-report-{stamp}.json")
    txt_path = os.path.join(out_dir, f"swarm-report-{stamp}.txt")
    html_path = os.path.join(out_dir, f"swarm-dashboard-{stamp}.html")
    exported = orch.export_report()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(exported, fh, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    try:
        from .dashboard import write as write_dashboard
        write_dashboard(exported, html_path)
    except Exception as e:
        _print(f"! dashboard failed: {e}")

    _print(f"\nwall time : {time.time() - t0:.1f}s")
    _print(f"json report: {json_path}")
    _print(f"txt report : {txt_path}")
    _print(f"dashboard  : {html_path}")
    return 0


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarm",
        description="A large multi-agent swarm: sequential 5-agent role squads, "
                    "a 15-AI curator council that authors and votes on prompts, "
                    "a shared discussion board, peer review, all executing in one "
                    "Daytona sandbox.",
    )
    sub = p.add_subparsers(dest="cmd")

    # run
    r = sub.add_parser("run", help="run the swarm")
    r.add_argument("--squads", type=int, default=12,
                   help="number of sequential 5-agent squads (default 12)")
    r.add_argument("--agents-per-squad", type=int, default=5,
                   help="agents per squad (default 5, per design)")
    r.add_argument("--curators", type=int, default=15,
                   help="number of curator AIs that author+vote on prompts (default 15)")
    r.add_argument("--concurrency", type=int, default=24,
                   help="max concurrent LLM calls (default 24)")
    r.add_argument("--llm-timeout", type=int, default=45,
                   help="per-LLM-call timeout seconds (default 45)")
    r.add_argument("--goal", type=str, default=None,
                   help="the project goal the swarm builds toward")
    r.add_argument("--offline", action="store_true",
                   help="use deterministic stubs instead of live LLM calls")
    r.add_argument("--no-sandbox", action="store_true",
                   help="skip all Daytona sandbox interaction")
    r.add_argument("--board-context", type=int, default=40,
                   help="how many discussion posts each agent sees (default 40)")
    r.add_argument("--reviews", type=int, default=2,
                   help="peer reviews per work unit (default 2)")
    r.add_argument("--discussants", type=int, default=3,
                   help="agents commenting publicly after each squad (default 3)")
    r.add_argument("--budget", type=int, default=0,
                   help="stop cleanly after ~N total tokens (0 = unlimited)")
    r.add_argument("--resume-from", type=str, default=None,
                   help="path to a previous run's _checkpoint.json to continue")
    r.add_argument("--run-prefix", type=str, default="run",
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
