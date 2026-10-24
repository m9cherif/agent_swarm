"""
The orchestrator: runs the whole swarm.

Sequence (repeated once per squad, squads run ONE AFTER ANOTHER):

   ┌──────────────────────────────────────────────────────────────────┐
   │ 1. CURATOR AUTHORING  15 curator AIs each draft a prompt         │
   │ 2. CURATOR VOTING     all curators score all candidate prompts   │
   │ 3. DISPATCH           winning prompt sent to the squad's 5 agents│
   │ 4. EXECUTION          5 worker agents run concurrently, writing  │
   │                       to the shared Daytona sandbox + posting    │
   │                       condensed digests to the discussion board  │
   │ 5. PEER REVIEW        agents score each other's work (0-10)      │
   │ 6. HANDOFF            results pushed to the blackboard, which    │
   │                       the NEXT squad reads as its input          │
   └──────────────────────────────────────────────────────────────────┘

Between squads the blackboard grows, so later squads operate on a richer
picture of the system and refine rather than restart.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .agents import Agent, WorkUnit, peer_review
from .discussion import discussion_round
from .verify import verify_squad_files
from .config import (
    PREMIUM_MODEL_CHAIN,
    SETTINGS,
    Scale,
    Settings,
    role_for_squad,
)
from .curators import CuratorCouncil
from .llm import LLMClient
from .sandbox import Sandbox
from .state import Blackboard, DiscussionBoard, EventLog, Metrics


@dataclass
class SquadReport:
    squad: int
    role: str
    duration: float
    winning_curator: str
    winner_score: float
    prompt_scores: List[float]
    works: List[WorkUnit]
    artifacts: List[str]
    avg_score: float
    discussion_posts: int
    ok: bool
    verification: str = ""
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "squad": self.squad,
            "role": self.role,
            "duration": round(self.duration, 3),
            "winning_curator": self.winning_curator,
            "winner_score": round(self.winner_score, 3),
            "prompt_scores": [round(s, 3) for s in self.prompt_scores],
            "works": [asdict(w) for w in self.works],
            "artifacts": list(self.artifacts),
            "avg_score": round(self.avg_score, 3),
            "discussion_posts": self.discussion_posts,
            "ok": self.ok,
            "verification": self.verification,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SquadReport":
        works = [WorkUnit(**w) for w in (d.get("works") or [])]
        return cls(
            squad=int(d.get("squad", 0)),
            role=str(d.get("role", "")),
            duration=float(d.get("duration", 0.0)),
            winning_curator=str(d.get("winning_curator", "")),
            winner_score=float(d.get("winner_score", 0.0)),
            prompt_scores=[float(s) for s in (d.get("prompt_scores") or [])],
            works=works,
            artifacts=list(d.get("artifacts") or []),
            avg_score=float(d.get("avg_score", 0.0)),
            discussion_posts=int(d.get("discussion_posts", 0)),
            ok=bool(d.get("ok", True)),
            verification=str(d.get("verification", "")),
            error=str(d.get("error", "")),
        )


class Orchestrator:
    def __init__(self, settings: Optional[Settings] = None,
                 out_dir: Optional[str] = None):
        self.settings = settings or SETTINGS
        self.scale: Scale = self.settings.scale

        # Local folder for checkpoints, event logs and reports.
        self.out_dir = os.path.abspath(
            out_dir or os.path.join(os.path.dirname(__file__), "..", "out")
        )
        os.makedirs(self.out_dir, exist_ok=True)

        # run folder inside the sandbox (dated so reruns never collide)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = f"/root/swarm/workspace/{self.settings.run_prefix}-{stamp}"

        # --- observable state ---
        self.board = DiscussionBoard()
        self.blackboard = Blackboard()
        self.log = EventLog(path=os.path.join(self.out_dir, f"events-{stamp}.jsonl"))
        self.metrics = Metrics()

        # --- infrastructure ---
        self.llm = LLMClient(
            default_models=PREMIUM_MODEL_CHAIN,
            max_concurrency=self.scale.concurrency,
            timeout=self.scale.llm_timeout,
            live=self.settings.live_llm,
        )
        self.sandbox = Sandbox(
            enabled=self.settings.use_sandbox,
            timeout=max(180, self.scale.agent_timeout),
        )

        # --- sub-systems ---
        self.council = CuratorCouncil(
            self.llm, self.board, self.metrics, self.log, self.scale
        )

        # Resume point: written locally after every squad (and mirrored into
        # the sandbox when it is enabled) so `--resume-from` always has a file
        # to point at, sandbox or not.
        self.checkpoint_path = os.path.join(self.out_dir, "_checkpoint.json")

        self.reports: List[SquadReport] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ util
    def _log(self, msg: str) -> None:
        if self.settings.verbose:
            print(msg, flush=True)

    def _phase(self, squad: int, total: int, text: str) -> None:
        self._log(f"  [squad {squad + 1}/{total}] {text}")

    # ---------------------------------------------------------------- bootstrap
    def bootstrap(self) -> Dict:
        """Prepare the sandbox and announce the run."""
        self._log("=" * 78)
        self._log("AGENT SWARM  |  bootstrapping")
        self._log("=" * 78)
        self._log(
            f"  agents       : {self.scale.total_agents} "
            f"({self.scale.squads} squads x {self.scale.agents_per_squad} agents, "
            "one role per squad, run sequentially)"
        )
        self._log(f"  curators     : {self.scale.curators} (author + vote on prompts)")
        self._log(f"  goal         : {self.settings.goal[:70]}...")
        self._log(f"  live LLM     : {self.settings.live_llm}")
        self._log(f"  sandbox      : {self.settings.use_sandbox}")

        if self.settings.use_sandbox:
            probe = self.sandbox.probe()
            self._log(f"  sandbox probe: reachable={probe['reachable']}  {probe['output'][:60]}")
            if probe["reachable"]:
                self.log.emit("system", "sandbox reachable", **probe)
                self.sandbox.ensure_dirs(self.run_dir, "/root/swarm/_tmp")
                ok = self.sandbox.bootstrap()
                pytest = self.sandbox.pytest_ready
                self._log(
                    f"  sandbox setup: python3 {'ready' if ok else 'MISSING'}, "
                    f"pytest {'ready' if pytest else 'missing (tests will fall back)'}"
                )
                self.log.emit("system", "bootstrap", python3_ready=ok,
                              pytest_ready=pytest, run_dir=self.run_dir)
            else:
                self.log.emit("system", "sandbox unreachable", **probe)
        else:
            self._log("  sandbox      : disabled (dry run)")

        self.board.post(
            squad=-1, role="system", author="orchestrator", kind="system",
            text=f"Swarm started: {self.scale.total_agents} agents, "
                 f"{self.scale.squads} sequential squads, {self.scale.curators} curators.",
        )
        return {"run_dir": self.run_dir}

    # ------------------------------------------------------------------- run
    def _budget_exceeded(self) -> bool:
        """True when the run has burned through its token ceiling."""
        budget = self.settings.max_total_tokens
        if not budget:
            return False
        if self.llm.total_tokens >= budget:
            self._log(
                f"  ! token budget reached "
                f"({self.llm.total_tokens:,} >= {budget:,}) -> halting pipeline"
            )
            self.log.emit("system", "budget reached",
                          tokens=self.llm.total_tokens, budget=budget)
            return True
        return False

    def run(self, start_squad: int = 0) -> List[SquadReport]:
        total = self.scale.squads
        if start_squad == 0:
            self.bootstrap()
        elif self.settings.use_sandbox:
            # resuming: the folder already exists, just make sure it is there
            self._log(f"  resuming into {self.run_dir} at squad {start_squad + 1}")
            self.sandbox.ensure_dirs(self.run_dir, "/root/swarm/_tmp")

        # `completed` must count squads from the ORIGINAL run, not reports
        # held in memory - a resumed run already has the earlier squads' reports
        # restored, so `len(reports)` alone would double-count them.
        ran = 0
        for i in range(start_squad, total):
            if self._stop.is_set():
                self._log("  ! stop requested, halting pipeline")
                break
            # token budget guard (also re-checked inside the squad so we
            # cannot overshoot by a whole squad's worth of calls)
            if self._budget_exceeded():
                break
            try:
                report = self._run_squad(i, total)
            except Exception as e:
                # One bad squad must not kill a 300-agent run: record it,
                # tell the board, and move on to the next squad.
                self._log(f"  ! squad {i + 1} crashed: {type(e).__name__}: {e}")
                self.log.emit("error", f"squad crashed: {e}", squad=i)
                self.board.post(
                    squad=i, role=role_for_squad(i).name, author="orchestrator",
                    kind="system", text=f"Squad {i + 1} failed: {e}",
                )
                report = SquadReport(
                    squad=i, role=role_for_squad(i).name, duration=0.0,
                    winning_curator="", winner_score=0.0, prompt_scores=[],
                    works=[], artifacts=[], avg_score=0.0, discussion_posts=0,
                    ok=False, verification="squad crashed",
                    error=f"{type(e).__name__}: {e}",
                )
            with self._lock:
                self.reports.append(report)
            ran += 1
            self._checkpoint(i + 1)

        # Always leave a resume point on disk, even for a run that stopped
        # before its first squad completed.
        self._checkpoint(start_squad + ran)
        return list(self.reports)

    # ----------------------------------------------------------- one squad
    def _run_squad(self, squad_index: int, total: int) -> SquadReport:
        role = role_for_squad(squad_index)
        self.metrics.start_squad(squad_index, role.name)
        t0 = time.time()
        self._phase(squad_index, total, f"ROLE = {role.name}")

        # context shared with every agent in this squad
        bb_ctx = self.blackboard.context_block()
        board_ctx = self.board.context_block(self.settings.board_context_limit)

        # -- 1. curator authoring ------------------------------------------
        self._phase(squad_index, total,
                    f"curators: {self.scale.curators} AIs drafting prompts...")
        drafts = self.council.author_prompts(
            squad_index, role.name, role.directive, role.deliverable,
            bb_ctx, board_ctx, self.settings.goal,
        )

        # -- 2. curator voting ---------------------------------------------
        self._phase(squad_index, total, "curators: voting on the candidate prompts...")
        means = self.council.vote_on_prompts(drafts, role.name, squad_index)

        # -- 3. dispatch ----------------------------------------------------
        plan = self.council.dispatch(drafts, means)
        best = max(means) if means else 0.0
        self._phase(
            squad_index, total,
            f"dispatch: winner={plan.winner_curator} (score {best:.2f}) "
            f"-> {self.scale.agents_per_squad} agents",
        )
        self.board.post(
            squad=squad_index, role=role.name, author="council",
            kind="curator",
            text=f"Curator council dispatched prompt from {plan.winner_curator} "
                 f"(avg {best:.2f}/10).",
            detail=plan.prompt,
        )
        self.log.emit("dispatch", "prompt dispatched", squad=squad_index,
                      curator=plan.winner_curator, score=round(best, 2))

        # -- 4. worker execution (5 concurrent agents) ----------------------
        self._phase(squad_index, total, "workers: 5 agents executing in parallel...")
        agents = [
            Agent(
                agent_id=f"S{squad_index:02d}-agent-{k + 1}",
                squad=squad_index, role=role, llm=self.llm, sandbox=self.sandbox,
                board=self.board, blackboard=self.blackboard,
                metrics=self.metrics, log=self.log, scale=self.scale,
                goal=self.settings.goal, run_dir=self.run_dir,
                board_context_limit=self.settings.board_context_limit,
            )
            for k in range(self.scale.agents_per_squad)
        ]
        works: List[WorkUnit] = []
        with ThreadPoolExecutor(max_workers=self.scale.agents_per_squad) as pool:
            futs = {
                pool.submit(a.work, plan.prompt, k): a
                for k, a in enumerate(agents)
            }
            for fut in as_completed(futs):
                a = futs[fut]
                try:
                    works.append(fut.result())
                except Exception as e:
                    works.append(WorkUnit(
                        agent_id=a.agent_id, squad=squad_index, role=role.name,
                        ok=False, summary=f"agent failed: {e}",
                        detail=str(e), error=str(e),
                    ))
        works.sort(key=lambda w: w.agent_id)

        # -- gather artefacts early (needed by sandbox + verifier) ----------
        artifacts: Dict[str, str] = {}
        for w in works:
            artifacts.update(w.artifacts)

        # persist files once, up front, so verification can run concurrently
        verification_summary = ""
        if artifacts and self.settings.use_sandbox:
            written = self.sandbox.write_files(
                {f"{self.run_dir}/{k}": v for k, v in artifacts.items()}
            )
            self._phase(squad_index, total,
                        f"files: {sum(1 for v in written.values() if v)}/{len(written)} "
                        f"persisted to sandbox")

        # -- 5. review + discussion + verification run CONCURRENTLY ---------
        # These three are mutually independent, so they collapse into a single
        # wave instead of three sequential ones (big wall-clock saving).
        self._phase(squad_index, total,
                    "review + discussion + sandbox verify (in parallel)...")

        # Re-check the ceiling here: curator authoring + voting + 5 workers is
        # already spent, and review/discussion add 13 more calls per squad.
        budget_blown = self._budget_exceeded()
        if budget_blown:
            self._phase(squad_index, total,
                        "budget: skipping review + discussion LLM phases")

        def _do_review():
            if budget_blown:
                self.log.emit("system", "budget: peer review skipped",
                              squad=squad_index)
                return
            peer_review(
                works, self.llm, self.metrics, self.log,
                reviewers=self.settings.peer_reviews_per_work,
            )

        def _do_discuss() -> int:
            if budget_blown:
                self.log.emit("system", "budget: discussion skipped",
                              squad=squad_index)
                return 0
            return discussion_round(
                works, role.name, squad_index, self.llm, self.board,
                self.blackboard, self.log, self.metrics, self.scale,
                self.settings.goal, count=self.settings.discussants,
            )

        def _do_verify():
            if not (artifacts and self.settings.use_sandbox):
                return None
            return verify_squad_files(
                self.sandbox, self.run_dir, artifacts,
                self.board, self.log, squad_index, role.name,
            )

        with ThreadPoolExecutor(max_workers=3) as triad:
            f_review = triad.submit(_do_review)
            f_discuss = triad.submit(_do_discuss)
            f_verify = triad.submit(_do_verify)
            # surface failures instead of hanging on one bad future
            phase_errors: List[str] = []
            for f, label in ((f_review, "review"), (f_discuss, "discussion"),
                             (f_verify, "verify")):
                try:
                    f.result()
                except Exception as e:
                    phase_errors.append(f"{label}: {e}")
                    self.log.emit("error", f"{label} phase failed: {e}",
                                  squad=squad_index)

        v = f_verify.result() if not f_verify.exception() else None
        if v is not None and v.ran:
            verification_summary = (
                f"ran `{v.target}` exit={v.exit_code} passed={v.passed}"
            )
        elif not self.settings.use_sandbox:
            verification_summary = "verification disabled (--no-sandbox)"
        elif not artifacts:
            verification_summary = "no artefacts to verify"
        else:
            verification_summary = "no runnable python artefact"
        self._phase(squad_index, total, f"verify: {verification_summary}")

        # -- 6. summarise + hand off ---------------------------------------
        scores = [w.score for w in works]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # consensus digest for the blackboard
        digest = self._compose_digest(role.name, works, avg_score)

        self.blackboard.add_squad_result(
            squad_index, role.name, digest, role.deliverable, avg_score,
            artifacts=artifacts,
        )

        self.board.post(
            squad=squad_index, role=role.name, author="orchestrator",
            kind="discussion",
            text=f"Squad {squad_index + 1} ({role.name}) completed: "
                 f"avg score {avg_score:.1f}/10, {len(artifacts)} artefact(s). "
                 + (works[0].summary[:120] if works else ""),
        )

        duration = time.time() - t0
        self.metrics.end_squad(squad_index)
        self._phase(squad_index, total,
                    f"done in {duration:.1f}s  avg={avg_score:.2f}  "
                    f"tokens+={self.llm.total_tokens:,}")

        squad_posts = sum(1 for p in self.board.all() if p.squad == squad_index)
        return SquadReport(
            squad=squad_index, role=role.name, duration=duration,
            winning_curator=plan.winner_curator, winner_score=best,
            prompt_scores=means, works=works,
            artifacts=sorted(artifacts.keys()), avg_score=avg_score,
            discussion_posts=squad_posts, ok=not phase_errors,
            verification=verification_summary,
            error="; ".join(phase_errors),
        )

    # ------------------------------------------------------------ digest
    @staticmethod
    def _compose_digest(role_name: str, works: List[WorkUnit], avg: float) -> str:
        """Merge 5 worker summaries into one blackboard entry."""
        if not works:
            return f"{role_name}: no output."
        joined = " ".join(w.summary for w in works)
        files = sorted({f for w in works for f in w.artifacts})
        base = f"{role_name} produced: {joined}"
        if files:
            base += f" Files: {', '.join(files[:8])}"
        return base

    # -------------------------------------------------------- checkpointing
    def _state(self, completed: int) -> Dict:
        """Everything needed to resume a run exactly where it stopped."""
        return {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_squads": completed,
            "total_squads": self.scale.squads,
            "run_dir": self.run_dir,
            "settings": {
                "goal": self.settings.goal,
                "squads": self.scale.squads,
                "agents_per_squad": self.scale.agents_per_squad,
                "curators": self.scale.curators,
                "concurrency": self.scale.concurrency,
                "llm_timeout": self.scale.llm_timeout,
                "live_llm": self.settings.live_llm,
                "use_sandbox": self.settings.use_sandbox,
                "run_prefix": self.settings.run_prefix,
            },
            "board": self.board.to_dict(),
            "blackboard": self.blackboard.to_dict(),
            "metrics": self.metrics.to_dict(),
            "events": self.log.to_dict(),
            "reports": [r.to_dict() for r in self.reports],
        }

    def _checkpoint(self, completed: int) -> None:
        """Persist a resume point locally (always) and in the sandbox (if on).

        A failure here is reported, never swallowed: losing the checkpoint
        silently would make `--resume-from` look like it worked while quietly
        restarting from squad 0.
        """
        try:
            state = self._state(completed)
            payload = json.dumps(state, ensure_ascii=False)
        except Exception as e:
            self._log(f"  ! could not build checkpoint: {type(e).__name__}: {e}")
            return

        # 1) local copy - always available, sandbox or not
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self.checkpoint_path)
        except Exception as e:
            self._log(f"  ! could not write local checkpoint: {e}")

        # 2) mirror into the shared sandbox so the artefact travels with the run
        if self.settings.use_sandbox:
            try:
                self.sandbox.write_file(
                    f"{self.run_dir}/_checkpoint.json", payload
                )
            except Exception as e:
                self._log(f"  ! could not write sandbox checkpoint: {e}")

    # --------------------------------------------------------------- resume
    def restore(self, state: Dict) -> int:
        """Reload a checkpoint (or an exported report) and return the squad to
        continue from, with the blackboard/discussion/metrics intact."""
        if not state:
            return 0
        if "version" in state or "completed_squads" in state:
            return self._restore_checkpoint(state)
        return self._restore_report(state)

    def _restore_checkpoint(self, state: Dict) -> int:
        run_dir = state.get("run_dir")
        if run_dir:
            self.run_dir = run_dir
        board = self.board.load(state.get("board"))
        blackboard = self.blackboard.load(state.get("blackboard"))
        squads = self.metrics.load(state.get("metrics"))
        events = self.log.load(state.get("events"))
        self.reports = [
            SquadReport.from_dict(r) for r in (state.get("reports") or [])
        ]
        completed = int(state.get("completed_squads") or 0)
        self._log(
            f"  restored: {completed} squad(s), {squads} squad metrics, "
            f"{blackboard} blackboard entries, {board} board posts, "
            f"{events} events"
        )
        self.log.emit("system", "resumed from checkpoint",
                      completed=completed, run_dir=self.run_dir)
        return max(0, min(completed, self.scale.squads))

    def _restore_report(self, state: Dict) -> int:
        """Best-effort resume from an exported swarm-report JSON.

        The report carries the board, per-squad rows and the leaderboard but
        not full work-unit detail, so this rebuilds what it can.
        """
        settings = state.get("settings") or {}
        metrics = state.get("metrics") or {}
        if settings.get("run_dir"):
            self.run_dir = settings["run_dir"]

        posts = [
            {"ts": "", "detail": "", "upvotes": 0, **p}
            for p in (state.get("board") or [])
        ]
        self.board.load({"posts": posts,
                         "seq": posts[-1].get("seq", 0) if posts else 0})

        # per-squad rows -> SquadMetrics (started=0 keeps duration intact)
        rows = [
            {
                "squad": int(s.get("squad", 0)),
                "role": s.get("role", ""),
                "started": 0.0,
                "ended": float(s.get("duration", 0.0)),
                "agent_scores": {},
                "curator_votes": [],
                "tokens": 0,
            }
            for s in (state.get("squads") or [])
        ]
        self.metrics.load({
            "t0": time.time(),
            "squads": rows,
            "agent_reputation": {},
            "agent_votes": {},
            "prompt_votes": [],
            "phase_counts": metrics.get("phase_counts") or {},
            "llm": metrics.get("llm") or {},
        })
        for row in (state.get("leaderboard") or []):
            agent = row.get("agent")
            if agent:
                self.metrics.agent_reputation[agent] = float(
                    row.get("reputation", 0.0)
                )
                self.metrics.agent_votes[agent] = int(row.get("reviews", 1) or 1)

        # artefact names survive; content is re-created by later squads
        names = state.get("artifacts") or []
        if names:
            bb = self.blackboard.to_dict()
            for name in names:
                bb["artifacts"].setdefault(name, "")
            self.blackboard.load(bb)

        completed = int(metrics.get("squads_completed") or len(rows))
        self._log(
            f"  restored from report: {completed} squad(s), {len(posts)} board "
            f"posts, {len(state.get('leaderboard') or [])} leaderboard rows"
        )
        self.log.emit("system", "resumed from report", completed=completed)
        return max(0, min(completed, self.scale.squads))

    # ---------------------------------------------------------- reporting
    def final_report(self) -> str:
        # capture the latest LLM counters so the report reflects real spend
        self.metrics.record_llm(self.llm.stats_snapshot())
        lines: List[str] = []
        snap = self.metrics.snapshot()
        llm = snap.get("llm", {})
        lines.append("=" * 78)
        lines.append("AGENT SWARM  |  FINAL REPORT")
        lines.append("=" * 78)
        lines.append(
            f"squads completed : {snap['squads_completed']}/{self.scale.squads}"
        )
        lines.append(
            f"agents           : {self.scale.total_agents} "
            f"({self.scale.squads} x {self.scale.agents_per_squad})"
        )
        lines.append(f"curators         : {self.scale.curators}")
        lines.append(f"elapsed          : {snap['elapsed_sec']}s")
        lines.append(
            f"LLM calls        : {llm.get('calls', 0)}  "
            f"(failures {llm.get('failures', 0)}, retries {llm.get('retries', 0)})"
        )
        lines.append(
            f"tokens           : {llm.get('total_tokens', 0):,} "
            f"(in {llm.get('tokens_in', 0):,} / out {llm.get('tokens_out', 0):,})"
        )
        lines.append(f"avg LLM latency  : {llm.get('avg_latency', 0)}s")
        lines.append(f"discussion posts : {self.board.count()}")
        lines.append(f"artefacts        : {len(self.blackboard.artifact_list())}")

        if llm.get("dead_models"):
            lines.append(f"dead models skip : {', '.join(llm['dead_models'])}")

        lines.append("")
        lines.append("-- SQUADS -----------------------------------------------------------")
        for row in self.metrics.squad_rows():
            vfy = ""
            for r in self.reports:
                if r.squad == row["squad"] and r.verification:
                    vfy = "  " + r.verification[:40]
                    break
            lines.append(
                f"  #{row['squad'] + 1:>2} {row['role']:<20} "
                f"{row['duration']:>6.1f}s  agents={row['agents']}  "
                f"avg={row['avg_score']:.2f}{vfy}"
            )

        lines.append("")
        lines.append("-- LEADERBOARD (peer-voted reputation) -----------------------------")
        for i, row in enumerate(self.metrics.leaderboard(10), 1):
            lines.append(
                f"  {i:>2}. {row['agent']:<16} rep={row['reputation']:<7.2f} "
                f"avg={row['avg']:<5.2f} reviews={row['reviews']}"
            )

        phases = snap.get("phase_counts") or {}
        if phases:
            lines.append("")
            lines.append("-- PHASES (times each stage ran) -----------------------------------")
            for name, count in sorted(phases.items()):
                lines.append(f"  {name:<26} {count}")

        failed = [r for r in self.reports if not r.ok]
        if failed:
            lines.append("")
            lines.append("-- FAILED SQUADS ---------------------------------------------------")
            for r in failed:
                lines.append(f"  #{r.squad + 1:>2} {r.role:<20} {r.error[:120]}")

        lines.append("")
        lines.append("-- ARTEFACTS IN SANDBOX -------------------------------------------")
        arts = self.blackboard.artifact_list()
        if not arts:
            lines.append("  (none)")
        for a in arts[:40]:
            lines.append(f"  {self.run_dir}/{a}")
        if len(arts) > 40:
            lines.append(f"  ... and {len(arts) - 40} more")

        if self.settings.use_sandbox:
            lines.append("")
            lines.append("-- SANDBOX TREE -----------------------------------------------------")
            tree = self.sandbox.list_dir(self.run_dir, depth=4)
            lines.append(tree.strip()[:2000] or "  (empty)")

        lines.append("")
        lines.append(f"checkpoint : {self.checkpoint_path}")
        if self.log.path:
            lines.append(f"event log  : {self.log.path}")
        lines.append("=" * 78)
        return "\n".join(lines)

    # -------------------------------------------------------------- control
    def stop(self) -> None:
        self._stop.set()

    def export_report(self) -> Dict:
        """Machine-readable summary (for JSON export / dashboard)."""
        self.metrics.record_llm(self.llm.stats_snapshot())
        return {
            "settings": {
                "goal": self.settings.goal,
                "squads": self.scale.squads,
                "agents_per_squad": self.scale.agents_per_squad,
                "total_agents": self.scale.total_agents,
                "curators": self.scale.curators,
                "live_llm": self.settings.live_llm,
                "use_sandbox": self.settings.use_sandbox,
                "run_dir": self.run_dir,
            },
            "metrics": self.metrics.snapshot(),
            "squads": self.metrics.squad_rows(),
            "leaderboard": self.metrics.leaderboard(50),
            "artifacts": self.blackboard.artifact_list(),
            "board": [
                {"seq": p.seq, "squad": p.squad, "role": p.role,
                 "author": p.author, "kind": p.kind, "text": p.text}
                for p in self.board.all()
            ],
            "reports": [
                {
                    "squad": r.squad, "role": r.role, "duration": round(r.duration, 2),
                    "winner": r.winning_curator, "winner_score": round(r.winner_score, 3),
                    "prompt_scores": [round(s, 3) for s in r.prompt_scores],
                    "avg_score": round(r.avg_score, 3),
                    "ok": r.ok, "error": r.error,
                    "discussion_posts": r.discussion_posts,
                    "verification": r.verification,
                    "artifacts": r.artifacts,
                    "works": [
                        {"agent": w.agent_id, "score": round(w.score, 2),
                         "summary": w.summary, "files": list(w.artifacts.keys()),
                         "model": w.model, "tokens": w.tokens, "ok": w.ok}
                        for w in r.works
                    ],
                }
                for r in self.reports
            ],
            "checkpoint": self.checkpoint_path,
            "events": self.log.all()[-400:],
        }
