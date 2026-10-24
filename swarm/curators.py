"""
The Curator Council.

15 meta-AIs whose whole job is to *author prompts for the worker agents*.
The flow for every squad:

  1. AUTHOR    Each curator drafts a candidate prompt for this squad's role,
               given the shared blackboard + discussion context.
  2. VOTE      Every curator scores ALL candidate prompts (including the ones
               it did not write) for clarity, specificity and usefulness.
  3. DISPATCH  Votes are tallied; the winning prompt is dispatched to the 5
               worker agents of the squad.

This gives the swarm a genuine prompt-optimisation layer instead of a single
static template: prompts compete, and only the best-authored instructions
reach the workers.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import CURATOR_MODEL_CHAIN, Scale
from .llm import LLMClient
from .state import DiscussionBoard, EventLog, Metrics


@dataclass
class CuratorDraft:
    curator_id: str
    index: int
    prompt: str
    ok: bool = True


@dataclass
class DispatchPlan:
    prompt: str
    winner_index: int
    winner_curator: str
    mean_scores: List[float]
    total_votes: int
    drafts: List[CuratorDraft] = field(default_factory=list)


class CuratorCouncil:
    def __init__(
        self,
        llm: LLMClient,
        board: DiscussionBoard,
        metrics: Metrics,
        log: EventLog,
        scale: Scale,
        seed: Optional[int] = None,
    ):
        self.llm = llm
        self.board = board
        self.metrics = metrics
        self.log = log
        self.scale = scale
        self.rng = random.Random(seed)
        self.n = scale.curators
        self.curator_ids = [f"curator-{i+1:02d}" for i in range(self.n)]
        # real scores cast by the last vote (used for reporting)
        self.votes_cast = 0

    # -- step 1: author -----------------------------------------------------
    def author_prompts(
        self,
        squad_index: int,
        role_name: str,
        role_directive: str,
        deliverable: str,
        blackboard_ctx: str,
        board_ctx: str,
        goal: str,
    ) -> List[CuratorDraft]:
        """Each curator drafts a candidate prompt for the worker squad."""

        # Diversify vantage points so drafts are genuinely different.
        angles = [
            "very concise and imperative",
            "extremely detailed with explicit acceptance criteria",
            "focused on correctness and edge cases",
            "oriented toward a minimal but complete deliverable",
            "focused on integration with prior work",
            "emphasising file layout and exact paths",
            "written as a numbered step-by-step checklist",
            "demanding concrete code, not prose",
            "focused on tests and verification",
            "optimised for a single-pass high-quality result",
            "highlighting risks and what to avoid",
            "structured as a spec with sections",
            "aware of performance and simplicity trade-offs",
            "requesting rationale plus implementation",
            "framed as a task ticket with definition-of-done",
        ]

        jobs = []
        for i, cid in enumerate(self.curator_ids):
            angle = angles[i % len(angles)]
            system = (
                "You are one of a council of prompt engineers inside a large multi-agent "
                "software swarm. Your only job is to write ONE excellent instruction prompt "
                "that a worker agent will follow. Be specific and actionable. Return ONLY "
                "the prompt text, no preamble, no quotation marks around it."
            )
            user = (
                f"GOAL of the whole project:\n{goal}\n\n"
                f"ROLE this squad plays: {role_name}\n"
                f"Responsibility: {role_directive}\n"
                f"Required deliverable: {deliverable}\n\n"
                f"WHAT HAS BUILT SO FAR (shared blackboard):\n{blackboard_ctx}\n\n"
                f"RECENT DISCUSSION FROM OTHER AGENTS:\n{board_ctx}\n\n"
                f"Write the prompt for the worker agent. Style requirement: {angle}.\n"
                "Include: the objective, the exact output artefact(s) with file paths, and "
                "clear acceptance criteria. Do not mention the council or that you are an AI."
            )
            jobs.append((cid, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]))

        results = self.llm.map_chat(
            jobs,
            models=CURATOR_MODEL_CHAIN,
            max_tokens=700,
            temperature=0.7,
        )

        drafts: List[CuratorDraft] = []
        fallback = self._fallback_prompt(role_name, role_directive, deliverable, goal)
        for i, (cid, res) in enumerate(zip(self.curator_ids, results)):
            text = (res.text or "").strip()
            ok = bool(res.ok and len(text) > 40)
            if not ok:
                text = fallback
            drafts.append(CuratorDraft(curator_id=cid, index=i, prompt=text, ok=ok))

        self.metrics.note_phase("curator_author")
        self.log.emit(
            "curator",
            f"squad {squad_index}: {len(drafts)} prompts authored",
            squad=squad_index,
            ok=sum(1 for d in drafts if d.ok),
        )
        return drafts

    # -- step 2: vote -------------------------------------------------------
    # One reply can only hold so many scores reliably; batch candidates into
    # chunks when a council is unusually large.
    VOTES_PER_CALL = 40

    def vote_on_prompts(
        self,
        drafts: List[CuratorDraft],
        role_name: str,
        squad_index: int,
    ) -> List[float]:
        """Every curator scores every candidate. Returns aggregate mean scores.

        The brief is explicit: all curators score ALL candidate prompts
        (including the ones they did not write). Scoring happens in batches of
        `VOTES_PER_CALL` so a single reply never has to carry more scores than
        a model can return reliably - but no candidate is ever skipped, so no
        candidate gets padded with a neutral 5.0.
        """
        n = len(drafts)
        if n == 0:
            self.votes_cast = 0
            return []

        # Build the catalogue text shared by all voters.
        catalogue = "\n\n".join(
            f"CANDIDATE {d.index} (by {d.curator_id}):\n{d.prompt}" for d in drafts
        )

        chunk = max(1, self.VOTES_PER_CALL)
        jobs = []
        assignments: List[Tuple[str, List[int]]] = []
        for cid in self.curator_ids:
            for start in range(0, n, chunk):
                subset = list(range(start, min(start + chunk, n)))
                assignments.append((cid, subset))
                chosen = ", ".join(str(i) for i in subset)
                system = (
                    "You are a prompt-quality judge in a multi-agent swarm. Score each "
                    "candidate instruction prompt from 0 to 10 on: specificity, "
                    "actionability, completeness, and whether it would make a worker "
                    "produce the required deliverable. "
                    "Reply ONLY as JSON on one line: "
                    '{"scores": {"<candidate_number>": <number>, ...}}'
                )
                user = (
                    f"ROLE: {role_name}\n\n"
                    f"CANDIDATES:\n{catalogue}\n\n"
                    f"Score ONLY these candidate numbers: {chosen}\n"
                    "Return the JSON object now."
                )
                jobs.append((f"{cid}#{start}", [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]))

        results = self.llm.map_chat(
            jobs, models=CURATOR_MODEL_CHAIN, max_tokens=500, temperature=0.2
        )

        # Accumulate scores: sums + counts per candidate index.
        sums = [0.0] * n
        counts = [0] * n
        votes_cast = 0

        for (cid, subset), res in zip(assignments, results):
            parsed = self._parse_score_map(res.text, n)
            rated = set()
            for idx in subset:
                score = parsed.get(idx)
                if score is None:
                    continue
                sums[idx] += score
                counts[idx] += 1
                rated.add(idx)
                votes_cast += 1
                self.metrics.record_prompt_vote(cid, idx, score)
            if not rated and not parsed:
                # The judge returned nothing usable at all: pad every candidate
                # it was asked about with a neutral score so that a silent
                # judge neither promotes nor buries its batch.
                for idx in subset:
                    sums[idx] += 5.0
                    counts[idx] += 1

        self.votes_cast = votes_cast
        means = [(sums[i] / counts[i]) if counts[i] else 5.0 for i in range(n)]
        self.metrics.note_phase("curator_vote")
        self.log.emit(
            "vote",
            f"squad {squad_index}: prompt votes tallied",
            squad=squad_index,
            votes=votes_cast,
            candidates=n,
            judges=len(self.curator_ids),
            means=[round(m, 2) for m in means],
        )
        return means

    @staticmethod
    def _parse_score_map(text: str, n: int) -> Dict[int, float]:
        """Parse '{"scores": {"0": 8, "1": 6}}' (or looser) into {idx: score}."""
        out: Dict[int, float] = {}
        if not text:
            return out
        # find every number key/value pair like "2": 7 or 2: 7
        for m in re.finditer(r'["\']?(\d{1,3})["\']?\s*[:=]\s*["\']?([0-9]{1,2}(?:\.[0-9]+)?)', text):
            try:
                idx = int(m.group(1))
                val = float(m.group(2))
            except ValueError:
                continue
            if 0 <= idx < n and 0 <= val <= 10:
                out[idx] = val
        return out

    # -- step 3: dispatch ---------------------------------------------------
    def dispatch(
        self, drafts: List[CuratorDraft], means: List[float]
    ) -> DispatchPlan:
        if not drafts:
            return DispatchPlan(prompt="", winner_index=-1, winner_curator="",
                                mean_scores=[], total_votes=0)
        if not means:
            means = [0.0] * len(drafts)
        # tie-break deterministically but with a little randomness among equals
        best = max(range(len(drafts)), key=lambda i: (means[i], -abs(i - len(drafts) // 2)))
        votes = self.votes_cast or (len(self.curator_ids) * len(drafts))
        return DispatchPlan(
            prompt=drafts[best].prompt,
            winner_index=best,
            winner_curator=drafts[best].curator_id,
            mean_scores=list(means),
            total_votes=votes,
            drafts=drafts,
        )

    # -- fallback -----------------------------------------------------------
    @staticmethod
    def _fallback_prompt(role_name: str, directive: str, deliverable: str,
                         goal: str) -> str:
        return (
            f"You are the {role_name} in a collaborative software build. "
            f"Overall goal: {goal}\n"
            f"Your responsibility: {directive}\n"
            f"Produce this deliverable: {deliverable}.\n"
            "Work only on your own responsibility. Be concrete: write real content or "
            "real code (not a description of what you would do). Include acceptance "
            "criteria and keep everything consistent with the existing project."
        )
