"""
Tests for the curator council's voting logic.

Run:  python -m unittest test_vote -v
      SWARM_LIVE=1 python -m unittest test_vote -v   # + real NVIDIA calls

The offline tests always run: they cover the parsing and tallying rules that
decide which prompt a squad gets. The live test (which spends tokens) is
opt-in via SWARM_LIVE=1.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swarm.config import CURATOR_MODEL_CHAIN, Scale  # noqa: E402
from swarm.curators import CuratorCouncil  # noqa: E402
from swarm.llm import LLMClient  # noqa: E402
from swarm.state import DiscussionBoard, EventLog, Metrics  # noqa: E402

LIVE = bool(os.environ.get("SWARM_LIVE"))

_parse_score_map = CuratorCouncil._parse_score_map


def _council(curators=4, live=False, seed=1):
    llm = LLMClient(default_models=CURATOR_MODEL_CHAIN, max_concurrency=8,
                    timeout=90, live=live)
    board = DiscussionBoard()
    log = EventLog()
    metrics = Metrics()
    return CuratorCouncil(llm, board, metrics, log, Scale(curators=curators),
                          seed=seed), llm


def _drafts(council, n):
    return council.author_prompts(
        squad_index=0, role_name="backend-core", role_directive="Implement core logic",
        deliverable="src/core.py",
        blackboard_ctx="- product-architect produced: skeleton", board_ctx="(empty)",
        goal="Build a CLI task manager",
    )[:n]


class ScoreMapParsing(unittest.TestCase):
    """_parse_score_map decides the winner, so it gets the tightest tests."""

    def test_strict_json_shape(self):
        self.assertEqual(
            _parse_score_map('{"scores": {"0": 8, "1": 6.5, "2": 10}}', 3),
            {0: 8.0, 1: 6.5, 2: 10.0},
        )

    def test_out_of_range_scores_dropped(self):
        # candidate 7 does not exist for n=3, and 99 is not a 0-10 score
        self.assertEqual(_parse_score_map('{"7": 4, "99": 5, "1": 6}', 3),
                         {1: 6.0})

    def test_unquoted_keys(self):
        self.assertEqual(_parse_score_map("{0: 9, 2: 3}", 3), {0: 9.0, 2: 3.0})

    def test_garbage_yields_nothing(self):
        self.assertEqual(_parse_score_map("I cannot judge these prompts", 4), {})
        self.assertEqual(_parse_score_map("", 4), {})

    def test_decimal_and_integer_scores_are_both_accepted(self):
        out = _parse_score_map('{"0": 7.75, "1": 8}', 2)
        self.assertEqual(out, {0: 7.75, 1: 8.0})


class OfflineCouncil(unittest.TestCase):
    """Pipeline behaviour that must hold without any network access."""

    def test_every_curator_gets_a_draft(self):
        council, _ = _council(curators=4)
        drafts = _drafts(council, 4)
        self.assertEqual(len(drafts), 4)
        self.assertEqual([d.index for d in drafts], [0, 1, 2, 3])
        self.assertTrue(all(d.prompt.strip() for d in drafts))

    def test_every_curator_scores_every_candidate(self):
        # The brief: all curators score ALL candidates - none may be skipped.
        council, _ = _council(curators=4)
        drafts = _drafts(council, 4)
        council.vote_on_prompts(drafts, "backend-core", 0)
        expected = len(council.curator_ids) * len(drafts)
        self.assertEqual(council.votes_cast, expected)

    def test_means_are_bounded_and_complete(self):
        council, _ = _council(curators=4)
        drafts = _drafts(council, 4)
        means = council.vote_on_prompts(drafts, "backend-core", 0)
        self.assertEqual(len(means), len(drafts))
        for m in means:
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, 10.0)

    def test_dispatch_returns_the_highest_scoring_prompt(self):
        council, _ = _council(curators=3)
        drafts = _drafts(council, 3)
        plan = council.dispatch(drafts, [4.0, 9.0, 6.0])
        self.assertEqual(plan.winner_index, 1)
        self.assertEqual(plan.winner_curator, drafts[1].curator_id)
        self.assertEqual(plan.prompt, drafts[1].prompt)
        self.assertGreater(plan.total_votes, 0)

    def test_empty_dispatch_is_safe(self):
        council, _ = _council(curators=2)
        plan = council.dispatch([], [])
        self.assertEqual(plan.winner_index, -1)
        self.assertEqual(plan.prompt, "")


@unittest.skipUnless(LIVE, "set SWARM_LIVE=1 to spend real NVIDIA tokens")
class LiveCouncil(unittest.TestCase):
    """End-to-end against the real NVIDIA NIM endpoint."""

    def test_live_drafts_and_a_real_vote(self):
        council, llm = _council(curators=4, live=True)
        drafts = _drafts(council, 4)
        self.assertTrue(any(d.ok for d in drafts),
                        "no curator produced a usable prompt")

        means = council.vote_on_prompts(drafts, "backend-core", 0)
        self.assertEqual(len(means), len(drafts))
        self.assertTrue(any(_parse_score_map(str(m), 10) for m in means) or
                        council.votes_cast > 0,
                        "the judge returned no parseable scores")
        snapshot = llm.stats_snapshot()
        self.assertGreater(snapshot["calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
