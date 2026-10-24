"""
The discussion round.

After each squad finishes and is peer-reviewed, a subset of agents explicitly
DISCUSS the work: they react to what the squad produced and to the running
project, posting short, public comments to the shared board.

This is the social/coordinating layer of the swarm - every participating AI
can see a less-detailed description of every other AI's work, so the next
squads inherit not just artefacts but the collective reasoning about them.
"""
from __future__ import annotations

from typing import List

from .config import VOTER_MODEL_CHAIN, Scale
from .llm import LLMClient
from .state import Blackboard, DiscussionBoard, EventLog, Metrics
from .agents import WorkUnit, _parse_json_reply


# The angles different discussants take, so the conversation is not uniform.
_ANGLES = [
    "agree with what was done and note the single most valuable part",
    "raise the biggest risk or weakness you see in this squad's output",
    "connect this squad's work to the overall project goal and what must come next",
    "point out any inconsistency with the earlier blackboard entries",
    "suggest one concrete improvement or missing piece",
    "summarise, in one line, what a later squad must not break",
]


def _build_jobs(
    works: List[WorkUnit],
    role_name: str,
    squad: int,
    blackboard_ctx: str,
    goal: str,
    count: int,
) -> List:
    jobs = []
    # Scores may still be unset if peer review is running concurrently, so we
    # only show a score when one is already available.
    digest_lines = []
    for w in works:
        if w.score and w.score > 0:
            digest_lines.append(f"- {w.agent_id}: {w.summary} (score {w.score:.1f})")
        else:
            digest_lines.append(f"- {w.agent_id}: {w.summary}")
    digests = "\n".join(digest_lines)
    for i in range(count):
        angle = _ANGLES[i % len(_ANGLES)]
        who = f"S{squad:02d}-discussant-{i + 1}"
        system = (
            "You are a participating agent in a large collaborative software swarm, taking "
            "part in an open discussion after a work squad finished. Reply with ONLY a JSON "
            'object: {"comment": "<your public comment, 1-2 short sentences>"}'
        )
        user = (
            f"PROJECT GOAL: {goal}\n\n"
            f"THE SQUAD THAT JUST FINISHED: {role_name}\n"
            f"WHAT ITS MEMBERS REPORTED:\n{digests}\n\n"
            f"PROJECT HISTORY (blackboard):\n{blackboard_ctx}\n\n"
            f"Your angle: {angle}.\n"
            "Post your comment for all agents to see. Keep it under 40 words."
        )
        jobs.append((who, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]))
    return jobs


def discussion_round(
    works: List[WorkUnit],
    role_name: str,
    squad: int,
    llm: LLMClient,
    board: DiscussionBoard,
    blackboard: Blackboard,
    log: EventLog,
    metrics: Metrics,
    scale: Scale,
    goal: str,
    count: int = 3,
) -> int:
    """Run an explicit discussion round and broadcast comments.

    Returns the number of comments posted.
    """
    if not works:
        return 0

    jobs = _build_jobs(works, role_name, squad, blackboard.context_block(10), goal, count)
    results = llm.map_chat(
        jobs, models=VOTER_MODEL_CHAIN, max_tokens=300, temperature=0.7
    )

    posted = 0
    for (who, _), res in zip(jobs, results):
        text = ""
        if res.ok and res.text:
            obj = _parse_json_reply(res.text)
            text = str(obj.get("comment", "")).strip() if obj else ""
            if not text:
                text = " ".join(res.text.split())
        if not text:
            continue
        text = " ".join(text.split())[:300]
        board.post(
            squad=squad, role=role_name, author=who, kind="discussion",
            text=text, detail=res.text,
        )
        posted += 1

    metrics.note_phase("discussion")
    log.emit("discussion", f"squad {squad}: {posted} comments posted",
             squad=squad, posted=posted)
    return posted
