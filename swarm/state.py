"""
Shared swarm state: the blackboard, the discussion board, the event log and
the metrics ledger.

* Blackboard   - the accumulated, authoritative result of all completed
                 squads. Every later squad reads it as its input context.
* DiscussionBoard - a public feed where every agent posts a SHORT digest of
                 its work that ALL other agents can see (the "less detailed
                 description of his work all the AIs can see").
* EventLog     - append-only JSONL audit trail of everything that happened.
* Metrics      - token spend, calls, timings, per-squad and per-agent scores.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Discussion board
# ---------------------------------------------------------------------------
@dataclass
class BoardPost:
    seq: int
    ts: str
    squad: int
    role: str
    author: str          # e.g. "S03-agent-2"
    kind: str            # 'digest' | 'discussion' | 'curator' | 'vote' | 'system'
    text: str            # short, broadcast-safe summary
    detail: str = ""     # full detail, kept out of others' context by default
    upvotes: int = 0

    def brief(self) -> str:
        """The compact form other agents see."""
        return f"[{self.role} | {self.author}] {self.text}"


class DiscussionBoard:
    """Append-only public feed visible to every agent.

    `recent()` returns the condensed posts (brief form) that get injected
    into another agent's context, so the whole swarm stays coordinated
    without drowning any single agent in full transcripts.
    """

    def __init__(self, maxlen: int = 4000):
        self._lock = threading.Lock()
        self._posts: List[BoardPost] = []
        self._seq = 0
        self._maxlen = maxlen

    def post(self, squad: int, role: str, author: str, text: str,
             kind: str = "digest", detail: str = "") -> BoardPost:
        with self._lock:
            self._seq += 1
            p = BoardPost(
                seq=self._seq, ts=_now(), squad=squad, role=role, author=author,
                kind=kind, text=" ".join(str(text).split())[:600],
                detail=detail[:8000],
            )
            self._posts.append(p)
            if len(self._posts) > self._maxlen:
                self._posts = self._posts[-self._maxlen:]
            return p

    def recent(self, limit: int = 40, kinds: Optional[set] = None) -> List[BoardPost]:
        with self._lock:
            items = list(self._posts)
        if kinds:
            items = [p for p in items if p.kind in kinds]
        return items[-limit:]

    def context_block(self, limit: int = 40) -> str:
        """Human-readable block of the latest posts for prompt injection."""
        posts = self.recent(limit=limit)
        if not posts:
            return "(discussion board is empty so far)"
        lines = [f"#{p.seq} {p.brief()}" for p in posts]
        return "\n".join(lines)

    def count(self) -> int:
        with self._lock:
            return len(self._posts)

    def all(self) -> List[BoardPost]:
        with self._lock:
            return list(self._posts)

    # -- persistence (checkpoint / resume) --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "posts": [asdict(p) for p in self._posts],
            }

    def load(self, data: Optional[Dict[str, Any]]) -> int:
        """Restore posts from a checkpoint. Returns how many were loaded."""
        if not data:
            return 0
        posts = [BoardPost(**p) for p in (data.get("posts") or [])]
        seq = data.get("seq")
        with self._lock:
            self._posts = posts
            self._seq = int(seq) if seq else (posts[-1].seq if posts else 0)
        return len(posts)


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------
class Blackboard:
    """Accumulated knowledge handed from squad to squad."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._artifacts: Dict[str, str] = {}  # relpath -> full content

    def add_squad_result(self, squad_index: int, role: str, summary: str,
                         deliverable: str, score: float,
                         artifacts: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._entries.append({
                "squad": squad_index,
                "role": role,
                "summary": " ".join(summary.split())[:1500],
                "deliverable": deliverable,
                "score": round(score, 2),
                "ts": _now(),
            })
            for k, v in (artifacts or {}).items():
                self._artifacts[k] = v

    def history(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries[-limit:])

    def artifacts(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._artifacts)

    def context_block(self, limit: int = 25) -> str:
        """Condensed blackboard to feed the next squad."""
        entries = self.history(limit)
        if not entries:
            return "(nothing has been built yet - you are the first squad)"
        lines = []
        for e in entries:
            lines.append(
                f"- [{e['role']} | squad {e['squad']} | score {e['score']}] "
                f"{e['summary']}"
            )
        return "\n".join(lines)

    def artifact_list(self) -> List[str]:
        with self._lock:
            return sorted(self._artifacts.keys())

    def get_artifact(self, relpath: str) -> Optional[str]:
        with self._lock:
            return self._artifacts.get(relpath)

    def total(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- persistence (checkpoint / resume) --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": list(self._entries),
                "artifacts": dict(self._artifacts),
            }

    def load(self, data: Optional[Dict[str, Any]]) -> int:
        if not data:
            return 0
        with self._lock:
            self._entries = list(data.get("entries") or [])
            self._artifacts = dict(data.get("artifacts") or {})
        return len(self._entries)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------
class EventLog:
    """Append-only JSONL log (also kept in memory for inspection)."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def emit(self, kind: str, msg: str, **fields: Any) -> Dict[str, Any]:
        ev = {"ts": _now(), "kind": kind, "msg": msg, **fields}
        with self._lock:
            self._events.append(ev)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                except Exception:
                    pass
        return ev

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    # -- persistence (checkpoint / resume) --------------------------------
    def to_dict(self, limit: int = 5000) -> Dict[str, Any]:
        """Serialise recent events. `limit` keeps a long run's checkpoint sane."""
        with self._lock:
            events = list(self._events[-limit:])
        return {"path": self.path or "", "events": events, "total": self.count()}

    def load(self, data: Optional[Dict[str, Any]]) -> int:
        """Restore events into memory only - the JSONL file already has them."""
        if not data:
            return 0
        with self._lock:
            self._events = list(data.get("events") or [])
        return len(self._events)


# ---------------------------------------------------------------------------
# Metrics / leaderboard
# ---------------------------------------------------------------------------
@dataclass
class SquadMetrics:
    squad: int
    role: str
    started: float = 0.0
    ended: float = 0.0
    agent_scores: Dict[str, float] = field(default_factory=dict)
    curator_votes: List[float] = field(default_factory=list)
    tokens: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.ended - self.started)


class Metrics:
    """Central ledger: spend, timings, quality scores, reputation."""

    def __init__(self):
        self._lock = threading.Lock()
        self.t0 = time.time()
        self.squads: Dict[int, SquadMetrics] = {}
        self.agent_reputation: Dict[str, float] = defaultdict(float)
        self.agent_votes: Dict[str, int] = defaultdict(int)
        self.prompt_votes: List[Dict[str, Any]] = []
        self.phase_counts: Dict[str, int] = defaultdict(int)
        self.llm = {}  # filled from LLMClient snapshot

    def start_squad(self, squad: int, role: str) -> None:
        with self._lock:
            self.squads[squad] = SquadMetrics(squad=squad, role=role, started=time.time())

    def end_squad(self, squad: int) -> None:
        with self._lock:
            if squad in self.squads:
                self.squads[squad].ended = time.time()

    def record_agent_score(self, squad: int, agent_id: str, score: float) -> None:
        with self._lock:
            if squad in self.squads:
                self.squads[squad].agent_scores[agent_id] = score
            self.agent_reputation[agent_id] += score
            self.agent_votes[agent_id] += 1

    def record_prompt_vote(self, curator: str, candidate_idx: int, score: float) -> None:
        with self._lock:
            self.prompt_votes.append({
                "curator": curator, "candidate": candidate_idx, "score": score
            })

    def note_phase(self, phase: str) -> None:
        with self._lock:
            self.phase_counts[phase] += 1

    def record_llm(self, snapshot: Dict) -> None:
        with self._lock:
            self.llm = snapshot

    # -- reporting ----------------------------------------------------------
    def leaderboard(self, top: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for agent, rep in self.agent_reputation.items():
                votes = self.agent_votes.get(agent, 1) or 1
                rows.append({
                    "agent": agent,
                    "reputation": round(rep, 2),
                    "avg": round(rep / votes, 2),
                    "reviews": votes,
                })
            rows.sort(key=lambda r: (-r["reputation"], -r["avg"]))
            return rows[:top]

    def squad_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for s in sorted(self.squads.values(), key=lambda x: x.squad):
                avg = (
                    sum(s.agent_scores.values()) / len(s.agent_scores)
                    if s.agent_scores else 0.0
                )
                rows.append({
                    "squad": s.squad,
                    "role": s.role,
                    "duration": round(s.duration, 1),
                    "agents": len(s.agent_scores),
                    "avg_score": round(avg, 2),
                })
            return rows

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total_agents_scored = len(self.agent_reputation)
            avg_rep = (
                sum(self.agent_reputation.values()) / total_agents_scored
                if total_agents_scored else 0.0
            )
            return {
                "elapsed_sec": round(time.time() - self.t0, 1),
                "squads_completed": sum(
                    1 for s in self.squads.values() if s.ended > 0
                ),
                "agents_scored": total_agents_scored,
                "avg_reputation": round(avg_rep, 2),
                "phase_counts": dict(self.phase_counts),
                "prompt_votes": len(self.prompt_votes),
                "llm": dict(self.llm),
            }

    # -- persistence (checkpoint / resume) --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "t0": self.t0,
                "squads": [asdict(s) for s in
                           sorted(self.squads.values(), key=lambda x: x.squad)],
                "agent_reputation": dict(self.agent_reputation),
                "agent_votes": dict(self.agent_votes),
                "prompt_votes": list(self.prompt_votes),
                "phase_counts": dict(self.phase_counts),
                "llm": dict(self.llm),
            }

    def load(self, data: Optional[Dict[str, Any]]) -> int:
        if not data:
            return 0
        with self._lock:
            self.t0 = float(data.get("t0") or time.time())
            self.squads = {
                s["squad"]: SquadMetrics(**s) for s in (data.get("squads") or [])
            }
            self.agent_reputation = defaultdict(
                float, data.get("agent_reputation") or {}
            )
            self.agent_votes = defaultdict(int, data.get("agent_votes") or {})
            self.prompt_votes = list(data.get("prompt_votes") or [])
            self.phase_counts = defaultdict(int, data.get("phase_counts") or {})
            self.llm = dict(data.get("llm") or {})
        return len(self.squads)
