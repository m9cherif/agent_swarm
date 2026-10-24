# 🧬 Agent Swarm

A large multi-agent software swarm that builds a project through **sequential
5-agent role squads**, orchestrated by a **15-AI curator council** that authors
and votes on prompts, coordinated through a **shared discussion board**, and
executing for real inside one **Daytona sandbox**.

Pure Python standard library — no third-party dependencies to install.

---

## Architecture at a glance

```
                    ┌─────────────────────────────────────────────┐
                    │   CURATOR COUNCIL  (15 meta-AIs)            │
                    │  1. AUTHOR  each drafts a prompt            │
                    │  2. VOTE    all score all candidates        │
                    │  3. DISPATCH winning prompt → the squad     │
                    └──────────────────┬──────────────────────────┘
                                       │
   ┌───────────────────────────────────▼────────────────────────────────┐
   │  SQUAD (5 worker agents, ONE shared role, run in parallel)         │
   │   • read the shared blackboard + discussion board                  │
   │   • produce real files / real code                                 │
   │   • write them into the Daytona sandbox                            │
   │   • post a SHORT digest visible to every other agent               │
   └───────────────────────────────────┬────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌──────────────┐             ┌────────────────┐             ┌──────────────────┐
 │  PEER REVIEW │             │ DISCUSSION     │             │ SANDBOX VERIFY   │
 │ agents score │             │ round: agents  │             │ code is actually │
 │ each other   │             │ comment public │             │ RUN, pass/fail   │
 │ (0-10)       │             │ for all to see │             │ posted to board  │
 └──────────────┘             └────────────────┘             └──────────────────┘
        │
        ▼  results pushed to the BLACKBOARD ──► the NEXT squad reads it as input
```

**Squads run one after another.** Each squad takes exactly one role from a
15-role catalogue (architect → API → data → backend → tests → security → … ),
consumes the accumulated blackboard, and hands its results forward. Larger
runs cycle the catalogue so later passes refine earlier work.

---

## What each part does

| Feature | Where | What it does |
|---|---|---|
| **Sequential role squads** | `orchestrator.py` | 5 agents per squad, one role per squad, squads run in order and pass state |
| **Curator council (15 AIs)** | `curators.py` | Author prompts → vote on all candidates → dispatch the winner |
| **Shared discussion board** | `state.py` | Condensed digests every agent can see (`board.context_block()`) |
| **Explicit discussion rounds** | `discussion.py` | Agents publicly comment after each squad |
| **Peer voting + leaderboard** | `agents.py`, `state.py` | Agents score each other 0–10; reputation leaderboard |
| **Blackboard handoff** | `state.py` | Accumulated project state fed to the next squad |
| **Real sandbox execution** | `sandbox.py` | Files + shell inside Daytona `agent_swarm` as root |
| **Code verification** | `verify.py` | Runs what agents wrote (tests via pytest/unittest); broadcasts real pass/fail |
| **Resilient LLM client** | `llm.py` | Model fallback chains, 429 cooldown, dead-model skip, reasoning-aware parsing |
| **Checkpoint + resume** | `orchestrator.py`, `cli.py` | `out/_checkpoint.json` saves board + blackboard + metrics + reports; `--resume-from` continues with all of it restored |
| **Token budget guard** | `orchestrator.py` | `--budget N` stops the run cleanly |
| **Offline mode** | `llm.py` | Deterministic stubs — full pipeline without spending tokens |
| **Reports** | `cli.py`, `dashboard.py` | Text + JSON + self-contained **HTML dashboard** |

---

## Setup

Python 3.8+ (standard library only).

```bash
# verify both APIs are reachable
python -m swarm probe

# list the NVIDIA models available to your key
python -m swarm models
```

Credentials default to the provided NVIDIA + Daytona keys and can be overridden:

```bash
set NVIDIA_API_KEY=nvapi-...
set DAYTONA_API_KEY=dtn_...
```

Everything runs in the Daytona sandbox `agent_swarm`
(id `dc6e6530-a40f-469f-bb82-191c3e8d42bf`), workspace
`/root/swarm/workspace/<run-prefix>-<timestamp>/`.

---

## Usage

```bash
# dry pipeline test — no tokens, no sandbox writes
python -m swarm run --squads 3 --offline --no-sandbox

# small live run
python -m swarm run --squads 4 --curators 8

# THE FULL SWARM: 300 agents = 60 squads x 5 agents, sequential
python -m swarm run --squads 60 --curators 15

# with a token ceiling
python -m swarm run --squads 60 --budget 3000000

# continue an interrupted run (writes out/_checkpoint.json after every squad)
python -m swarm run --squads 60 --resume-from out/_checkpoint.json
#   an exported report works too: --resume-from out/swarm-report-<stamp>.json
```

Everything a run produces lands in `out/` (git-ignored):

* `swarm-report-*.txt` — human-readable final report
* `swarm-report-*.json` — full machine-readable export
* `swarm-dashboard-*.html` — open in a browser for the visual dashboard
* `_checkpoint.json` — resume point: board, blackboard, metrics and squad reports
* `events-*.jsonl` — append-only audit trail of every dispatch/vote/review/verify event

---

## Tests

```bash
# offline tests (no network, no tokens)
python -m unittest discover -s . -p "test_*.py" -v

# also run the live NVIDIA / Daytona tests
SWARM_LIVE=1 python -m unittest discover -s . -p "test_*.py" -v
```

`test_vote.py` covers curator scoring and dispatch; `test_verify.py` covers the
verification decision tree — including the case where a test file only *defines*
its cases and must therefore **not** be reported as verified.

---

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--squads` | 12 | Number of sequential 5-agent squads (×5 = total agents) |
| `--agents-per-squad` | 5 | Agents per squad (design constant) |
| `--curators` | 15 | Curator AIs that author + vote on prompts |
| `--concurrency` | 16 | Max simultaneous LLM calls |
| `--llm-timeout` | 90 | Per-LLM-call timeout in seconds |
| `--reviews` | 2 | Peer reviews per work unit |
| `--discussants` | 3 | Agents commenting in each discussion round |
| `--board-context` | 40 | Discussion posts each agent sees |
| `--budget` | 0 | Stop after ~N total tokens (0 = unlimited) |
| `--resume-from` | — | Continue from `out/_checkpoint.json` or a report JSON |
| `--run-prefix` | `run` | Sandbox workspace folder prefix |
| `--out` | `out/` | Where reports, dashboards and the checkpoint are written |
| `--quiet` | off | Suppress the live transcript |
| `--offline` | off | Deterministic stubs instead of live LLM |
| `--no-sandbox` | off | Skip all Daytona interaction |
| `--goal` | (built-in) | The project the swarm builds |

Numeric defaults come from `swarm/config.py` (`Scale` / `Settings`), so
`--help`, the config and this table cannot drift apart.

---

## Model notes (verified against the live API)

The NVIDIA key exposes 81 model IDs, but only a handful actually respond:

| Model | Behaviour |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | **Primary** — fast (~7s), clean JSON output |
| `poolside/laguna-xs-2.1` | Very fast (~0.5s) but intermittently 503 |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | Works, but reasoning-heavy (~35s) |
| `openai/gpt-oss-20b` | Capable but slow (~30s) — last resort |
| `z-ai/glm-5.3*`, `deepseek-v4.1-flash` | **Advertised but never respond** (timed out at 150s ×3, also with streaming). Kept in `EXPERIMENTAL_MODELS` only. |

All configured pools are **fallback chains**: the client skips dead (404)
models permanently, drops a 5xx model for the rest of that call, and applies a
global cooldown on 429s.

Because most NIM endpoints are **reasoning models**, the visible `content` can
be empty while text lands in `reasoning_content`. `llm.py` extracts from every
field and budgets `max_tokens` accordingly. It also sends
`reasoning_effort: "none"` so reasoning models answer immediately (0 reasoning
tokens); any model that rejects the parameter is remembered and simply never
asked again — the call is retried without it instead of failing.

---

## Layout

```
swarm/
  __init__.py
  __main__.py        # python -m swarm
  cli.py             # run / probe / models commands
  config.py          # keys, endpoints, model pools, roles, scale, settings
  llm.py             # resilient NVIDIA NIM client
  sandbox.py         # Daytona sandbox client (exec + files)
  state.py           # blackboard, discussion board, event log, metrics
  curators.py        # 15-AI council: author, vote, dispatch
  agents.py          # worker agents + peer review
  discussion.py      # public discussion rounds
  verify.py          # run agent code in the sandbox, broadcast results
  orchestrator.py    # the pipeline
  dashboard.py       # self-contained HTML dashboard
```

---

## Original brief (as filed)

> **the prompt to opencode:**
>
> use this nvidia api key: `nvapi-[REDACTED]` to make a full agent swarm that there is a lot of
> agents work (300+) : every 5 has a single role they do it, one after one, then pass to the
> next 5 agents role... and there is 15 other AIs that create prompts to every ai and when
> they complete the prompts creating they vote for them and send the best prompts. and all the
> participated ais discuss and less detailled description of his work all the ais can see it.
> and all of them work on the sandbox of that daytona api key: `dtn_[REDACTED]`
> (the sandbox is agent_swarm) and they have full access to it. and add MANY other features,
> go fast and hard.

How the implementation maps to it:

| Requirement | Implementation |
|---|---|
| 300+ agents, 5 per role, sequential handoff | `--squads 60` × 5 workers; `ROLE_CATALOG` in `config.py`; squads run one after another via `orchestrator.py` |
| 15 AIs author prompts, vote, send the best | `curators.py` — author → vote → dispatch for every squad |
| All agents discuss; condensed work visible to all | `state.py` DiscussionBoard + `discussion.py` rounds (every agent sees short digests) |
| Work in the Daytona `agent_swarm` sandbox, full access | `sandbox.py` — root shell + file writes in sandbox `dc6e6530-a40f-469f-bb82-191c3e8d42bf` |
| MANY other features | peer-review leaderboard, blackboard handoff, sandbox code verification, checkpoint/resume, token budget, HTML dashboard, offline mode, model fallback chains |
