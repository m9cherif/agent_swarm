"""
Swarm configuration.

All tunables live here. Endpoints, keys, model pools, the role catalog and
the default scale of the swarm. Keys are read from the environment first so
nothing is hard-locked to a single account, but sensible built-in defaults
are provided so the swarm runs out of the box.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Credentials (env override, fallback to the provided keys)
# ---------------------------------------------------------------------------
NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
DAYTONA_API_KEY = os.environ.get(
    "DAYTONA_API_KEY",
    "dtn_ba8b7dbca04c40586743d5c60dd26c52f1262df022fa58601f76031b99ab203d",
)

# ---------------------------------------------------------------------------
# Endpoints (discovered + verified against the live APIs)
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_URL = f"{NVIDIA_BASE_URL}/chat/completions"
NVIDIA_MODELS_URL = f"{NVIDIA_BASE_URL}/models"

DAYTONA_API_BASE = "https://app.daytona.io/api"
DAYTONA_TOOLBOX_BASE = "https://proxy.app.daytona.io/toolbox"

# The sandbox the whole swarm works inside of. Verified running.
DAYTONA_SANDBOX_NAME = "agent_swarm"
DAYTONA_SANDBOX_ID = "dc6e6530-a40f-469f-bb82-191c3e8d42bf"

# Root folder inside the sandbox that every agent shares.
SANDBOX_WORKSPACE = "/root/swarm/workspace"


# ---------------------------------------------------------------------------
# Model pools
# ---------------------------------------------------------------------------
# Three pools, tuned for throughput and cost. 300+ agents each make several
# LLM calls, so we lean on fast, cheap models and keep a couple of stronger
# ones for high-leverage roles (curators, integrator).
#
# VERIFIED against the live API for this key (see `python -m swarm models`
# and the availability sweep). Only models that actually returned content are
# listed:
#   * poolside/laguna-xs-2.1                  - fastest (~0.5s), light reasoning
#   * nvidia/nemotron-3-super-120b-a12b       - strong + fast (~1.2s)
#   * nvidia/nemotron-3.5-lightning-30b-a3b   - fast MoE, but a REASONING model
#     (empty `content` unless max_tokens is generous - the client handles this)
#   * openai/gpt-oss-20b                      - capable, slow (~30s): last resort
#
# NOTE: `z-ai/glm-5.3-flash` and `deepseek-ai/deepseek-v4.1-flash` are
# advertised on this key but NEVER respond (verified: 150s x3 timeouts, and
# no response even with streaming). They are kept in EXPERIMENTAL_MODELS below
# for optional opt-in but are deliberately excluded from the default chains so
# a single hanging endpoint can never stall a 300-agent run.
#
# Every pool entry is a FALLBACK CHAIN: the client tries each model in order
# until one succeeds. This makes the swarm resilient to a single slow/dead
# endpoint without stalling the pipeline.
WORKER_MODEL_CHAIN: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/gpt-oss-20b",
]

CURATOR_MODEL_CHAIN: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/gpt-oss-20b",
]

VOTER_MODEL_CHAIN: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
]

# Strong models reserved for the small number of high-leverage calls
# (the final integrator report, the curator that breaks voting ties).
PREMIUM_MODEL_CHAIN: List[str] = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "poolside/laguna-xs-2.1",
]

# Advertised but currently unresponsive on this key. Opt in only if you are
# prepared for very long per-call latencies.
EXPERIMENTAL_MODELS: List[str] = [
    "z-ai/glm-5.3-flash",
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3",
]


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
@dataclass
class Scale:
    """How big a run is. Tunable from the CLI."""

    # Number of 5-agent squads. Each squad has exactly one role and the
    # squads execute one-after-another, passing their state forward.
    squads: int = 12

    # Agents per squad. Fixed at 5 by the design brief.
    agents_per_squad: int = 5

    # Curator AIs that author + vote on prompts. Fixed at 15 by the brief.
    curators: int = 15

    # Max concurrent LLM calls anywhere in the swarm.
    # Kept moderate: at 36 the live API returned 429 storms which stall every
    # thread on a global cooldown. 16 saturates the per-squad fan-out (the
    # largest burst is 15 curator calls) while staying under the rate limit.
    concurrency: int = 16

    # Per-LLM-call timeout (seconds). Reasoning models under load need room.
    llm_timeout: int = 90

    # Per-agent wall-clock budget inside the sandbox (seconds).
    agent_timeout: int = 300

    @property
    def total_agents(self) -> int:
        return self.squads * self.agents_per_squad


# ---------------------------------------------------------------------------
# Role catalog
# ---------------------------------------------------------------------------
# Each squad draws ONE role from this catalog. The pipeline walks through the
# catalog in order, cycling so that a large run revisits roles at a higher
# level of refinement, always building on the shared blackboard.
@dataclass(frozen=True)
class Role:
    name: str
    directive: str          # what this role is responsible for
    deliverable: str        # the concrete artifact it must produce
    writes_code: bool       # whether it should push files into the sandbox


ROLE_CATALOG: List[Role] = [
    Role(
        "product-architect",
        "Decompose the goal into a coherent system: modules, boundaries, and the contracts between them.",
        "ARCHITECTURE.md describing modules, responsibilities, data flow and interfaces",
        False,
    ),
    Role(
        "api-designer",
        "Define the public interfaces and function signatures other roles will build against.",
        "api/CONTRACTS.md with precise function/class signatures and data shapes",
        False,
    ),
    Role(
        "data-modeler",
        "Design the schemas, records and in-memory data structures the system will use.",
        "models/SCHEMAS.md plus a models.py stub with dataclasses",
        True,
    ),
    Role(
        "backend-core",
        "Implement the core domain logic against the agreed contracts.",
        "src/core.py implementing the central algorithms",
        True,
    ),
    Role(
        "backend-api",
        "Implement the interface layer that exposes core logic to the rest of the system.",
        "src/api.py implementing the public entry points",
        True,
    ),
    Role(
        "db-specialist",
        "Persist state: define storage layout, read/write helpers and any migrations.",
        "src/storage.py with save/load and a storage schema",
        True,
    ),
    Role(
        "frontend-ui",
        "Build the presentation layer that humans will interact with.",
        "src/ui.py (or cli.py) implementing the user-facing surface",
        True,
    ),
    Role(
        "frontend-state",
        "Manage client-side state, validation and the wiring between UI and API.",
        "src/state.py handling validation and state transitions",
        True,
    ),
    Role(
        "test-engineer",
        "Write an automated test suite that exercises the contracts and core logic.",
        "tests/test_suite.py that can run and assert real behaviour",
        True,
    ),
    Role(
        "qa-reviewer",
        "Review everything built so far. Find defects, gaps and inconsistencies.",
        "reviews/QA.md listing concrete, actionable findings ranked by severity",
        False,
    ),
    Role(
        "security-auditor",
        "Threat-model the system and harden it: input validation, injection, secrets, unsafe calls.",
        "security/THREAT_MODEL.md plus concrete fixes applied where safe",
        False,
    ),
    Role(
        "perf-engineer",
        "Profile and optimise the hot paths. Remove waste, add caching where it pays off.",
        "perf/NOTES.md with measured or reasoned optimisations",
        False,
    ),
    Role(
        "docs-writer",
        "Write developer-facing documentation that makes the system usable and maintainable.",
        "docs/README.md with usage, examples and architecture summary",
        False,
    ),
    Role(
        "devops",
        "Provide the operational glue: build, run, lint and CI scripts.",
        "ops/run.sh and ops/README.md to build and run the project",
        True,
    ),
    Role(
        "integrator",
        "Reconcile all prior work into one coherent, runnable whole. Resolve conflicts and fill gaps.",
        "INTEGRATION.md plus a reconciled entry point that ties modules together",
        True,
    ),
]


def role_for_squad(index: int) -> Role:
    """Map a squad index to a role, cycling the catalog."""
    return ROLE_CATALOG[index % len(ROLE_CATALOG)]


# ---------------------------------------------------------------------------
# Behaviour toggles
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    scale: Scale = field(default_factory=Scale)

    # The overall goal the swarm is driving toward.
    goal: str = (
        "Build a small but genuinely working command-line task manager in Python "
        "with a clean architecture: task model, storage, core logic, a CLI, tests "
        "and documentation."
    )

    # Push agent-written code into the sandbox and actually run it.
    use_sandbox: bool = True

    # Run every LLM call for real (False => deterministic offline stubs so the
    # whole pipeline can be exercised without spending tokens).
    live_llm: bool = True

    # Emit a running console transcript.
    verbose: bool = True

    # How many discussion-board digests a single agent is shown as context.
    board_context_limit: int = 40

    # Peer-vote: how many squadmates review each work unit.
    peer_reviews_per_work: int = 2

    # Discussion round: how many agents comment publicly after each squad.
    discussants: int = 3

    # Hard token ceiling for the whole run; the swarm stops cleanly when hit.
    # None/0 disables the budget guard.
    max_total_tokens: int = 0

    # Artifact folder under the sandbox workspace.
    run_prefix: str = "run"


SETTINGS = Settings()
