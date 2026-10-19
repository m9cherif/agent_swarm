import sys, json, re
sys.path.insert(0, r"C:\Users\Labo info\Documents\Projet par défaut")
from swarm.llm import LLMClient
from swarm.config import CURATOR_MODEL_CHAIN
from swarm.curators import CuratorCouncil
from swarm.curators import CuratorCouncil as _CC
_parse_score_map = _CC._parse_score_map
from swarm.state import DiscussionBoard, EventLog, Metrics
from swarm.config import Scale

llm = LLMClient(default_models=CURATOR_MODEL_CHAIN, max_concurrency=8, timeout=90, live=True)
board = DiscussionBoard(); log = EventLog(); metrics = Metrics()
council = CuratorCouncil(llm, board, metrics, log, Scale(curators=4), seed=1)

drafts = council.author_prompts(
    squad_index=0, role_name="backend-core", role_directive="Implement core logic",
    deliverable="src/core.py",
    blackboard_ctx="- product-architect produced: skeleton", board_ctx="(empty)",
    goal="Build a CLI task manager",
)
print("=== DRAFTS (%d) ===" % len(drafts))
for d in drafts:
    print(" [%s] ok=%s len=%d :: %s" % (d.curator_id, d.ok, len(d.prompt), d.prompt[:90].replace("\n"," ")))

# Now show a RAW voting response so we can see what the judge returns
n = len(drafts)
catalogue = "\n\n".join("CANDIDATE %d (by %s):\n%s" % (d.index, d.curator_id, d.prompt) for d in drafts)
system = (
    "You are a prompt-quality judge in a multi-agent swarm. Score each candidate "
    "instruction prompt from 0 to 10 on: specificity, actionability, completeness, "
    "and whether it would make a worker produce the required deliverable. "
    "Reply ONLY as JSON on one line: "
    '{"scores": {"<candidate_number>": <number>, ...}}'
)
user = "ROLE: backend-core\n\nCANDIDATES:\n%s\n\nScore ONLY these candidate numbers: 0, 1, 2, 3\nReturn the JSON object now." % catalogue
res = llm.chat([{"role":"system","content":system},{"role":"user","content":user}],
               models=CURATOR_MODEL_CHAIN, max_tokens=500, temperature=0.2)
print("\n=== RAW VOTE RESPONSE ===")
print("ok=%s model=%s" % (res.ok, res.model))
print(repr(res.text[:600]))
print("\nPARSED:", _parse_score_map(res.text, n))
print("FALLBACK (neutral) would be:", {i: 5.0 for i in range(n)})

# Run the real aggregate vote
means = council.vote_on_prompts(drafts, "backend-core", 0)
print("\n=== AGGREGATE MEANS ===", [round(m,2) for m in means])
print("distinct values?", len(set(round(m,2) for m in means)))
