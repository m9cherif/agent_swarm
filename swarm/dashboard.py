"""
HTML dashboard: renders a swarm run as a single self-contained page.

Takes the dict from `Orchestrator.export_report()` and writes an offline HTML
file (no external JS/CSS dependencies) showing squad progression, the curator
voting outcome, the discussion board feed, the peer-voted leaderboard and the
artefacts that landed in the sandbox.
"""
from __future__ import annotations

import html
import json
from typing import Dict, List


def _esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _score_class(v: float) -> str:
    if v >= 8:
        return "good"
    if v >= 6:
        return "mid"
    return "low"


def render(report: Dict) -> str:
    s = report.get("settings", {})
    m = report.get("metrics", {})
    llm = m.get("llm", {})
    squads: List[Dict] = report.get("squads", [])
    board: List[Dict] = report.get("board", [])
    leaders: List[Dict] = report.get("leaderboard", [])
    artifacts: List[str] = report.get("artifacts", [])
    reports: List[Dict] = report.get("reports", [])

    max_dur = max([row.get("duration", 0) for row in squads] or [1]) or 1
    max_rep = max([r.get("reputation", 0) for r in leaders] or [1]) or 1

    # --- squad timeline bars ---
    squad_bars = []
    for row in squads:
        d = row.get("duration", 0)
        pct = max(3, int(100 * d / max_dur))
        vfy = ""
        for r in reports:
            if r.get("squad") == row.get("squad"):
                v = r.get("verification", "")
                if v:
                    vfy = f'<div class="vfy">{_esc(v)}</div>'
                break
        avg = row.get("avg_score", 0)
        squad_bars.append(f"""
        <div class="row">
          <div class="squad-label">#{row.get('squad',0)+1} {_esc(row.get('role',''))}</div>
          <div class="bar-track">
            <div class="bar {_score_class(avg)}" style="width:{pct}%"></div>
          </div>
          <div class="squad-meta">{d:.1f}s · avg {avg:.2f}</div>
        </div>{vfy}""")

    # --- leaderboard ---
    leader_rows = []
    for i, r in enumerate(leaders[:15], 1):
        rep = r.get("reputation", 0)
        pct = max(4, int(100 * rep / max_rep))
        leader_rows.append(f"""
        <div class="row">
          <div class="squad-label">{i}. {_esc(r.get('agent',''))}</div>
          <div class="bar-track"><div class="bar rep" style="width:{pct}%"></div></div>
          <div class="squad-meta">rep {rep:.1f} · avg {r.get('avg',0):.1f}</div>
        </div>""")

    # --- discussion feed ---
    feed = []
    kind_emoji = {
        "digest": "🤖", "discussion": "💬", "curator": "🗳️", "vote": "🗳️",
        "verification": "✅", "system": "🏁", "work": "⚙️",
    }
    for p in board[-60:]:
        emoji = kind_emoji.get(p.get("kind", ""), "•")
        feed.append(
            f'<div class="post"><span class="k">{emoji}</span> '
            f'<span class="tag">S{int(p.get("squad",-1))+1 if p.get("squad",-1)>=0 else "-"}</span> '
            f'<b>{_esc(p.get("author",""))}</b> '
            f'<i class="role">{_esc(p.get("role",""))}</i>: {_esc(p.get("text",""))}</div>'
        )

    # --- artifacts ---
    art = "".join(f"<li><code>{_esc(a)}</code></li>" for a in artifacts) or "<li>none</li>"

    total = s.get("total_agents", 0)
    curators = s.get("curators", 0)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Swarm Dashboard</title>
<style>
 :root {{ --bg:#0b0e14; --panel:#131824; --line:#1f2634; --fg:#e6ebf5;
   --mut:#8a94a8; --good:#3ddc97; --mid:#f5c451; --low:#ff6b6b; --acc:#5aa9ff; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--fg);
   font:14px/1.5 ui-monospace,'Cascadia Code',Menlo,Consolas,monospace; }}
 header {{ padding:22px 26px; border-bottom:1px solid var(--line);
   background:linear-gradient(180deg,#121826,#0b0e14); }}
 h1 {{ margin:0 0 6px; font-size:22px; letter-spacing:.5px; }}
 .sub {{ color:var(--mut); font-size:13px; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; padding:18px 26px; }}
 .card {{ background:var(--panel); border:1px solid var(--line);
   border-radius:10px; padding:12px 16px; min-width:130px; }}
 .card .v {{ font-size:24px; font-weight:700; color:var(--acc); }}
 .card .l {{ font-size:11px; color:var(--mut); text-transform:uppercase;
   letter-spacing:.8px; }}
 section {{ padding:10px 26px 24px; }}
 h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:1.4px;
   color:var(--mut); border-bottom:1px solid var(--line); padding-bottom:8px;
   margin:22px 0 14px; }}
 .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:26px; }}
 @media(max-width:900px){{ .grid{{grid-template-columns:1fr}} }}
 .row {{ display:grid; grid-template-columns:190px 1fr 130px; align-items:center;
   gap:10px; padding:5px 0; }}
 .squad-label {{ font-size:12px; white-space:nowrap; overflow:hidden;
   text-overflow:ellipsis; }}
 .bar-track {{ background:#0e1220; border:1px solid var(--line); border-radius:6px;
   height:16px; overflow:hidden; }}
 .bar {{ height:100%; background:var(--acc); }}
 .bar.good {{ background:var(--good); }} .bar.mid {{ background:var(--mid); }}
 .bar.low {{ background:var(--low); }} .bar.rep {{ background:#a970ff; }}
 .squad-meta {{ font-size:11px; color:var(--mut); text-align:right; }}
 .vfy {{ font-size:11px; color:var(--good); padding-left:200px; }}
 #feed {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
   padding:12px; max-height:440px; overflow:auto; }}
 .post {{ padding:5px 4px; border-bottom:1px dashed #1b2130; font-size:12.5px; }}
 .post .k {{ opacity:.8; }} .tag {{ color:var(--acc); font-weight:700;
   font-size:11px; }}
 .role {{ color:var(--mut); font-size:11px; }}
 ul {{ margin:0; padding-left:18px; }} li {{ font-size:12.5px; padding:2px 0; }}
 code {{ color:var(--good); }}
 .pill {{ display:inline-block; background:#182033; border:1px solid var(--line);
   border-radius:20px; padding:3px 10px; font-size:11px; color:var(--mut);
   margin:4px 6px 0 0; }}
</style></head>
<body>
<header>
  <h1>🧬 AGENT SWARM DASHBOARD</h1>
  <div class="sub">{_esc(s.get('goal',''))[:160]}</div>
  <div class="sub" style="margin-top:6px">
    run dir <code>{_esc(s.get('run_dir',''))}</code>
  </div>
</header>

<div class="cards">
  <div class="card"><div class="v">{total}</div><div class="l">agents</div></div>
  <div class="card"><div class="v">{s.get('squads',0)}</div><div class="l">squads (seq)</div></div>
  <div class="card"><div class="v">{curators}</div><div class="l">curators</div></div>
  <div class="card"><div class="v">{m.get('squads_completed',0)}</div><div class="l">completed</div></div>
  <div class="card"><div class="v">{llm.get('calls',0)}</div><div class="l">llm calls</div></div>
  <div class="card"><div class="v">{llm.get('total_tokens',0):,}</div><div class="l">tokens</div></div>
  <div class="card"><div class="v">{m.get('elapsed_sec',0)}s</div><div class="l">elapsed</div></div>
  <div class="card"><div class="v">{len(board)}</div><div class="l">board posts</div></div>
</div>

<section>
  <div class="grid">
    <div>
      <h2>Squad timeline (sequential roles)</h2>
      {''.join(squad_bars) or '<div class="sub">no squads</div>'}
    </div>
    <div>
      <h2>Peer-voted leaderboard</h2>
      {''.join(leader_rows) or '<div class="sub">no votes yet</div>'}
      <h2>Sandbox artefacts</h2>
      <ul>{art}</ul>
      <h2>Model usage</h2>
      <div>{''.join(f'<span class="pill">{_esc(k)}: {v}</span>' for k,v in (llm.get('model_calls',{}) or {}).items())}</div>
    </div>
  </div>
</section>

<section>
  <h2>Discussion board (visible to every agent)</h2>
  <div id="feed">{''.join(feed) or '<div class="sub">empty</div>'}</div>
</section>
</body></html>"""


def write(report: Dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report))
    return path
