"""Builds a self-contained HTML robustness report from a run directory."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from .analyst import sensitivities
from .schemas import Axis, Scenario, StressPlan

E = html.escape

CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--warn-bg:#fdf3dc;--warn-ink:#6b4a00;
--seq0:#f0efec;--seq1:#cde2fb;--seq2:#9ec5f4;--seq3:#6da7ec;--seq4:#3987e5;--seq5:#256abf;--seq6:#184f95;--seq7:#0d366b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface:#1a1a19;
--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--warn-bg:#3a2f12;--warn-ink:#f5d78a;--seq0:#2a2a28}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;
--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--warn-bg:#3a2f12;--warn-ink:#f5d78a;--seq0:#2a2a28}
*{box-sizing:border-box}body{margin:0;background:var(--page)}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);min-height:100vh;
padding:32px 16px 64px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;font-weight:600;margin:0 0 4px}h2{font-size:18px;font-weight:600;margin:40px 0 4px}
.sub{color:var(--ink2);margin:0 0 16px;line-height:1.5}.cap{color:var(--muted);font-size:13px;margin:6px 0 0}
.banner{background:var(--warn-bg);color:var(--warn-ink);border-radius:10px;padding:10px 14px;font-size:14px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr));gap:12px;margin-top:20px}
.tile,.card{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:16px}
.tile .k{color:var(--ink2);font-size:13px}.tile .v{font-size:32px;font-weight:600;margin:6px 0 2px}
.tile .d{color:var(--muted);font-size:13px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));gap:12px;margin-top:12px}
.card h3{font-size:15px;font-weight:600;margin:0 0 2px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:8px 0 4px}
.legend i{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px;border-radius:1px}
svg text{font-family:inherit}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.heat td.cell{text-align:center;font-variant-numeric:tabular-nums;border:2px solid var(--surface);border-radius:4px;min-width:64px}
.clusters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:12px;margin-top:12px}
.thumbs{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.thumbs img{width:88px;border-radius:6px;box-shadow:0 0 0 1px var(--ring)}
.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;background:var(--seq0);color:var(--ink2);margin:2px 4px 2px 0}
.round{border-left:2px solid var(--grid);padding:2px 0 2px 14px;margin:14px 0}.round b{font-weight:600}
details{margin-top:10px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 4px 14px rgba(0,0,0,.12);
border-radius:8px;padding:8px 10px;font-size:13px;display:none;z-index:10;min-width:150px}
#tip .row{display:flex;align-items:center;gap:8px;margin:2px 0}#tip .row i{width:12px;height:2px;border-radius:1px}
#tip .row b{font-weight:600}#tip .t{color:var(--ink2);font-size:12px;margin-bottom:4px}
a{color:var(--s1);word-break:break-all}.scroll{overflow-x:auto}
"""

JS = """
const tip=document.getElementById('tip');
function showTip(e,title,rows){tip.replaceChildren();const t=document.createElement('div');t.className='t';t.textContent=title;tip.appendChild(t);
for(const r of rows){const d=document.createElement('div');d.className='row';if(r.color){const i=document.createElement('i');i.style.background=r.color;d.appendChild(i);}
const b=document.createElement('b');b.textContent=r.value;d.appendChild(b);const s=document.createElement('span');s.textContent=r.label;d.appendChild(s);tip.appendChild(d);}
tip.style.display='block';const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}
function hideTip(){tip.style.display='none';}
document.querySelectorAll('[data-tip]').forEach(el=>{const d=JSON.parse(el.dataset.tip);
const on=e=>{showTip(e.clientX!==undefined?e:{clientX:el.getBoundingClientRect().right,clientY:el.getBoundingClientRect().top},d.title,d.rows);
const cross=el.dataset.cross&&document.getElementById(el.dataset.cross);if(cross){cross.style.opacity=1;}};
const off=()=>{hideTip();const cross=el.dataset.cross&&document.getElementById(el.dataset.cross);if(cross){cross.style.opacity=0;}};
el.addEventListener('pointermove',on);el.addEventListener('pointerleave',off);el.addEventListener('focus',on);el.addEventListener('blur',off);});
"""


def _tip(title: str, rows: list[dict]) -> str:
    return E(json.dumps({"title": title, "rows": rows}), quote=True)


def line_chart(cid: str, title: str, caption: str, series: dict[str, list[dict]], key: str, ylabel: str) -> str:
    """Two-series step chart over cumulative rollouts, with a crosshair tooltip per round."""
    W, H, L, R, T, B = 480, 230, 44, 92, 14, 36
    colors = {"adaptive": "var(--s1)", "random": "var(--s2)"}
    names = {"adaptive": "Adaptive", "random": "Random"}
    xs_all = [p["rollouts"] for s in series.values() for p in s]
    ys_all = [p[key] for s in series.values() for p in s]
    xmax = max(xs_all) if xs_all else 1
    ymax = max(1, max(ys_all) if ys_all else 1)
    ymax = int(np.ceil(ymax * 1.1)) if key != "distinct_modes" else ymax + 1

    def sx(x: float) -> float:
        return L + (W - L - R) * x / xmax

    def sy(y: float) -> float:
        return H - B - (H - T - B) * y / ymax

    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(title)}">']
    ticks = np.linspace(0, ymax, 5) if key != "distinct_modes" else range(0, ymax + 1)
    for t in ticks:
        y = sy(t)
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{int(round(t))}</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    rounds = series.get("adaptive") or series.get("random") or []
    for p in rounds:
        out.append(f'<text x="{sx(p["rollouts"]):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">{p["rollouts"]}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 4}" text-anchor="middle" font-size="11" fill="var(--muted)">Rollouts run</text>')
    out.append(f'<line id="{cid}-x" x1="0" x2="0" y1="{T}" y2="{H - B}" stroke="var(--muted)" stroke-width="1" opacity="0"/>')
    for name, pts in series.items():
        if not pts:
            continue
        path = f"M{sx(0):.1f},{sy(0):.1f}"
        for p in pts:
            path += f" H{sx(p['rollouts']):.1f} V{sy(p[key]):.1f}"
        out.append(f'<path d="{path}" fill="none" stroke="{colors[name]}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pts:
            out.append(f'<circle cx="{sx(p["rollouts"]):.1f}" cy="{sy(p[key]):.1f}" r="4" fill="{colors[name]}" '
                       f'stroke="var(--surface)" stroke-width="2"/>')
        last = pts[-1]
        y_lab = sy(last[key]) + 4
        if name == "random" and series.get("adaptive"):
            gap = y_lab - (sy(series["adaptive"][-1][key]) + 4)
            if abs(gap) < 14:  # keep the two end labels from colliding
                y_lab += 14 if gap >= 0 else -14
        out.append(f'<text x="{W - R + 8}" y="{y_lab:.1f}" font-size="12" fill="var(--ink2)">'
                   f'{names[name]} {last[key]}</text>')
    # hover bands, one per round, carrying every series' value
    for i, p in enumerate(rounds):
        x0 = sx(rounds[i - 1]["rollouts"]) if i else sx(0)
        x1 = sx(p["rollouts"])
        rows = []
        for name, pts in series.items():
            if i < len(pts):
                q = pts[i]
                rows.append({"color": colors[name].replace("var(--s1)", "#2a78d6").replace("var(--s2)", "#eb6834"),
                             "value": str(q[key]), "label": f"{names[name]} · {q['gpu_hours']:.2f} GPU-h"})
        mid = (x0 + x1) / 2
        out.append(f'<rect x="{x0:.1f}" y="{T}" width="{x1 - x0:.1f}" height="{H - T - B}" fill="transparent" tabindex="0" '
                   f'data-cross="{cid}-x" data-tip="{_tip(f"Round {p["round"]} · {p["rollouts"]} rollouts", rows)}" '
                   f'onpointerenter="document.getElementById(\'{cid}-x\').setAttribute(\'x1\',{x1:.1f});'
                   f'document.getElementById(\'{cid}-x\').setAttribute(\'x2\',{x1:.1f})"/>')
    out.append("</svg>")
    legend = ('<div class="legend"><span><i style="background:var(--s1)"></i>Adaptive search</span>'
              '<span><i style="background:var(--s2)"></i>Random search, same budget</span></div>')
    return (f'<div class="card"><h3>{E(title)}</h3><div class="cap" style="margin:0">{E(ylabel)}</div>{legend}'
            f'{"".join(out)}<p class="cap">{E(caption)}</p></div>')


def heatmap(axes: list[Axis], scenarios: list[Scenario], failed: np.ndarray) -> str:
    sens = sensitivities(axes, scenarios, failed)
    by_axis = {a.name: a for a in axes}
    rows = []
    for s in sens:
        a = by_axis[s.axis]
        cells = []
        for b, rate in enumerate(s.failure_rate_by_bin):
            lo, hi = s.bin_edges[b], s.bin_edges[b + 1]
            rng_txt = f"{lo:.2f}-{hi:.2f} {a.unit}".strip()
            if rate is None:
                cells.append(f'<td class="cell" style="background:var(--seq0);color:var(--muted)" tabindex="0" '
                             f'data-tip="{_tip(f"{a.name} {rng_txt}", [{"value": "n/a", "label": "fewer than 3 rollouts"}])}">–</td>')
                continue
            step = min(7, int(rate / 0.6 * 7)) if rate > 0 else 0
            ink = "#ffffff" if step >= 4 else ("#0b0b0b" if step > 0 else "var(--ink)")
            bg = "var(--seq0)" if step == 0 else f"var(--seq{step})"
            cells.append(f'<td class="cell" style="background:{bg};color:{ink}" tabindex="0" '
                         f'data-tip="{_tip(f"{a.name} {rng_txt}", [{"value": f"{rate:.0%}", "label": "failure rate"}])}">{rate:.0%}</td>')
        rows.append(f'<tr><th>{E(a.name)}<div class="cap" style="margin:0">{a.low:g} to {a.high:g} {E(a.unit)}</div></th>'
                    f'{"".join(cells)}<td class="num">{s.spread:.0%}</td></tr>')
    head = "".join(f"<th style='text-align:center'>Bin {i + 1}</th>" for i in range(5))
    return (f'<div class="card"><div class="scroll"><table class="heat"><tr><th>Axis (low to high)</th>{head}<th class="num">Spread</th></tr>'
            f'{"".join(rows)}</table></div><p class="cap">Failure rate in five equal-width bins per axis, all rollouts pooled. '
            f'Sorted by spread (worst bin minus best bin): the axes at the top matter most. Adaptive sampling over-represents '
            f'failure regions, so read rates as relative, and use the robustness score above for the absolute number.</p></div>')


def _thumb(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img alt="Keyframe of {E(p.stem)}" src="data:image/png;base64,{data}">'


def build_report(run_dir: Path, out: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("aborted"):
        raise SystemExit(f"Run was aborted: {summary.get('reason')}")
    plan = StressPlan.model_validate_json((run_dir / "plan.json").read_text())
    rounds = json.loads((run_dir / "rounds.json").read_text())
    recs = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines()]
    recs = [r for r in recs if r["scenario"]["strategy"] != "sanity"]
    scen = [Scenario.model_validate(r["scenario"]) for r in recs]
    failed = np.array([r["label"]["failed"] for r in recs])
    frames = {r["scenario"]["id"]: r["result"]["frames"] for r in recs}

    spec = summary["spec"]
    rob = summary["robustness"]
    ca, cr = summary["curves"]["adaptive"], summary["curves"]["random"]
    la, lr = ca[-1], (cr[-1] if cr else None)

    banner = ""
    if summary["backend"] == "mock":
        banner = ('<div class="banner">Synthetic mock backend. These numbers test the pipeline itself and say nothing '
                  'about a real policy. Switch the backend to Isaac Lab-Arena for real results.</div>')
    offline = []
    st = summary["settings"]
    if not st["llm"]:
        offline.append("Nemotron calls (offline defaults used)")
    if not st["judge"]:
        offline.append("Cosmos Reason judge (simulator labels used)")
    if not st["tavily"]:
        offline.append("Tavily research")
    if offline:
        banner += f'<div class="banner">Not connected in this run: {E("; ".join(offline))}.</div>'

    def tile(k: str, v: str, d: str) -> str:
        return f'<div class="tile"><div class="k">{E(k)}</div><div class="v">{E(v)}</div><div class="d">{E(d)}</div></div>'

    tiles = [tile("Robustness score", f"{rob['success_rate']:.0%}",
                  f"95% CI {rob['ci95'][0]:.0%}-{rob['ci95'][1]:.0%} · {rob['n']} rollouts from the {rob['estimated_from']}")]
    if lr:
        tiles.append(tile("Failure types found", f"{la['distinct_modes']} vs {lr['distinct_modes']}", "adaptive vs random, same budget"))
        fa = la["failures"] / max(la["gpu_hours"], 1e-9)
        fr = lr["failures"] / max(lr["gpu_hours"], 1e-9)
        tiles.append(tile("Failures per GPU-hour", f"{fa:.1f} vs {fr:.1f}",
                          f"{fa / fr:.1f}x more with adaptive search" if fr > 0 else "adaptive vs random"))
        first_all = next((p["rollouts"] for p in ca if p["distinct_modes"] == la["distinct_modes"]), None)
        first_all_r = next((p["rollouts"] for p in cr if p["distinct_modes"] == la["distinct_modes"]), None)
        tiles.append(tile("Rollouts to find every type", f"{first_all} vs {first_all_r if first_all_r else 'not reached'}",
                          "adaptive vs random"))
    else:
        tiles.append(tile("Failure types found", str(la["distinct_modes"]), "adaptive search"))

    charts = ""
    if lr:
        charts = '<div class="charts">' + line_chart(
            "c1", "Distinct failure types discovered", "Timeouts are excluded from failure types: they are usually noise.",
            {"adaptive": ca, "random": cr}, "distinct_modes", "Cumulative count") + line_chart(
            "c2", "Failing scenarios found", "Each failing scenario is a test case you can replay or train on.",
            {"adaptive": ca, "random": cr}, "failures", "Cumulative count") + "</div>"

    cards = []
    noise = {"timeout", "other"}
    for c in sorted(summary["clusters"], key=lambda c: next(iter(c["categories"])) in noise):
        cat = next(iter(c["categories"]))
        conds = "".join(f'<span class="pill">{E(k)} {lo:.2f} to {hi:.2f}</span>' for k, (lo, hi) in c["bounds"].items()) \
            or '<span class="pill">no concentrated region</span>'
        thumbs = "".join(_thumb(f) for sid in c["example_ids"][:4] for f in frames.get(sid, [])[:1])
        cards.append(f'<div class="card"><h3>{E(c.get("title") or cat)}</h3>'
                     f'<div class="cap" style="margin:0 0 8px">{E(cat)} · {c["size"]} failing rollouts'
                     f'{" · scattered, likely background noise" if cat in noise and not c["bounds"] else ""}</div>{conds}'
                     f'<div class="thumbs">{thumbs}</div></div>')

    round_html = []
    for r in rounds:
        regs = "".join(f"<li>{E(g['hypothesis'] or 'Region')}: "
                       + E(", ".join(f"{k} {v[0]:.2f}-{v[1]:.2f}" for k, v in g["bounds"].items())) + "</li>"
                       for g in r["next_regions"])
        round_html.append(f'<div class="round"><b>Round {r["round"]}</b> <span class="cap">· {E(r["analyst_model"])}</span>'
                          f'<p class="sub" style="margin:4px 0">{E(r["summary"])}</p>'
                          + (f'<div class="cap">Next round probes:</div><ul class="cap">{regs}</ul>' if regs else "") + '</div>')

    axes_rows = "".join(f"<tr><td>{E(a.name)}</td><td>{E(a.kind)}</td><td class='num'>{a.low:g}</td><td class='num'>{a.high:g}</td>"
                        f"<td>{E(a.unit)}</td><td>{E(a.description)}</td></tr>" for a in plan.axes)
    sources = "".join(f'<li><a href="{E(u)}">{E(u)}</a></li>' for u in plan.sources)
    kfm = "".join(f"<li>{E(x)}</li>" for x in plan.known_failure_modes)

    ledger_rows = "".join(f"<tr><td>{E(m)}</td><td>{E(', '.join(v['purposes']))}</td><td class='num'>{v['calls']}</td>"
                          f"<td class='num'>{v['prompt_tokens'] + v['completion_tokens']:,}</td></tr>"
                          for m, v in summary["ledger"].items())
    if not ledger_rows:
        ledger_rows = "<tr><td colspan='4' class='cap'>No model calls in this run (offline mode).</td></tr>"
    models = summary["settings"]["models"]

    curve_rows = "".join(
        f"<tr><td>{E(name)}</td><td class='num'>{p['round']}</td><td class='num'>{p['rollouts']}</td>"
        f"<td class='num'>{p['gpu_hours']:.2f}</td><td class='num'>{p['failures']}</td><td class='num'>{p['distinct_modes']}</td>"
        f"<td>{E(', '.join(p['modes']))}</td></tr>"
        for name, pts in (("adaptive", ca), ("random", cr)) for p in pts)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stress Test Report</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">
<h1>Stress test report</h1>
<p class="sub">{E(spec['policy'].get('label') or spec['policy']['source'])} · {E(spec['policy']['embodiment'])} ·
“{E(spec['task']['instruction'])}”</p>
{banner}
<div class="tiles">{''.join(tiles)}</div>
<h2>Adaptive versus random search</h2>
<p class="sub">Both strategies ran the same number of rollouts. Hover a round for exact values.</p>
{charts}
<h2>Failure types</h2>
<p class="sub">Grouped by the judge's failure category, with the parameter ranges where each one concentrates.</p>
<div class="clusters">{''.join(cards) or '<p class="cap">No failures found.</p>'}</div>
<h2>What the policy is sensitive to</h2>
{heatmap(plan.axes, scen, failed)}
<h2>How the search steered itself</h2>
<p class="sub">After each round the analyst summarised the failures and chose regions for the next round.</p>
{''.join(round_html)}
<h2>Test plan</h2>
<p class="sub">{E(plan.rationale)} <span class="cap">· planned by {E(plan.planner_model)}</span></p>
<div class="card scroll"><table><tr><th>Axis</th><th>Kind</th><th class="num">Low</th><th class="num">High</th><th>Unit</th><th>Why</th></tr>{axes_rows}</table></div>
{f'<div class="card" style="margin-top:12px"><h3>Known failure modes from research</h3><ul class="cap">{kfm}</ul></div>' if kfm else ''}
{f'<div class="card" style="margin-top:12px"><h3>Sources (Tavily)</h3><ul class="cap">{sources}</ul></div>' if sources else ''}
<h2>Models used</h2>
<p class="sub">Routing: Ultra plans ({E(models['ultra'])}), Super analyses each round ({E(models['super'])}),
Nano names clusters ({E(models['nano'])}), Cosmos Reason 2 judges rollout video.</p>
<div class="card scroll"><table><tr><th>Model</th><th>Used for</th><th class="num">Calls</th><th class="num">Tokens</th></tr>{ledger_rows}</table></div>
<details><summary>Data table for the charts</summary><div class="card scroll" style="margin-top:8px"><table>
<tr><th>Strategy</th><th class="num">Round</th><th class="num">Rollouts</th><th class="num">GPU-h</th><th class="num">Failures</th>
<th class="num">Types</th><th>Types found</th></tr>{curve_rows}</table></div></details>
</div><div id="tip" role="tooltip"></div></div><script>{JS}</script></body></html>"""
    out = Path(out) if out else run_dir / "report.html"
    out.write_text(page)
    return out
