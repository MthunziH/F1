"""Builds the static, phone-friendly results page (site/index.html)."""
from __future__ import annotations

import json
from html import escape

from f1predict import config
from f1predict.predictor import live_summary, load_predictions, utc_now_iso

TEAM_COLORS = {
    "red_bull": "#3671C6", "mercedes": "#27F4D2", "ferrari": "#E8002D",
    "mclaren": "#FF8000", "aston_martin": "#229971", "alpine": "#FF87BC",
    "williams": "#64C4FF", "rb": "#6692FF", "alphatauri": "#5E8FAA",
    "sauber": "#52E252", "alfa": "#C92D4B", "audi": "#9A9A9A", "haas": "#B6BABD",
    "cadillac": "#C8A951", "racing_point": "#F596C8", "renault": "#FFF500",
    "force_india": "#F596C8", "toro_rosso": "#469BFF",
}


def _color(team_id) -> str:
    return TEAM_COLORS.get(str(team_id), "#888")


def _pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p < 0.01:
        return "<1%"
    return f"{p:.0%}"


def _time(iso: str | None, label: str = "") -> str:
    if not iso:
        return ""
    return f'<time datetime="{escape(iso)}">{escape(label or iso.replace("T", " ").replace("Z", " UTC"))}</time>'


def _driver(rec, code):
    return next((d for d in rec["drivers"] if d["code"] == code), {"code": code, "name": code, "team": "", "teamId": ""})


def _podium_card(rec: dict) -> str:
    medals = ["P1", "P2", "P3"]
    actual = (rec.get("actual") or {}).get("podium")
    rows = []
    for i, code in enumerate(rec["podium"]):
        d = _driver(rec, code)
        mark = ""
        if actual:
            ok = actual[i] == code
            mark = f'<span class="mark {"ok" if ok else ("near" if code in actual else "no")}">{"✓" if ok else ("~" if code in actual else "✗")}</span>'
        rows.append(f"""
      <li class="pod pod{i+1}" style="--team:{_color(d['teamId'])}">
        <span class="pos">{medals[i]}</span>
        <span class="who"><strong>{escape(d['name'])}</strong><small>{escape(d['team'])} · starts P{d.get('quali_pos') or '?'}</small></span>
        <span class="odds"><b>{_pct(d['p_win'])}</b><small>win</small></span>
        <span class="odds"><b>{_pct(d['p_podium'])}</b><small>podium</small></span>
        {mark}
      </li>""")
    result = ""
    if actual:
        s = rec["score"]
        names = " · ".join(escape(_driver(rec, c)["name"].split()[-1]) for c in actual)
        verdict = "Exact podium 🎯" if s["exact"] else f"Winner {'right ✓' if s['winner'] else 'wrong ✗'} · {s['podium_hits']}/3 on podium"
        result = f'<div class="result"><span>Actual: <strong>{names}</strong></span><span class="chip">{verdict}</span></div>'
    return f"""
  <section class="card hero">
    <p class="eyebrow">{rec['season']} · Round {rec['round']}</p>
    <h2>{escape(rec['eventName'])}</h2>
    <p class="meta">Predicted after qualifying · {_time(rec.get('generated_utc'))}</p>
    <ol class="podium">{''.join(rows)}
    </ol>{result}
  </section>"""


def _all_drivers(rec: dict) -> str:
    rows = []
    for d in sorted(rec["drivers"], key=lambda d: (-d["p_win"], -d["p_podium"])):
        rows.append(f"""
      <tr style="--team:{_color(d['teamId'])}">
        <td class="num">{d.get('quali_pos') or '–'}</td>
        <td class="drv"><span class="stripe"></span><span><strong>{escape(d['code'])}</strong> <small>{escape(d['team'])}</small></span></td>
        <td class="bar"><span style="--w:{d['p_win']*100:.1f}%"></span><em>{_pct(d['p_win'])}</em></td>
        <td class="bar pod"><span style="--w:{d['p_podium']*100:.1f}%"></span><em>{_pct(d['p_podium'])}</em></td>
      </tr>""")
    return f"""
  <section class="card">
    <h3>Every driver's chances</h3>
    <table class="grid">
      <thead><tr><th class="num">Grid</th><th>Driver</th><th>Win</th><th>Podium</th></tr></thead>
      <tbody>{''.join(rows)}
      </tbody>
    </table>
  </section>"""


def _stat(value, label) -> str:
    return f'<div class="stat"><b>{value}</b><small>{label}</small></div>'


def _season_record(recs: list[dict]) -> str:
    summ = live_summary(recs)
    if not recs:
        return ""
    stats = ""
    if summ:
        stats = f"""<div class="stats">
      {_stat(f"{summ['winners_right']}/{summ['races']}", "winners right")}
      {_stat(f"{summ['podium_hits_avg']:.1f}/3", "podium drivers right per race")}
      {_stat(f"{summ['exact_pct']:.0f}%", "exact podiums")}
    </div>"""
    items = []
    for r in reversed(recs):
        a = (r.get("actual") or {}).get("podium")
        if a:
            s = r["score"]
            badge = f'<span class="chip {"good" if s["winner"] else ""}">{s["podium_hits"]}/3{" · winner ✓" if s["winner"] else ""}</span>'
            actual = f"<small>Actual: {' · '.join(a)}</small>"
        else:
            badge, actual = '<span class="chip pending">awaiting race</span>', ""
        items.append(f"""
      <li><span><strong>{escape(r['eventName'])}</strong><small>Predicted: {' · '.join(r['podium'])}</small>{actual}</span>{badge}</li>""")
    return f"""
  <section class="card">
    <h3>This season's record</h3>
    <p class="meta">Live predictions made before each race, scored after it.</p>
    {stats}
    <ul class="list">{''.join(items)}
    </ul>
  </section>"""


def _backtest() -> str:
    if not config.BACKTEST_JSON.exists():
        return ""
    bt = json.loads(config.BACKTEST_JSON.read_text())
    o = bt["summary"]["overall"]
    if not o["model"]:
        return ""
    m, b = o["model"], o["baseline"]
    rows = [f"""<tr><th>All</th><td>{m['races']}</td><td>{m['winner_pct']:.0f}% <small>vs {b['winner_pct']:.0f}%</small></td><td>{m['podium_hits_avg']:.2f} <small>vs {b['podium_hits_avg']:.2f}</small></td></tr>"""]
    for season, s in bt["summary"]["by_season"].items():
        sm, sb = s["model"], s["baseline"]
        rows.append(f"""<tr><th>{season}</th><td>{sm['races']}</td><td>{sm['winner_pct']:.0f}% <small>vs {sb['winner_pct']:.0f}%</small></td><td>{sm['podium_hits_avg']:.2f} <small>vs {sb['podium_hits_avg']:.2f}</small></td></tr>""")
    races = []
    for r in reversed(bt["races"]):
        s = r["model"]
        races.append(f"""<li><span><strong>{r['season']} {escape(r['eventName'])}</strong><small>Predicted: {' · '.join(r['predicted'])}</small><small>Actual: {' · '.join(r['actual'])}</small></span><span class="chip {"good" if s["winner"] else ""}">{s['podium_hits']}/3{" · winner ✓" if s["winner"] else ""}</span></li>""")
    return f"""
  <section class="card">
    <h3>Backtest: how it would have done</h3>
    <p class="meta">Each race since {bt['from_season']} replayed using only data available before it. Compared with simply taking the top 3 from qualifying ("vs").</p>
    <div class="stats">
      {_stat(f"{m['winner_pct']:.0f}%", "winners right")}
      {_stat(f"{m['podium_hits_avg']:.2f}/3", "podium drivers right")}
      {_stat(f"{m['exact_pct']:.0f}%", "exact podiums")}
    </div>
    <table class="bt"><thead><tr><th></th><th>Races</th><th>Winner</th><th>Podium /3</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <details><summary>Race by race</summary><ul class="list">{''.join(races)}</ul></details>
  </section>"""


def _upcoming(next_event: dict | None, latest: dict | None) -> str:
    if not next_event:
        return ""
    if latest and latest["season"] == next_event["season"] and latest["round"] == next_event["round"]:
        return ""
    return f"""
  <section class="card next">
    <p class="eyebrow">Next up · Round {next_event['round']}</p>
    <h2>{escape(next_event['eventName'])}</h2>
    <p class="meta">Prediction appears here soon after qualifying ends: {_time(next_event.get('quali_utc'))}</p>
  </section>"""


def build_site(next_event: dict | None = None) -> None:
    recs = load_predictions()
    latest = recs[-1] if recs else None
    body = _upcoming(next_event, latest)
    if latest:
        body += _podium_card(latest) + _all_drivers(latest)
    body += _season_record(recs) + _backtest()
    body += """
  <section class="card about">
    <h3>How it works</h3>
    <p>After qualifying, a gradient-boosting model trained on every race since 2018 rates each driver using:
    qualifying position and gap to pole, teammate comparison, recent driver and team form, reliability,
    season points, and past results at this track. That's blended 50/50 with how often each grid slot has historically
    won or reached the podium, which stops the model from over-ruling the pole sitter on form alone.
    It outputs each driver's chance to win and to finish on the podium.
    Data: <a href="https://github.com/theOehrly/Fast-F1">FastF1</a>.</p>
    <p>Grid penalties announced after qualifying aren't included, and nobody can predict crashes, rain or safety cars, so treat these as odds, not certainties.</p>
  </section>"""

    html = TEMPLATE.replace("{{BODY}}", body).replace("{{UPDATED}}", _time(utc_now_iso()))
    config.SITE_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DIR / "index.html").write_text(html)
    (config.SITE_DIR / ".nojekyll").write_text("")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>F1 Podium Predictor</title>
<meta name="theme-color" content="#e10600">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏁</text></svg>">
<style>
:root{--bg:#f4f4f6;--card:#fff;--ink:#15151e;--muted:#6b6b78;--line:#e6e6ec;--accent:#e10600;--good:#1a7f37;--warn:#b26a00;--track:#ececf1}
@media (prefers-color-scheme:dark){:root{--bg:#0f0f14;--card:#1a1a22;--ink:#f2f2f5;--muted:#9a9aa8;--line:#2a2a35;--accent:#ff3b30;--good:#3fb950;--warn:#d29922;--track:#2a2a35}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-text-size-adjust:100%}
header{background:var(--accent);color:#fff;padding:calc(14px + env(safe-area-inset-top)) 16px 14px}
header h1{margin:0;font-size:20px;letter-spacing:.2px}
header p{margin:2px 0 0;opacity:.85;font-size:13px}
main{max-width:640px;margin:0 auto;padding:12px 12px 32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:12px 0}
h2{margin:2px 0 4px;font-size:22px}
h3{margin:0 0 8px;font-size:17px}
.eyebrow{margin:0;color:var(--accent);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.6px}
.meta{margin:0 0 10px;color:var(--muted);font-size:13px}
small{color:var(--muted)}
.podium{list-style:none;margin:12px 0 0;padding:0;display:grid;gap:8px}
.pod{display:grid;grid-template-columns:40px 1fr auto auto auto;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;background:var(--bg);border-left:5px solid var(--team)}
.pos{font-weight:800;font-size:18px}
.pod1 .pos{color:#c9a227}.pod2 .pos{color:#9aa0a6}.pod3 .pos{color:#b06d3a}
.who{display:flex;flex-direction:column;min-width:0}
.who strong{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.who small{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.odds{display:flex;flex-direction:column;align-items:flex-end;min-width:44px}
.odds b{font-variant-numeric:tabular-nums}
.odds small{font-size:11px}
.mark{font-weight:800;width:18px;text-align:center}
.mark.ok{color:var(--good)}.mark.near{color:var(--warn)}.mark.no{color:var(--accent)}
.result{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;margin-top:12px;font-size:14px}
.chip{display:inline-block;padding:3px 9px;border-radius:999px;background:var(--track);font-size:12px;font-weight:600;white-space:nowrap}
.chip.good{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.chip.pending{color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;padding:6px 4px;border-bottom:1px solid var(--line)}
td{padding:7px 4px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:0}
.num{text-align:center;width:38px;font-variant-numeric:tabular-nums}
.drv{display:flex;align-items:center;gap:8px}
.drv small{font-size:11px}
.stripe{width:4px;height:22px;border-radius:2px;background:var(--team);flex:none}
.bar{position:relative;width:28%;min-width:84px}
.bar span{display:block;height:20px;border-radius:5px;background:var(--team);opacity:.85;width:max(var(--w),2px)}
.bar.pod span{opacity:.45}
.bar em{position:absolute;right:6px;top:50%;transform:translateY(-50%);font-style:normal;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 12px}
.stat{background:var(--bg);border-radius:10px;padding:10px;text-align:center}
.stat b{display:block;font-size:20px}
.stat small{font-size:11px;line-height:1.2;display:block}
.list{list-style:none;margin:0;padding:0}
.list li{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
.list li:last-child{border-bottom:0}
.list li>span:first-child{display:flex;flex-direction:column;min-width:0}
.list small{font-size:12px}
.bt td small{font-size:11px}
details summary{cursor:pointer;color:var(--accent);font-weight:600;margin-top:10px}
.about p{font-size:14px;color:var(--muted);margin:0 0 8px}
a{color:var(--accent)}
footer{text-align:center;color:var(--muted);font-size:12px;padding:0 16px 24px}
@media (max-width:400px){.pod{grid-template-columns:28px 1fr auto auto auto;gap:7px;padding:9px 10px}.odds{min-width:38px}.bar{min-width:70px}}
</style>
</head>
<body>
<header><h1>🏁 F1 Podium Predictor</h1><p>Who finishes 1-2-3, predicted after qualifying</p></header>
<main>{{BODY}}
</main>
<footer>Updated {{UPDATED}}</footer>
<script>
document.querySelectorAll('time[datetime]').forEach(function(t){
  var d=new Date(t.getAttribute('datetime'));
  if(!isNaN(d)) t.textContent=d.toLocaleString(undefined,{weekday:'short',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});
});
</script>
</body>
</html>
"""
