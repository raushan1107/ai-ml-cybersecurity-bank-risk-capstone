"""
frontend/views/7_🗺️_AI_ML_Roadmap.py

The whole AI/ML learning path as an interactive tree: 12 stages → 56 topics.
Click a topic to reveal its real-world uses, an analogy, the maths, code, where
it lives in CAMEL Sentinel and its security angle. Content lives in
frontend/roadmap_content/; design in docs/08_ai_ml_roadmap.md.
Stages are named by what they teach — never "Module N".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roadmap_content import TRACKS, load_curriculum  # noqa: E402
from ui_kit import HANDBOOK_REPO, HANDBOOK_URL, SERIES, html, stage_url  # noqa: E402

st.set_page_config(page_title="AI/ML Roadmap · CAMEL Sentinel", page_icon="🗺️", layout="wide")

stages = load_curriculum()
for s in stages:
    s["url"] = stage_url(s["n"])

st.title("🗺️ The AI/ML Roadmap")
st.caption("The complete learning path behind CAMEL Sentinel — 12 stages, "
           f"{sum(len(s['topics']) for s in stages)} topics, each shown working in the real world and in this project.")
st.info(f"📘 This roadmap is the **map**; the [**AI/ML + Cybersecurity Handbook**]({HANDBOOK_URL}) "
        f"([GitHub]({HANDBOOK_REPO})) by **[Raushan Ranjan](https://raushan-ranjan.azurewebsites.net)**, the developer of this project, is the **textbook**. "
        "Every topic card links to the handbook chapter where it is taught step by step.")

track_colors = dict(zip(TRACKS, [SERIES[0], SERIES[1], SERIES[2], "#eda100"]))
DATA = json.dumps({"stages": stages, "tracks": track_colors}, ensure_ascii=False).replace("</", "<\\/")

PAGE = r"""
<style>
:root{--c0:#2a78d6;--card:var(--bg)}
body{padding:2px}
.top{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{border:0;background:transparent;color:var(--ink);padding:7px 14px;cursor:pointer;font:inherit}
.seg button.on{background:var(--accent);color:#fff}
#q{flex:1 1 180px;min-width:150px;padding:8px 12px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);font:inherit}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:12px;padding:3px 10px;border-radius:999px;border:1.5px solid;cursor:pointer;user-select:none;background:var(--panel)}
.chip.off{opacity:.35}
.chip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.wrap{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:14px;height:780px}
.pane{border:1px solid var(--line);border-radius:14px;background:var(--panel);overflow:auto;position:relative}
/* ---------- tree ---------- */
#tree{position:relative;padding:14px 16px 14px 10px;min-width:500px}
#links{position:absolute;inset:0;pointer-events:none;overflow:visible}
#links path{fill:none;stroke-width:1.6;opacity:.55;stroke-dasharray:600;stroke-dashoffset:600;animation:draw .7s ease forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.cols{display:grid;grid-template-columns:96px 190px minmax(150px,1fr);column-gap:26px;align-items:start}
.root{position:sticky;top:40%;padding:12px;border-radius:14px;background:var(--accent);color:#fff;text-align:center;font-weight:700;font-size:13px}
.root small{display:block;font-weight:400;opacity:.9}
.stage{display:flex;gap:8px;align-items:center;padding:7px 10px;margin:5px 0;border-radius:11px;background:var(--bg);border:1.5px solid var(--line);
 cursor:pointer;transition:transform .15s,box-shadow .15s;border-left-width:5px}
.stage:hover{transform:translateX(3px)} .stage.open{box-shadow:0 4px 14px rgba(0,0,0,.14)}
.stage .num{font-weight:800;font-size:12px;color:var(--muted);min-width:18px}
.stage .nm{font-size:12.5px;line-height:1.25}
.stage .ct{margin-left:auto;font-size:11px;color:var(--muted)}
.topics{display:flex;flex-direction:column;gap:5px}
.topic{padding:7px 11px;border-radius:10px;background:var(--bg);border:1px solid var(--line);cursor:pointer;font-size:12.5px;
 opacity:0;transform:translateX(-10px);animation:pop .35s ease forwards;border-left:4px solid}
.topic:hover{border-color:var(--accent)} .topic.sel{outline:2px solid var(--accent)}
.topic.hit{box-shadow:0 0 0 3px color-mix(in srgb,#eda100 55%,transparent)}
.group{margin-bottom:10px}
.glabel{font-size:11px;color:var(--muted);margin:4px 0 2px;text-transform:uppercase;letter-spacing:.04em}
@keyframes pop{to{opacity:1;transform:none}}
.stage.dim,.topic.dim{opacity:.25}
/* ---------- journey ---------- */
#journey{padding:16px;display:none}
.path{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}
.jc{position:relative;padding:12px;border-radius:14px;background:var(--bg);border:1px solid var(--line);border-top:5px solid;cursor:pointer;
 opacity:0;animation:pop .45s ease forwards}
.jc:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.14)}
.jc .big{font-size:26px}.jc .n{position:absolute;top:10px;right:12px;font-weight:800;color:var(--muted)}
.jc h4{margin:4px 0;font-size:14px}.jc p{margin:4px 0;font-size:12px;color:var(--muted)}
.jc .hand{font-size:11.5px;border-top:1px dashed var(--line);padding-top:6px;margin-top:6px}
.arrow{font-size:11px;color:var(--muted);text-align:center;margin-top:6px}
/* ---------- card ---------- */
#card{padding:16px}
.empty{color:var(--muted);text-align:center;margin-top:34%}
.crumb{font-size:12px;color:var(--muted)}
#card h2{margin:4px 0 6px;font-size:20px}
.one{font-size:14px;margin:0 0 10px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}
.tabs button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:5px 9px;cursor:pointer;font:inherit;font-size:12px}
.tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.body{animation:pop .3s ease forwards;opacity:0}
.body pre{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;white-space:pre;font-size:12.5px;line-height:1.5}
.rw{border-left:3px solid var(--accent);padding:6px 10px;margin:8px 0;background:var(--bg);border-radius:6px}
.nav{display:flex;justify-content:space-between;margin-top:14px;gap:8px}
.nav button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;font-size:12px}
a{color:var(--accent)}
@media (max-width:760px){
 .wrap{grid-template-columns:1fr;height:auto}
 .pane{max-height:560px}
 #tree{min-width:0}
 .cols{grid-template-columns:1fr;row-gap:8px}
 .root{position:static}
 #links{display:none}
}
</style>
<div class="top">
  <div class="seg"><button id="vTree" class="on" onclick="view('tree')">🌳 Tree</button><button id="vJour" onclick="view('journey')">🧭 Journey</button></div>
  <input id="q" placeholder="Search topics — e.g. attention, SHAP, Docker, deepfake…" oninput="search(this.value)">
  <div class="chips" id="chips"></div>
</div>
<div class="wrap">
  <div class="pane" id="left">
    <div id="tree"><svg id="links"></svg>
      <div class="cols"><div><div class="root" id="root">AI/ML + Cyber<small>learning path</small></div></div>
      <div id="stageCol"></div><div id="topicCol" class="topics"></div></div>
    </div>
    <div id="journey"><div class="path" id="path"></div></div>
  </div>
  <div class="pane" id="card"><div class="empty">👈 Open a stage, then click a topic<br><br>
  <span class="muted">Each card shows: one-line definition · real-world uses · analogy · the maths · code · where it lives in CAMEL Sentinel · security angle</span></div></div>
</div>
<script>
const D = __DATA__;
const S = D.stages, TC = D.tracks;
const open = new Set(["1"]); let sel = null, activeTracks = new Set(Object.keys(TC)), q = "";
const LENSES = [["one","💡 In one line"],["rw","🌍 Real world"],["an","🧩 Analogy"],["math","∑ Math"],["code","⌨️ Code"],["proj","🐫 In CAMEL Sentinel"],["sec","🛡️ Security angle"]];
let lens = "rw";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const flat = []; S.forEach(s => s.topics.forEach(t => flat.push([s, t])));

function chips(){ document.getElementById('chips').innerHTML = Object.entries(TC).map(([k,c]) =>
  `<span class="chip ${activeTracks.has(k)?'':'off'}" style="border-color:${c}" onclick="toggleTrack('${k}')"><i style="background:${c}"></i>${esc(k)}</span>`).join(''); }
function toggleTrack(k){ activeTracks.has(k) ? activeTracks.delete(k) : activeTracks.add(k); if(!activeTracks.size) activeTracks = new Set(Object.keys(TC)); render(); }
function match(t){ if(!q) return false; const h=(t.name+' '+t.oneliner+' '+t.real_world.map(r=>r.join(' ')).join(' ')).toLowerCase(); return h.includes(q); }

function render(){
  chips();
  const sc = document.getElementById('stageCol'), tc = document.getElementById('topicCol');
  sc.innerHTML = S.map(s => {
    const hits = s.topics.filter(match).length, dim = !activeTracks.has(s.track) || (q && !hits);
    return `<div class="stage ${open.has(s.n)?'open':''} ${dim?'dim':''}" id="st${s.n}" style="border-left-color:${TC[s.track]}" onclick="toggleStage('${s.n}')">
      <span class="num">${s.n}</span><span>${s.icon}</span><span class="nm">${esc(s.name)}</span><span class="ct">${hits?('🔎'+hits):s.topics.length}</span></div>`;}).join('');
  let delay = 0;
  tc.innerHTML = S.filter(s => open.has(s.n) && activeTracks.has(s.track)).map(s =>
    `<div class="group"><div class="glabel">${s.n} · ${esc(s.name)}</div>` + s.topics.map(t => {
      const d = (delay++)*0.04;
      return `<div class="topic ${sel===t.id?'sel':''} ${match(t)?'hit':''} ${q&&!match(t)?'dim':''}" id="tp_${t.id}" data-stage="${s.n}"
        style="border-left-color:${TC[s.track]};animation-delay:${d}s" onclick="pick('${t.id}')">${esc(t.name)}</div>`;}).join('') + `</div>`).join('')
    || `<div class="muted" style="margin-top:40px">Click a stage to grow its branches 🌱</div>`;
  requestAnimationFrame(links);
}
function links(){
  const svg = document.getElementById('links'), box = document.getElementById('tree').getBoundingClientRect();
  if (getComputedStyle(svg).display === 'none') return;
  svg.setAttribute('width', box.width); svg.setAttribute('height', document.getElementById('tree').scrollHeight);
  const pt = (el, side) => { const r = el.getBoundingClientRect(); return [side==='r' ? r.right-box.left : r.left-box.left, r.top-box.top + r.height/2]; };
  const curve = ([x1,y1],[x2,y2]) => `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`;
  const root = document.getElementById('root'); let p = '';
  S.forEach(s => { const el = document.getElementById('st'+s.n); if(!el) return;
    p += `<path d="${curve(pt(root,'r'), pt(el,'l'))}" stroke="${TC[s.track]}"/>`;
    if (open.has(s.n)) s.topics.forEach(t => { const te = document.getElementById('tp_'+t.id); if(te) p += `<path d="${curve(pt(el,'r'), pt(te,'l'))}" stroke="${TC[s.track]}"/>`; });
  });
  svg.innerHTML = p;
}
function toggleStage(n){ open.has(n) ? open.delete(n) : open.add(n); render(); }
function pick(id){
  sel = id; const [s, t] = flat.find(([,t]) => t.id === id);
  open.add(s.n); render(); card(s, t);
  const el = document.getElementById('tp_'+id); if (el) el.scrollIntoView({block:'nearest', behavior:'smooth'});
}
function lensBody(s, t){
  switch(lens){
    case 'one': return `<p class="one">${esc(t.oneliner)}</p><p class="muted">Part of <b>${esc(s.name)}</b> — ${esc(s.tagline)}</p><p><b>Hands over to the next stage:</b> ${esc(s.handoff)}</p>`;
    case 'rw': return t.real_world.map(([w, x]) => `<div class="rw"><b>${esc(w)}</b> — ${esc(x)}</div>`).join('');
    case 'an': return `<p style="font-size:15px">🧩 ${esc(t.analogy)}</p>`;
    case 'math': return `<pre>${esc(t.math)}</pre>`;
    case 'code': return `<pre>${esc(t.code)}</pre>`;
    case 'proj': return `<p>🐫 ${esc(t.in_project)}</p>`;
    case 'sec': return `<p>🛡️ ${esc(t.security)}</p><p class="muted">Try the attacks and defences live on the 🛡️ AI Security Lab page.</p>`;
  }
}
function card(s, t){
  const i = flat.findIndex(([,x]) => x.id === t.id), prev = flat[i-1], next = flat[i+1];
  document.getElementById('card').innerHTML = `
    <div class="crumb"><span style="color:${TC[s.track]}">●</span> ${esc(s.track)} · ${s.n} · ${esc(s.name)}</div>
    <h2>${s.icon} ${esc(t.name)}</h2><p class="one">${esc(t.oneliner)}</p>
    <div class="tabs">${LENSES.map(([k,l]) => `<button class="${lens===k?'on':''}" onclick="setLens('${k}')">${l}</button>`).join('')}</div>
    <div class="body" id="lb">${lensBody(s, t)}</div>
    <p style="margin-top:14px">📘 <a href="${s.url}" target="_blank" rel="noopener">Learn it step by step in the handbook → ${esc(s.name)}</a></p>
    <div class="nav"><button ${prev?'':'disabled'} onclick="pick('${prev?prev[1].id:''}')">← ${prev?esc(prev[1].name):''}</button>
    <button ${next?'':'disabled'} onclick="pick('${next?next[1].id:''}')">${next?esc(next[1].name):''} →</button></div>`;
}
function setLens(k){ lens = k; const [s,t] = flat.find(([,x]) => x.id === sel); card(s, t); }
function search(v){ q = v.trim().toLowerCase(); if(q){ S.forEach(s => { if (s.topics.some(match)) open.add(s.n); }); } render(); }
function view(v){
  document.getElementById('tree').style.display = v==='tree' ? 'block' : 'none';
  document.getElementById('journey').style.display = v==='journey' ? 'block' : 'none';
  document.getElementById('vTree').classList.toggle('on', v==='tree'); document.getElementById('vJour').classList.toggle('on', v==='journey');
  if (v==='tree') requestAnimationFrame(links);
}
function journey(){
  document.getElementById('path').innerHTML = S.map((s, i) => `
    <div class="jc" style="border-top-color:${TC[s.track]};animation-delay:${i*0.06}s" onclick="view('tree');open.add('${s.n}');pick('${s.topics[0].id}')">
      <span class="n">${s.n}</span><div class="big">${s.icon}</div><h4>${esc(s.name)}</h4><p>${esc(s.tagline)}</p>
      <div class="hand">➜ ${esc(s.handoff)}</div></div>`).join('');
}
window.addEventListener('resize', () => requestAnimationFrame(links));
document.getElementById('left').addEventListener('scroll', () => requestAnimationFrame(links));
journey(); render();
</script>
"""

html(PAGE.replace("__DATA__", DATA), height=860)

st.subheader("How the stages connect in this project")
st.dataframe(pd.DataFrame([{"#": s["n"], "Stage": f"{s['icon']} {s['name']}", "Track": s["track"],
                            "Topics": len(s["topics"]), "Hands over": s["handoff"]} for s in stages]),
             width="stretch", hide_index=True)
st.caption("Content source: frontend/roadmap_content/ · Design: docs/08_ai_ml_roadmap.md · "
           "Real-world examples are limited to publicly documented products, papers and incidents.")
