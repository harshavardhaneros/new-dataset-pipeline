#!/usr/bin/env python3
"""Build an interactive HTML metric-distribution dashboard from a run's metadata.jsonl.

For each quality metric (Final, Aes, Tech, DOVER, Mot, CLIP, ICR, AOD) it renders a
histogram binned into 0-0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8 / 0.8-1.0. Clicking any bar
lists every clip whose score falls in that range.

Usage:
    python scripts/metric_distribution_report.py <metadata.jsonl | movie_dir> [-o out.html]

Examples:
    python scripts/metric_distribution_report.py \
        /mnt/data0/harsha/new_dataset_pipeline/v3_outputs_dense/Devdas_dense
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# label -> field name in metadata.jsonl
METRICS = [
    ("Final", "final_score"),
    ("Aes", "aesthetic_score"),
    ("Tech", "technical_score"),
    ("DOVER", "dover_score"),
    ("Mot", "motion_score"),
    ("CLIP", "clip_score"),
    ("ICR", "icr"),
    ("AOD", "aod"),
]


def resolve_path(p: Path) -> Path:
    if p.is_dir():
        cand = p / "metadata.jsonl"
        if cand.exists():
            return cand
        raise SystemExit(f"No metadata.jsonl found in {p}")
    return p


def load_records(path: Path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def to_clip(rec, movie, video):
    cap = rec.get("caption")
    if isinstance(cap, dict):
        cap = cap.get("text") or cap.get("caption") or json.dumps(cap)
    cap = (cap or "").strip()
    cid = rec.get("clip_id", "")
    return {
        "id": cid,
        "movie": movie,
        "video": video,
        "verdict": rec.get("verdict", ""),
        "keep": bool(rec.get("keep", True)),
        "ts": [rec.get("timestamp_start"), rec.get("timestamp_end")],
        "cap": cap,
        "s": {label: round(float(rec.get(field, 0) or 0), 4) for label, field in METRICS},
    }


def discover_movies(root: Path):
    """Return [(movie_name, metadata_path)] for every subdir with a metadata.jsonl."""
    found = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            md = sub / "metadata.jsonl"
            if md.exists():
                found.append((sub.name, md))
    return found


def build_html(clips, title, movies):
    payload = json.dumps(
        {
            "clips": clips,
            "metrics": [m[0] for m in METRICS],
            "title": title,
            "movies": movies,
        },
        separators=(",", ":"),
    )
    return HTML_TEMPLATE.replace("/*__DATA__*/", payload).replace("__TITLE__", title)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Metric Distributions</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --edge:#2a3240;
    --txt:#e6edf3; --mut:#8b949e; --accent:#58a6ff; --accent2:#3fb950;
    --bar:#3b6ea5; --barhi:#58a6ff; --warn:#d29922;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{padding:20px 28px;border-bottom:1px solid var(--edge);background:var(--panel)}
  h1{margin:0;font-size:20px;font-weight:650}
  .sub{color:var(--mut);font-size:13px;margin-top:4px}
  .controls{display:flex;gap:18px;align-items:center;margin-top:14px;flex-wrap:wrap}
  .controls label{color:var(--mut);font-size:13px;cursor:pointer;user-select:none}
  .controls input{vertical-align:middle;margin-right:5px}
  .controls select{background:var(--panel2);color:var(--txt);border:1px solid var(--edge);
    border-radius:6px;padding:4px 8px;font-size:13px;margin-left:4px}
  #movieWrap{color:var(--txt)}
  .stat{color:var(--accent2);font-weight:600}
  main{padding:24px 28px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
  .card{background:var(--panel);border:1px solid var(--edge);border-radius:10px;padding:14px 16px}
  .card h3{margin:0 0 2px;font-size:15px;display:flex;justify-content:space-between;align-items:baseline}
  .card h3 .avg{color:var(--mut);font-size:12px;font-weight:500}
  .chart{width:100%;height:170px;display:block;margin-top:6px}
  .chart rect.bar{fill:var(--bar);cursor:pointer;transition:fill .12s}
  .chart rect.bar:hover{fill:var(--barhi)}
  .chart rect.bar.sel{fill:var(--accent2)}
  .chart text{fill:var(--mut);font-size:10px}
  .chart text.cnt{fill:var(--txt);font-size:11px;font-weight:600}
  /* clip list panel */
  #panel{margin-top:26px;background:var(--panel);border:1px solid var(--edge);border-radius:10px;overflow:hidden}
  #panelHead{padding:12px 16px;border-bottom:1px solid var(--edge);background:var(--panel2);
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
  #panelHead .t{font-weight:600}
  #panelHead .t span{color:var(--accent)}
  #panelHead .hint{color:var(--mut);font-size:12px}
  .tablewrap{overflow-x:auto;max-height:60vh;overflow-y:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--edge);white-space:nowrap}
  th{position:sticky;top:0;background:var(--panel2);color:var(--mut);font-weight:600;cursor:pointer}
  td.cap{white-space:normal;min-width:280px;max-width:460px;color:var(--mut)}
  td.score{font-variant-numeric:tabular-nums;font-weight:600}
  .pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600}
  .v-FINAL{background:#16321f;color:#3fb950}
  .v-REVIEW{background:#3a2f12;color:#d29922}
  .v-DISCARD{background:#3a1d20;color:#f85149}
  .empty{padding:30px 16px;color:var(--mut);text-align:center}
  code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:12px}
  .play{cursor:pointer;border:1px solid var(--edge);background:var(--panel2);color:var(--accent);
    border-radius:6px;padding:3px 9px;font-size:13px;line-height:1}
  .play:hover{background:var(--accent);color:#0d1117;border-color:var(--accent)}
  /* video modal */
  #modal{position:fixed;inset:0;background:rgba(0,0,0,.78);display:none;
    align-items:center;justify-content:center;z-index:50;padding:24px}
  #modal.open{display:flex}
  #modalBox{background:var(--panel);border:1px solid var(--edge);border-radius:12px;
    max-width:min(900px,94vw);width:auto;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.5)}
  #modalHead{display:flex;justify-content:space-between;align-items:center;gap:16px;
    padding:12px 16px;border-bottom:1px solid var(--edge);background:var(--panel2)}
  #modalHead .mid{font-weight:600}#modalHead .mid code{font-size:13px}
  #modalClose{cursor:pointer;border:none;background:transparent;color:var(--mut);font-size:22px;line-height:1}
  #modalClose:hover{color:var(--txt)}
  #modal video{display:block;max-width:100%;max-height:70vh;background:#000}
  #modalMeta{padding:10px 16px;color:var(--mut);font-size:12px;border-top:1px solid var(--edge)}
  #modalMeta .err{color:#f85149}
  #modalCap{padding:10px 16px;font-size:13px;color:var(--txt);max-width:760px}
</style>
</head>
<body>
<header>
  <h1>__TITLE__ — Metric Distributions</h1>
  <div class="sub">Per-clip quality scores binned into 0.2-wide ranges. Click any bar to list the clips in that range.</div>
  <div class="controls">
    <label id="movieWrap">Movie:
      <select id="movieSel"></select>
    </label>
    <label><input type="radio" name="scope" value="all" checked> All clips (<span id="nAll" class="stat"></span>)</label>
    <label><input type="radio" name="scope" value="kept"> Kept only (<span id="nKept" class="stat"></span>)</label>
    <span class="hint" style="color:var(--mut)">Bins: <code>0–0.2</code> <code>0.2–0.4</code> <code>0.4–0.6</code> <code>0.6–0.8</code> <code>0.8–1.0</code></span>
  </div>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div id="panel">
    <div id="panelHead">
      <div class="t" id="panelTitle">No range selected</div>
      <div class="hint" id="panelHint">Click a bar above to see its clips.</div>
    </div>
    <div class="tablewrap" id="tablewrap"><div class="empty">Nothing selected yet.</div></div>
  </div>
</main>
<div id="modal">
  <div id="modalBox">
    <div id="modalHead">
      <div class="mid">▶ <code id="modalId"></code></div>
      <button id="modalClose" title="Close (Esc)">&times;</button>
    </div>
    <video id="modalVideo" controls autoplay playsinline></video>
    <div id="modalMeta"></div>
    <div id="modalCap"></div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/;
const METRICS = DATA.metrics;
const MOVIES = DATA.movies || [];
const MULTI = MOVIES.length > 1;
const BY_KEY = {}; DATA.clips.forEach(c=>BY_KEY[c.movie+"/"+c.id]=c);
let movieFilter = "__all__";
const BINS = [[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0001]];
const BIN_LABELS = ["0–0.2","0.2–0.4","0.4–0.6","0.6–0.8","0.8–1.0"];
let scope = "all";
let sortKey = null, sortDir = -1;
let current = null; // {metric, bin}

const $ = s => document.querySelector(s);

// movie dropdown (hidden when only one movie)
const movieSel = $('#movieSel');
if(MULTI){
  const opts = [`<option value="__all__">All movies (${MOVIES.length})</option>`]
    .concat(MOVIES.map(m=>`<option value="${m}">${m}</option>`));
  movieSel.innerHTML = opts.join('');
  movieSel.addEventListener('change',()=>{ movieFilter=movieSel.value; refreshCounts(); render(); });
} else {
  $('#movieWrap').style.display='none';
}

function movieClips(){ return movieFilter==="__all__" ? DATA.clips : DATA.clips.filter(c=>c.movie===movieFilter); }
function activeClips(){ const m=movieClips(); return scope==="kept" ? m.filter(c=>c.keep) : m; }
function refreshCounts(){
  const m=movieClips();
  $('#nAll').textContent = m.length;
  $('#nKept').textContent = m.filter(c=>c.keep).length;
}
refreshCounts();
function binOf(v){ for(let i=0;i<BINS.length;i++){ if(v>=BINS[i][0] && v<BINS[i][1]) return i; } return BINS.length-1; }

function render(){
  const clips = activeClips();
  const grid = $('#grid'); grid.innerHTML='';
  METRICS.forEach(metric=>{
    const counts = [0,0,0,0,0];
    let sum=0;
    clips.forEach(c=>{ const v=c.s[metric]; counts[binOf(v)]++; sum+=v; });
    const avg = clips.length ? (sum/clips.length) : 0;
    const max = Math.max(1, ...counts);
    const W=300,H=170, padB=26, padT=16, padL=6, padR=6;
    const bw = (W-padL-padR)/5;
    let bars='';
    for(let i=0;i<5;i++){
      const h = (counts[i]/max)*(H-padB-padT);
      const x = padL + i*bw + 4;
      const y = H-padB-h;
      const w = bw-8;
      const sel = current && current.metric===metric && current.bin===i ? ' sel':'';
      bars += `<rect class="bar${sel}" x="${x}" y="${y}" width="${w}" height="${h}" data-m="${metric}" data-b="${i}"><title>${BIN_LABELS[i]}: ${counts[i]} clips</title></rect>`;
      if(counts[i]>0) bars += `<text class="cnt" x="${x+w/2}" y="${y-4}" text-anchor="middle">${counts[i]}</text>`;
      bars += `<text x="${x+w/2}" y="${H-10}" text-anchor="middle">${BIN_LABELS[i]}</text>`;
    }
    const card=document.createElement('div'); card.className='card';
    card.innerHTML = `<h3>${metric}<span class="avg">avg ${avg.toFixed(3)} · n=${clips.length}</span></h3>
      <svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${bars}</svg>`;
    grid.appendChild(card);
  });
  grid.querySelectorAll('rect.bar').forEach(r=>{
    r.addEventListener('click',()=> selectBin(r.dataset.m, +r.dataset.b));
  });
  if(current) showClips();
}

function selectBin(metric,bin){
  current={metric,bin}; sortKey=metric; sortDir=-1;
  render(); showClips();
  $('#panel').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function showClips(){
  if(!current) return;
  const {metric,bin}=current;
  const lo=BINS[bin][0], hi=BINS[bin][1];
  let rows = activeClips().filter(c=>{ const v=c.s[metric]; return v>=lo && v<hi; });
  $('#panelTitle').innerHTML = `<span>${metric}</span> in range ${BIN_LABELS[bin]}`;
  $('#panelHint').textContent = `${rows.length} clip${rows.length===1?'':'s'} · ${scope==="kept"?"kept only":"all clips"} · click a header to sort`;
  const wrap = $('#tablewrap');
  if(!rows.length){ wrap.innerHTML='<div class="empty">No clips in this range.</div>'; return; }
  const sk = sortKey;
  rows.sort((a,b)=>{
    let av,bv;
    if(sk==='id'){av=a.id;bv=b.id;} else if(sk==='verdict'){av=a.verdict;bv=b.verdict;}
    else if(sk==='movie'){av=a.movie;bv=b.movie;}
    else {av=a.s[sk];bv=b.s[sk];}
    if(av<bv)return -sortDir; if(av>bv)return sortDir; return 0;
  });
  const head = `<tr>
    <th></th>
    ${MULTI?'<th data-k="movie">movie</th>':''}
    <th data-k="id">clip_id</th>
    <th data-k="verdict">verdict</th>
    ${METRICS.map(m=>`<th data-k="${m}" style="${m===metric?'color:var(--accent)':''}">${m}</th>`).join('')}
    <th>timestamp</th><th>caption</th></tr>`;
  const body = rows.map(c=>`<tr>
    <td><button class="play" data-key="${c.movie}/${c.id}" title="Play clip">▶</button></td>
    ${MULTI?`<td style="color:var(--mut)">${c.movie}</td>`:''}
    <td><code class="play-id" data-key="${c.movie}/${c.id}" style="cursor:pointer">${c.id}</code></td>
    <td><span class="pill v-${c.verdict||'NONE'}">${c.verdict||'—'}</span></td>
    ${METRICS.map(m=>`<td class="score" style="${m===metric?'color:var(--accent2)':''}">${c.s[m].toFixed(3)}</td>`).join('')}
    <td>${c.ts[0]!=null?c.ts[0]+'–'+c.ts[1]+'s':'—'}</td>
    <td class="cap">${escapeHtml(c.cap)||'—'}</td></tr>`).join('');
  wrap.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  wrap.querySelectorAll('th[data-k]').forEach(th=>{
    th.addEventListener('click',()=>{ const k=th.dataset.k; if(sortKey===k)sortDir*=-1; else{sortKey=k;sortDir=-1;} showClips(); });
  });
  wrap.querySelectorAll('.play,.play-id').forEach(el=>{
    el.addEventListener('click',()=>openVideo(el.dataset.key));
  });
}

function escapeHtml(s){return (s||'').replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}

const modal=$('#modal'), mVideo=$('#modalVideo');
function openVideo(key){
  const c=BY_KEY[key]; if(!c)return;
  const url=c.video;
  $('#modalId').textContent=(MULTI?c.movie+" / ":"")+c.id;
  $('#modalCap').textContent=c.cap||'';
  $('#modalMeta').innerHTML=`<a href="${url}" target="_blank" style="color:var(--accent)">${url}</a>`
    +` · ${c.verdict||'—'} · ${c.ts[0]!=null?c.ts[0]+'–'+c.ts[1]+'s':''}`
    +` · `+METRICS.map(m=>`${m} ${c.s[m].toFixed(2)}`).join(' · ');
  mVideo.src=url;
  mVideo.onerror=()=>{ $('#modalMeta').innerHTML=`<span class="err">Could not load <code>${url}</code> — is the server rooted so this path is reachable?</span>`; };
  modal.classList.add('open');
  mVideo.play().catch(()=>{});
}
function closeVideo(){ modal.classList.remove('open'); mVideo.pause(); mVideo.removeAttribute('src'); mVideo.load(); }
$('#modalClose').addEventListener('click',closeVideo);
modal.addEventListener('click',e=>{ if(e.target===modal) closeVideo(); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeVideo(); });

document.querySelectorAll('input[name=scope]').forEach(r=>{
  r.addEventListener('change',e=>{ scope=e.target.value; render(); });
});
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "source",
        help="metadata.jsonl, a single movie dir, OR a parent dir of many movie dirs",
    )
    ap.add_argument("-o", "--out", help="output HTML path")
    ap.add_argument("--title", help="dashboard title")
    ap.add_argument(
        "--clip-base",
        default="clips",
        help="single-movie only: path prefix where <clip_id>.mp4 lives, relative "
        "to the HTML (default: clips — correct when serving from the movie dir)",
    )
    args = ap.parse_args()

    src = Path(args.source).expanduser()

    # Decide single vs. multi-movie.
    single_md = None
    if src.is_file():
        single_md = src
    elif src.is_dir() and (src / "metadata.jsonl").exists():
        single_md = src / "metadata.jsonl"

    if single_md is not None:
        recs = load_records(single_md)
        if not recs:
            raise SystemExit(f"No records in {single_md}")
        movie = single_md.parent.name
        cb = args.clip_base.rstrip("/")
        clips = [to_clip(r, movie, f"{cb}/{r.get('clip_id','')}.mp4") for r in recs]
        title = args.title or recs[0].get("video_id") or movie
        movies = [movie]
        out = Path(args.out).expanduser() if args.out else single_md.parent / "reports" / "metric_distributions.html"
    else:
        if not src.is_dir():
            raise SystemExit(f"Not a file or directory: {src}")
        found = discover_movies(src)
        if not found:
            raise SystemExit(f"No */metadata.jsonl found under {src}")
        clips = []
        movies = []
        for movie, md in found:
            recs = load_records(md)
            movies.append(movie)
            # served from the parent root → clips live at <movie>/clips/<id>.mp4
            clips.extend(
                to_clip(r, movie, f"{movie}/clips/{r.get('clip_id','')}.mp4") for r in recs
            )
        title = args.title or f"{src.name} — {len(movies)} movies"
        out = Path(args.out).expanduser() if args.out else src / "index.html"

    html = build_html(clips, title, movies)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(clips)} clips, {len(movies)} movie(s), {len(html)//1024} KB)")


if __name__ == "__main__":
    main()
