#!/usr/bin/env python3
"""Generate an HTML dashboard from pipeline results."""

import base64
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
import io


def img_to_base64_thumb(img_path: str, max_size: int = 200) -> str:
    """Convert image to base64 thumbnail for inline HTML."""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def generate_dashboard(work_dir: str, output_path: str | None = None):
    work = Path(work_dir)
    if output_path is None:
        output_path = str(work / "dashboard.html")

    # ── Load all data ─────────────────────────────────────────────────────
    manifest = {}
    mp = work / "manifest.json"
    if mp.exists():
        with open(mp) as f:
            manifest = json.load(f)

    vlm_results = []
    vlm_dir = work / "vlm_results"
    if vlm_dir.exists():
        for jp in sorted(vlm_dir.glob("*.json")):
            with open(jp) as f:
                vlm_results.append(json.load(f))

    captions = []
    cap_dir = work / "captions"
    if cap_dir.exists():
        for cp in sorted(cap_dir.glob("*_caption.json")):
            with open(cp) as f:
                captions.append(json.load(f))

    scores_data = []
    sp = work / "scores.csv"
    if sp.exists():
        with open(sp) as f:
            scores_data = list(csv.DictReader(f))

    gated_data = []
    gp = work / "gated.csv"
    if gp.exists():
        with open(gp) as f:
            gated_data = list(csv.DictReader(f))

    report = {}
    rp = work / "report.json"
    if rp.exists():
        with open(rp) as f:
            report = json.load(f)

    log_text = ""
    lp = work / "run.log"
    if not lp.exists():
        lp = work / "pipeline.log"
    if lp.exists():
        log_text = lp.read_text()

    # ── Compute stats ─────────────────────────────────────────────────────
    n_videos = len(manifest.get("videos", []))
    n_images = len(manifest.get("images", []))
    n_total_input = n_videos + n_images

    cats = Counter()
    rejected = 0
    accepted = 0
    source_types = Counter()
    for r in vlm_results:
        source_types[r.get("source_type", "unknown")] += 1
        if r.get("rejected"):
            rejected += 1
        else:
            accepted += 1
            cats[r.get("category", "unknown")] += 1

    n_captioned = len(captions)
    n_scored = len(scores_data)

    gate_counts = Counter()
    for r in gated_data:
        gate_counts[r.get("gate", "")] += 1

    # Score stats
    ss = report.get("score_stats", {})
    ss_combined = ss.get("combined_score", {}).get("mean", 0)
    ss_clip = ss.get("clip_score", {}).get("mean", 0)
    ss_icr = ss.get("icr_score", {}).get("mean", 0)
    ss_aod = ss.get("aod_score", {}).get("mean", 0)

    # Extract timing from log
    step_times = {}
    for line in log_text.split("\n"):
        m = re.search(r"Step '(\w+)' completed in ([\d.]+)s", line)
        if m:
            step_times[m.group(1)] = float(m.group(2))
    total_time = sum(step_times.values())

    # Per-bucket for chart
    bucket_labels = json.dumps(list(cats.keys()))
    bucket_values = json.dumps(list(cats.values()))

    # Gate for chart
    gate_labels = json.dumps(["Final", "Review", "Discard"])
    gate_values = json.dumps([
        gate_counts.get("final", 0),
        gate_counts.get("review", 0),
        gate_counts.get("discard", 0),
    ])

    # Source type for chart
    st_labels = json.dumps(list(source_types.keys()))
    st_values = json.dumps(list(source_types.values()))

    # Step timing for chart
    time_labels = json.dumps(list(step_times.keys()))
    time_values = json.dumps(list(step_times.values()))

    # Sample images (up to 12 from gated with scores)
    sample_cards = []
    shown = 0
    for row in sorted(gated_data, key=lambda r: -float(r.get("combined_score", 0)))[:20]:
        img_path = row.get("image_path", "")
        if not Path(img_path).exists():
            continue
        b64 = img_to_base64_thumb(img_path, 250)
        if not b64:
            continue

        # Find caption
        cap_text = row.get("caption", "")[:200]
        if not cap_text:
            for c in captions:
                if c.get("image", "") == img_path:
                    cap_text = c.get("caption", "")[:200]
                    break

        sample_cards.append({
            "b64": b64,
            "name": Path(img_path).name,
            "bucket": row.get("bucket", "?"),
            "gate": row.get("gate", "?"),
            "combined": float(row.get("combined_score", 0)),
            "clip": float(row.get("clip_score", 0)),
            "icr": float(row.get("icr_score", 0)),
            "aod": float(row.get("aod_score", 0)),
            "source_type": row.get("source_type", "?"),
            "caption": cap_text,
        })
        shown += 1
        if shown >= 12:
            break

    cards_html = ""
    for card in sample_cards:
        gate_color = {"final": "#22c55e", "review": "#eab308", "discard": "#ef4444"}.get(card["gate"], "#888")
        cards_html += f"""
        <div class="card">
            <img src="data:image/jpeg;base64,{card['b64']}" alt="{card['name']}">
            <div class="card-body">
                <div class="card-title">{card['name']}</div>
                <span class="badge" style="background:{gate_color}">{card['gate'].upper()}</span>
                <span class="badge" style="background:#6366f1">{card['bucket']}</span>
                <span class="badge" style="background:#0891b2">{card['source_type']}</span>
                <div class="scores">
                    <span>Combined: <b>{card['combined']:.3f}</b></span>
                    <span>CLIP: {card['clip']:.3f}</span>
                    <span>ICR: {card['icr']:.3f}</span>
                    <span>AOD: {card['aod']:.3f}</span>
                </div>
                <div class="caption">{card['caption']}...</div>
            </div>
        </div>"""

    # Pipeline steps status
    all_steps = ["discover", "extract", "dedup_intra", "classify", "dedup_cross", "tag_actors", "caption", "score", "gate", "export", "report"]
    steps_html = ""
    for i, s in enumerate(all_steps, 1):
        t = step_times.get(s, 0)
        done = t > 0
        icon = "&#10003;" if done else "&#10007;"
        color = "#22c55e" if done else "#ef4444"
        steps_html += f"""
        <div class="step {'step-done' if done else 'step-pending'}">
            <span class="step-icon" style="color:{color}">{icon}</span>
            <span class="step-name">Step {i}: {s}</span>
            <span class="step-time">{t:.1f}s</span>
        </div>"""

    # ── Build HTML ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Master Pipeline Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
  h1 {{ text-align: center; font-size: 2rem; margin-bottom: 8px; background: linear-gradient(135deg, #a3e635, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .stat-card h3 {{ color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .stat-card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .stat-card .sub {{ color: #64748b; font-size: 0.85rem; margin-top: 4px; }}
  .section {{ background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin-bottom: 24px; }}
  .section h2 {{ font-size: 1.3rem; margin-bottom: 16px; color: #f1f5f9; }}
  .chart-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 24px; }}
  .chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .chart-box h3 {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px; text-align: center; }}
  canvas {{ max-height: 280px; }}
  .steps-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }}
  .step {{ display: flex; align-items: center; gap: 8px; padding: 10px 14px; background: #0f172a; border-radius: 8px; }}
  .step-icon {{ font-size: 1.2rem; font-weight: bold; }}
  .step-name {{ flex: 1; font-size: 0.85rem; }}
  .step-time {{ color: #64748b; font-size: 0.8rem; }}
  .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
  .card {{ background: #0f172a; border-radius: 10px; overflow: hidden; border: 1px solid #334155; }}
  .card img {{ width: 100%; height: 200px; object-fit: cover; }}
  .card-body {{ padding: 12px; }}
  .card-title {{ font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; color: #fff; margin-right: 4px; }}
  .scores {{ margin-top: 8px; font-size: 0.78rem; color: #94a3b8; display: flex; flex-wrap: wrap; gap: 8px; }}
  .caption {{ margin-top: 8px; font-size: 0.78rem; color: #64748b; line-height: 1.4; max-height: 60px; overflow: hidden; }}
  .green {{ color: #22c55e; }} .yellow {{ color: #eab308; }} .red {{ color: #ef4444; }} .blue {{ color: #38bdf8; }}
  .total-time {{ text-align: center; color: #94a3b8; margin-top: 16px; font-size: 0.9rem; }}

  /* ── Wavy glow overlay ────────────────────────────────────── */
  .wave-overlay {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
    mix-blend-mode: screen;
  }}

  /* Huge glowing blobs — bright colored aurora that drifts */
  .wave-overlay .wave {{
    position: absolute;
    border-radius: 45%;
    filter: blur(100px);
  }}
  .wave-overlay .wave:nth-child(1) {{
    width: 70vw; height: 70vh;
    top: -20%; left: -15%;
    background: radial-gradient(ellipse, rgba(6,182,212,0.6) 0%, rgba(6,182,212,0.3) 35%, transparent 70%);
    animation: blobDrift1 7s ease-in-out infinite;
  }}
  .wave-overlay .wave:nth-child(2) {{
    width: 80vw; height: 80vh;
    top: 5%; right: -20%;
    background: radial-gradient(ellipse, rgba(139,92,246,0.55) 0%, rgba(139,92,246,0.25) 35%, transparent 70%);
    animation: blobDrift2 9s ease-in-out 1s infinite;
  }}
  .wave-overlay .wave:nth-child(3) {{
    width: 65vw; height: 65vh;
    bottom: -15%; left: 15%;
    background: radial-gradient(ellipse, rgba(6,182,212,0.5) 0%, rgba(34,197,94,0.25) 35%, transparent 70%);
    animation: blobDrift3 11s ease-in-out 2s infinite;
  }}
  .wave-overlay .wave:nth-child(4) {{
    width: 75vw; height: 75vh;
    top: 25%; left: 25%;
    background: radial-gradient(ellipse, rgba(168,85,247,0.45) 0%, rgba(59,130,246,0.2) 40%, transparent 70%);
    animation: blobDrift4 13s ease-in-out 0.5s infinite;
  }}

  /* Thick glowing streaks — bright horizontal bands that sweep */
  .wave-overlay .glow-streak {{
    position: absolute;
    width: 200%;
    left: -50%;
    filter: blur(45px);
    border-radius: 50%;
  }}
  .wave-overlay .glow-streak:nth-child(5) {{
    top: 18%; height: 180px;
    background: linear-gradient(90deg, transparent 0%, rgba(6,182,212,0.7) 20%, rgba(6,182,212,0.9) 45%, rgba(139,92,246,0.7) 75%, transparent 100%);
    animation: streakSweep1 6s ease-in-out infinite;
  }}
  .wave-overlay .glow-streak:nth-child(6) {{
    top: 50%; height: 200px;
    background: linear-gradient(90deg, transparent 0%, rgba(168,85,247,0.65) 20%, rgba(236,72,153,0.8) 50%, rgba(239,68,68,0.6) 80%, transparent 100%);
    animation: streakSweep2 8s ease-in-out 2s infinite;
  }}
  .wave-overlay .glow-streak:nth-child(7) {{
    top: 80%; height: 150px;
    background: linear-gradient(90deg, transparent 0%, rgba(34,197,94,0.6) 25%, rgba(163,230,53,0.8) 50%, rgba(6,182,212,0.6) 75%, transparent 100%);
    animation: streakSweep3 10s ease-in-out 3.5s infinite;
  }}

  @keyframes blobDrift1 {{
    0%   {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.8; }}
    25%  {{ transform: translate(18vw, 10vh) scale(1.15) rotate(3deg); opacity: 1; }}
    50%  {{ transform: translate(30vw, 3vh) scale(0.95) rotate(-2deg); opacity: 0.7; }}
    75%  {{ transform: translate(12vw, -8vh) scale(1.2) rotate(1deg); opacity: 0.95; }}
    100% {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.8; }}
  }}
  @keyframes blobDrift2 {{
    0%   {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.75; }}
    30%  {{ transform: translate(-22vw, 15vh) scale(1.2) rotate(-3deg); opacity: 1; }}
    60%  {{ transform: translate(-10vw, 25vh) scale(1.05) rotate(2deg); opacity: 0.65; }}
    100% {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.75; }}
  }}
  @keyframes blobDrift3 {{
    0%   {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.7; }}
    35%  {{ transform: translate(20vw, -18vh) scale(1.25) rotate(4deg); opacity: 0.95; }}
    65%  {{ transform: translate(10vw, -28vh) scale(1.0) rotate(-2deg); opacity: 0.6; }}
    100% {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.7; }}
  }}
  @keyframes blobDrift4 {{
    0%   {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.65; }}
    20%  {{ transform: translate(-15vw, -10vh) scale(1.1) rotate(-2deg); opacity: 0.9; }}
    50%  {{ transform: translate(-28vw, 6vh) scale(1.3) rotate(3deg); opacity: 0.7; }}
    80%  {{ transform: translate(-6vw, 12vh) scale(1.05) rotate(-1deg); opacity: 1; }}
    100% {{ transform: translate(0, 0) scale(1) rotate(0deg); opacity: 0.65; }}
  }}

  @keyframes streakSweep1 {{
    0%   {{ transform: translateX(-70%) scaleY(1); opacity: 0; }}
    10%  {{ opacity: 0.9; }}
    50%  {{ transform: translateX(25%) scaleY(1.6); opacity: 0.7; }}
    90%  {{ opacity: 0.9; }}
    100% {{ transform: translateX(-70%) scaleY(1); opacity: 0; }}
  }}
  @keyframes streakSweep2 {{
    0%   {{ transform: translateX(50%) scaleY(1); opacity: 0; }}
    15%  {{ opacity: 0.85; }}
    50%  {{ transform: translateX(-35%) scaleY(1.8); opacity: 0.65; }}
    85%  {{ opacity: 0.85; }}
    100% {{ transform: translateX(50%) scaleY(1); opacity: 0; }}
  }}
  @keyframes streakSweep3 {{
    0%   {{ transform: translateX(-60%) scaleY(1); opacity: 0; }}
    20%  {{ opacity: 0.8; }}
    50%  {{ transform: translateX(18%) scaleY(1.5); opacity: 0.6; }}
    80%  {{ opacity: 0.8; }}
    100% {{ transform: translateX(-60%) scaleY(1); opacity: 0; }}
  }}
</style>
</head>
<body>

<!-- Wavy dark glow overlay -->
<div class="wave-overlay">
  <div class="wave"></div>
  <div class="wave"></div>
  <div class="wave"></div>
  <div class="wave"></div>
  <div class="glow-streak"></div>
  <div class="glow-streak"></div>
  <div class="glow-streak"></div>
</div>

<h1>Master Pipeline Dashboard</h1>
<p class="subtitle">Indic Cultural Image Dataset &mdash; End-to-End Run Results</p>

<!-- KPI Cards -->
<div class="grid">
  <div class="stat-card">
    <h3>Input Sources</h3>
    <div class="value">{n_videos + n_images}</div>
    <div class="sub">{n_videos} videos + {n_images} precaptioned images</div>
  </div>
  <div class="stat-card">
    <h3>Classified</h3>
    <div class="value"><span class="green">{accepted}</span> <span style="font-size:1rem;color:#64748b">/ {accepted+rejected}</span></div>
    <div class="sub">{rejected} rejected ({rejected*100//(accepted+rejected) if accepted+rejected else 0}%)</div>
  </div>
  <div class="stat-card">
    <h3>Captioned</h3>
    <div class="value blue">{n_captioned}</div>
    <div class="sub">Fresh VLM captions (bucket-specific prompts)</div>
  </div>
  <div class="stat-card">
    <h3>Final Export</h3>
    <div class="value green">{gate_counts.get('final', 0)}</div>
    <div class="sub">{gate_counts.get('review', 0)} review, {gate_counts.get('discard', 0)} discard</div>
  </div>
  <div class="stat-card">
    <h3>Mean Combined Score</h3>
    <div class="value yellow">{ss_combined:.4f}</div>
    <div class="sub">CLIP={ss_clip:.3f} &middot; ICR={ss_icr:.3f} &middot; AOD={ss_aod:.3f}</div>
  </div>
  <div class="stat-card">
    <h3>Total Time</h3>
    <div class="value">{total_time:.0f}s</div>
    <div class="sub">{total_time/60:.1f} minutes</div>
  </div>
</div>

<!-- Charts -->
<div class="chart-row">
  <div class="chart-box">
    <h3>Bucket Distribution</h3>
    <canvas id="bucketChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Gate Results</h3>
    <canvas id="gateChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Source Types</h3>
    <canvas id="sourceChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Step Timing (seconds)</h3>
    <canvas id="timeChart"></canvas>
  </div>
</div>

<!-- Pipeline Steps -->
<div class="section">
  <h2>Pipeline Steps</h2>
  <div class="steps-grid">
    {steps_html}
  </div>
  <div class="total-time">Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)</div>
</div>

<!-- Sample Images -->
<div class="section">
  <h2>Top Scored Images (by combined score)</h2>
  <div class="cards-grid">
    {cards_html}
  </div>
</div>

<script>
const chartColors = ['#22c55e','#06b6d4','#a855f7','#f59e0b','#ef4444','#3b82f6','#ec4899','#14b8a6','#f97316','#8b5cf6','#64748b','#84cc16'];
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';

new Chart(document.getElementById('bucketChart'), {{
  type: 'doughnut',
  data: {{ labels: {bucket_labels}, datasets: [{{ data: {bucket_values}, backgroundColor: chartColors }}] }},
  options: {{ plugins: {{ legend: {{ position: 'right', labels: {{ font: {{ size: 11 }} }} }} }} }}
}});

new Chart(document.getElementById('gateChart'), {{
  type: 'doughnut',
  data: {{ labels: {gate_labels}, datasets: [{{ data: {gate_values}, backgroundColor: ['#22c55e','#eab308','#ef4444'] }}] }},
  options: {{ plugins: {{ legend: {{ position: 'right' }} }} }}
}});

new Chart(document.getElementById('sourceChart'), {{
  type: 'pie',
  data: {{ labels: {st_labels}, datasets: [{{ data: {st_values}, backgroundColor: ['#3b82f6','#06b6d4','#a855f7','#f59e0b'] }}] }},
  options: {{ plugins: {{ legend: {{ position: 'right' }} }} }}
}});

new Chart(document.getElementById('timeChart'), {{
  type: 'bar',
  data: {{ labels: {time_labels}, datasets: [{{ label: 'seconds', data: {time_values}, backgroundColor: '#06b6d4' }}] }},
  options: {{ indexAxis: 'y', plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ grid: {{ color: '#1e293b' }} }}, y: {{ grid: {{ display: false }} }} }} }}
}});
</script>

</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Dashboard -> {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dashboard.py <work_dir> [output.html]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    generate_dashboard(sys.argv[1], out)
