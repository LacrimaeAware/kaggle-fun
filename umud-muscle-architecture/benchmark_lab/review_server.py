"""Review human benchmark labels against submission/candidate CSVs.

This is a local-only viewer. It does not modify submissions and does not call Kaggle.

Example:
    python benchmark_lab/review_server.py --port 8767
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
LAYERS = ("apo", "fasc", "ignore")


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>UMUD Label Review</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: #151515;
      color: #eeeeee;
      overflow: hidden;
    }
    #topbar {
      height: 54px;
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px 12px;
      background: #202020;
      border-bottom: 1px solid #333;
      white-space: nowrap;
    }
    button, select, textarea, input {
      font: inherit;
      border: 1px solid #444;
      background: #2b2b2b;
      color: #f2f2f2;
      border-radius: 6px;
      padding: 6px 8px;
    }
    button { cursor: pointer; }
    button.active { background: #123a57; outline: 2px solid #7ab7ff; }
    #wrap {
      height: calc(100vh - 54px);
      display: grid;
      grid-template-columns: 310px 1fr 390px;
      min-width: 0;
    }
    #list, #side {
      overflow: auto;
      background: #1e1e1e;
      border-right: 1px solid #333;
      padding: 10px;
    }
    #side {
      border-right: 0;
      border-left: 1px solid #333;
    }
    #stage {
      position: relative;
      overflow: auto;
      background: #050505;
    }
    #stack {
      position: relative;
      width: max-content;
      height: max-content;
      margin: 18px;
    }
    #stack img {
      position: absolute;
      inset: 0;
      transform-origin: top left;
      image-rendering: auto;
    }
    #base { position: relative !important; z-index: 0; }
    #apo { z-index: 2; opacity: 0.75; }
    #fasc { z-index: 3; opacity: 0.75; }
    #ignore { z-index: 4; opacity: 0.45; }
    .item {
      border-bottom: 1px solid #333;
      padding: 7px 6px;
      cursor: pointer;
      font-size: 12px;
      line-height: 1.35;
    }
    .item.current { background: #123a57; }
    .item .score { color: #ffcf7a; }
    .item.reviewed::before { content: "reviewed "; color: #9fe59f; }
    .kv { font-size: 12px; color: #bbbbbb; line-height: 1.45; word-break: break-word; }
    .kv b { color: #f5f5f5; }
    .section { border-top: 1px solid #333; margin-top: 12px; padding-top: 12px; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
    .pill { border: 1px solid #444; border-radius: 999px; padding: 2px 7px; color: #ddd; font-size: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #333; padding: 5px 4px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; }
    td.bad { color: #ff9388; }
    td.ok { color: #e7e7e7; }
    td.good { color: #9fe59f; }
    textarea { width: 100%; min-height: 90px; resize: vertical; }
    #status { color: #9fe59f; }
    .summaryTable td, .summaryTable th { font-size: 11px; }
    input[type="range"] { width: 130px; }
  </style>
</head>
<body>
  <div id="topbar">
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <span id="counter"></span>
    <button id="zoomOut">-</button>
    <button id="zoomReset">100%</button>
    <button id="zoomIn">+</button>
    <button id="saveReview">Save Review</button>
    <span id="status"></span>
  </div>
  <div id="wrap">
    <nav id="list"></nav>
    <main id="stage">
      <div id="stack">
        <img id="base">
        <img id="apo">
        <img id="fasc">
        <img id="ignore">
      </div>
    </main>
    <aside id="side">
      <div id="meta" class="kv"></div>
      <div class="section">
        <div class="row">
          <button class="layer active" data-layer="apo">apo</button>
          <button class="layer active" data-layer="fasc">fasc</button>
          <button class="layer" data-layer="ignore">ignore</button>
        </div>
        <div class="row">
          <label>overlay <input id="opacity" type="range" min="0" max="100" value="75"></label>
          <span id="opacityVal">75%</span>
        </div>
      </div>
      <div class="section">
        <div class="kv"><b>Candidate summary on labeled rows</b></div>
        <div id="summary"></div>
      </div>
      <div class="section">
        <div class="kv"><b>Current image benchmark</b></div>
        <div id="candidateTable"></div>
      </div>
      <div class="section">
        <div class="row">
          <label>label quality
            <select id="labelQuality">
              <option value="">unreviewed</option>
              <option value="good">good</option>
              <option value="usable_rough">usable rough</option>
              <option value="lazy_or_bad">lazy/bad</option>
              <option value="ambiguous">ambiguous</option>
              <option value="redo">redo</option>
            </select>
          </label>
        </div>
        <div class="row">
          <label>what failed
            <select id="failureKind">
              <option value="">not marked</option>
              <option value="human_label_issue">human label issue</option>
              <option value="model_mask_issue">model mask issue</option>
              <option value="scale_issue">scale issue</option>
              <option value="geometry_issue">geometry issue</option>
              <option value="unclear_structure">unclear structure</option>
            </select>
          </label>
        </div>
        <label class="kv">notes</label>
        <textarea id="notes" placeholder="Say what looks wrong: label was lazy, boundary wrong, fragment unclear, etc."></textarea>
      </div>
      <div class="section kv">
        Shortcuts: arrows navigate, S saves review note, A/F/I toggle layers, Ctrl/Cmd +/-/0 zoom.
      </div>
    </aside>
  </div>
<script>
const state = { rows: [], summary: [], idx: 0, zoom: 1, visible: {apo: true, fasc: true, ignore: false} };
const imgs = {base: document.getElementById('base'), apo: document.getElementById('apo'), fasc: document.getElementById('fasc'), ignore: document.getElementById('ignore')};
function qs(id){ return document.getElementById(id); }
function row(){ return state.rows[state.idx]; }
function fmt(v, digits=2){ if(v===null || v===undefined || v==='') return ''; const n=Number(v); return Number.isFinite(n) ? n.toFixed(digits) : ''; }
function clsNorm(n){ if(n === null || n === undefined || n === '') return 'ok'; n=Number(n); if(n >= 1) return 'bad'; if(n <= 0.35) return 'good'; return 'ok'; }
function setStatus(txt){ qs('status').textContent = txt; setTimeout(()=>{ if(qs('status').textContent===txt) qs('status').textContent=''; }, 2500); }

function applyZoom(){
  const w = imgs.base.naturalWidth || 1;
  const h = imgs.base.naturalHeight || 1;
  const sw = Math.round(w * state.zoom);
  const sh = Math.round(h * state.zoom);
  for (const im of Object.values(imgs)) {
    im.style.width = sw + 'px';
    im.style.height = sh + 'px';
  }
  qs('stack').style.width = sw + 'px';
  qs('stack').style.height = sh + 'px';
  qs('zoomReset').textContent = Math.round(state.zoom * 100) + '%';
}
function setZoom(z){ state.zoom = Math.min(4, Math.max(0.35, z)); applyZoom(); }

function updateLayerVisibility(){
  for (const layer of ['apo','fasc','ignore']) {
    imgs[layer].style.display = state.visible[layer] ? 'block' : 'none';
    document.querySelector(`button[data-layer="${layer}"]`).classList.toggle('active', state.visible[layer]);
  }
  const op = Number(qs('opacity').value) / 100;
  qs('opacityVal').textContent = qs('opacity').value + '%';
  imgs.apo.style.opacity = op;
  imgs.fasc.style.opacity = op;
  imgs.ignore.style.opacity = Math.min(op, 0.55);
}

function renderList(){
  const box = qs('list');
  box.innerHTML = '';
  state.rows.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'item' + (i === state.idx ? ' current' : '') + (r.review && r.review.updated_at ? ' reviewed' : '');
    d.innerHTML = `<b>${i+1}. ${r.image_id}</b><br><span class="score">score ${fmt(r.sort_score,2)}</span> ` +
      `<span class="kv">fasc ${r.n_fascicles || ''}</span><br>` +
      `<span class="kv">${r.review?.label_quality || ''} ${r.review?.failure_kind || ''}</span>`;
    d.onclick = () => { state.idx = i; loadCurrent(); };
    box.appendChild(d);
  });
}

function renderSummary(){
  let html = '<table class="summaryTable"><tr><th>candidate</th><th>overall</th><th>PA</th><th>FL</th><th>MT</th><th>n</th></tr>';
  for (const s of state.summary) {
    html += `<tr><td>${s.name}</td><td class="${clsNorm(s.overall_norm)}">${fmt(s.overall_norm,2)}</td>` +
      `<td>${fmt(s.pa_norm,2)}</td><td>${fmt(s.fl_norm,2)}</td><td>${fmt(s.mt_norm,2)}</td><td>${s.n}</td></tr>`;
  }
  html += '</table>';
  qs('summary').innerHTML = html;
}

function renderCandidateTable(){
  const r = row();
  let html = '<table><tr><th>source</th><th>PA</th><th>FL</th><th>MT</th><th>norm</th></tr>';
  html += `<tr><td>human mask</td><td>${fmt(r.human.pa_deg)}</td><td>${fmt(r.human.fl_mm)}</td><td>${fmt(r.human.mt_mm)}</td><td></td></tr>`;
  for (const c of r.candidates) {
    html += `<tr><td>${c.name}</td><td>${fmt(c.pa_deg)}</td><td>${fmt(c.fl_mm)}</td><td>${fmt(c.mt_mm)}</td><td class="${clsNorm(c.overall_norm)}">${fmt(c.overall_norm,2)}</td></tr>`;
    html += `<tr><td class="kv">delta</td><td class="${clsNorm(Math.abs(c.delta_pa)/6)}">${fmt(c.delta_pa)}</td>` +
      `<td class="${clsNorm(Math.abs(c.delta_fl)/12)}">${fmt(c.delta_fl)}</td>` +
      `<td class="${clsNorm(Math.abs(c.delta_mt)/3)}">${fmt(c.delta_mt)}</td><td></td></tr>`;
  }
  html += '</table>';
  qs('candidateTable').innerHTML = html;
}

function loadCurrent(){
  const r = row();
  qs('counter').textContent = `${state.idx + 1}/${state.rows.length}`;
  qs('meta').innerHTML = `<b>${r.label_id}</b><br>image: ${r.image_id}<br>quality: ${r.quality || ''}<br>` +
    `pixels: apo ${r.apo_pixels}, fasc ${r.fasc_pixels}<br>measured fragments: ${r.n_fascicles || ''}<br>` +
    `sort disagreement: ${fmt(r.sort_score,2)}`;
  imgs.base.onload = applyZoom;
  imgs.base.src = `/image/${state.idx}?t=${Date.now()}`;
  for (const layer of ['apo','fasc','ignore']) imgs[layer].src = `/label/${state.idx}/${layer}.png?t=${Date.now()}`;
  qs('labelQuality').value = r.review?.label_quality || '';
  qs('failureKind').value = r.review?.failure_kind || '';
  qs('notes').value = r.review?.notes || '';
  renderList();
  renderCandidateTable();
  updateLayerVisibility();
}

async function saveReview(){
  const payload = {
    index: state.idx,
    label_quality: qs('labelQuality').value,
    failure_kind: qs('failureKind').value,
    notes: qs('notes').value,
  };
  const res = await fetch('/api/save_review', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if(!res.ok){ setStatus(await res.text()); return; }
  const saved = await res.json();
  row().review = saved.review;
  setStatus('review saved');
  renderList();
}

document.querySelectorAll('button.layer').forEach(b => b.onclick = () => { state.visible[b.dataset.layer] = !state.visible[b.dataset.layer]; updateLayerVisibility(); });
qs('opacity').oninput = updateLayerVisibility;
qs('saveReview').onclick = saveReview;
qs('prev').onclick = () => { if(state.idx > 0){ state.idx--; loadCurrent(); } };
qs('next').onclick = () => { if(state.idx + 1 < state.rows.length){ state.idx++; loadCurrent(); } };
qs('zoomOut').onclick = () => setZoom(state.zoom / 1.25);
qs('zoomIn').onclick = () => setZoom(state.zoom * 1.25);
qs('zoomReset').onclick = () => setZoom(1);
window.addEventListener('keydown', ev => {
  if (ev.target && ['TEXTAREA','INPUT','SELECT'].includes(ev.target.tagName)) return;
  if ((ev.ctrlKey || ev.metaKey) && ['+','='].includes(ev.key)) { ev.preventDefault(); setZoom(state.zoom * 1.25); return; }
  if ((ev.ctrlKey || ev.metaKey) && ev.key === '-') { ev.preventDefault(); setZoom(state.zoom / 1.25); return; }
  if ((ev.ctrlKey || ev.metaKey) && ev.key === '0') { ev.preventDefault(); setZoom(1); return; }
  if (ev.key === 'ArrowLeft') qs('prev').click();
  if (ev.key === 'ArrowRight') qs('next').click();
  if (ev.key === 's' || ev.key === 'S') saveReview();
  if (ev.key === 'a' || ev.key === 'A') { state.visible.apo = !state.visible.apo; updateLayerVisibility(); }
  if (ev.key === 'f' || ev.key === 'F') { state.visible.fasc = !state.visible.fasc; updateLayerVisibility(); }
  if (ev.key === 'i' || ev.key === 'I') { state.visible.ignore = !state.visible.ignore; updateLayerVisibility(); }
});

async function init(){
  const res = await fetch('/api/review');
  const data = await res.json();
  state.rows = data.rows;
  state.summary = data.summary;
  renderSummary();
  if(!state.rows.length){ qs('meta').textContent = 'No labeled rows found.'; return; }
  loadCurrent();
}
init();
</script>
</body>
</html>
"""


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def stem(value: str) -> str:
    return Path(str(value)).stem


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_keyed(path: Path) -> dict[str, dict[str, str]]:
    return {stem(r.get("image_id", "")): r for r in read_csv(path)}


def parse_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def browser_image_bytes(path: Path) -> tuple[bytes, str]:
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff", ".bmp"}:
        return path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    if cv2 is None or np is None:
        return path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    if arr.dtype != np.uint8:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = ((arr.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    return bytes(buf), "image/png"


def default_candidate_csvs() -> list[tuple[str, Path]]:
    candidates = [
        ("baseline_0619", ROOT / "results" / "submission_local.csv"),
        ("mt_vertical3_0625", ROOT / "results" / "submission_host_mt_vertical3_no_subpixel.csv"),
        ("scale_tail_bar_0667", ROOT / "results" / "submission_scale_tail_bar_only.csv"),
        ("segmentation_old", ROOT / "results" / "submission_segmentation.csv"),
    ]
    return [(name, path) for name, path in candidates if path.exists()]


def parse_candidate_arg(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(value).stem
        out.append((safe_id(name), Path(path)))
    return out


def norm_delta(human: dict[str, float | None], pred: dict[str, float | None]) -> tuple[dict[str, float | None], float | None]:
    deltas = {
        "delta_pa": None if human.get("pa_deg") is None or pred.get("pa_deg") is None else pred["pa_deg"] - human["pa_deg"],
        "delta_fl": None if human.get("fl_mm") is None or pred.get("fl_mm") is None else pred["fl_mm"] - human["fl_mm"],
        "delta_mt": None if human.get("mt_mm") is None or pred.get("mt_mm") is None else pred["mt_mm"] - human["mt_mm"],
    }
    vals = []
    if deltas["delta_pa"] is not None:
        vals.append(abs(deltas["delta_pa"]) / TOL["pa_deg"])
    if deltas["delta_fl"] is not None:
        vals.append(abs(deltas["delta_fl"]) / TOL["fl_mm"])
    if deltas["delta_mt"] is not None:
        vals.append(abs(deltas["delta_mt"]) / TOL["mt_mm"])
    return deltas, (sum(vals) / len(vals) if vals else None)


def load_review(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_rows(
    manifest_path: Path,
    labels_dir: Path,
    scores_path: Path,
    calibration_path: Path,
    candidate_csvs: list[tuple[str, Path]],
    review_dir: Path,
) -> tuple[list[dict], list[dict]]:
    manifest = read_csv(manifest_path)
    scores = read_keyed(scores_path)
    cal = read_keyed(calibration_path)
    candidate_data = [(name, read_keyed(path)) for name, path in candidate_csvs]
    rows = []
    summary_acc: dict[str, dict[str, list[float]]] = {
        name: {"pa": [], "fl": [], "mt": [], "overall": []} for name, _ in candidate_data
    }

    for src in manifest:
        label_id = safe_id(src["label_id"])
        image_id = stem(src.get("image_id", label_id))
        score = scores.get(image_id, {})
        if not (score.get("has_apo") == "true" and score.get("has_fasc") == "true"):
            continue
        scale = parse_float(cal.get(image_id, {}).get("px_per_mm"))
        human = {
            "pa_deg": parse_float(score.get("pa_deg_measured")),
            "fl_mm": None,
            "mt_mm": None,
        }
        fl_px = parse_float(score.get("fl_px_measured"))
        mt_px = parse_float(score.get("mt_px_measured"))
        if scale:
            human["fl_mm"] = None if fl_px is None else fl_px / scale
            human["mt_mm"] = None if mt_px is None else mt_px / scale
        candidates = []
        for name, data in candidate_data:
            r = data.get(image_id, {})
            pred = {
                "pa_deg": parse_float(r.get("pa_deg")),
                "fl_mm": parse_float(r.get("fl_mm")),
                "mt_mm": parse_float(r.get("mt_mm")),
            }
            deltas, overall = norm_delta(human, pred)
            cand = {"name": name, **pred, **deltas, "overall_norm": overall}
            candidates.append(cand)
            if overall is not None:
                summary_acc[name]["overall"].append(overall)
            if deltas["delta_pa"] is not None:
                summary_acc[name]["pa"].append(abs(deltas["delta_pa"]) / TOL["pa_deg"])
            if deltas["delta_fl"] is not None:
                summary_acc[name]["fl"].append(abs(deltas["delta_fl"]) / TOL["fl_mm"])
            if deltas["delta_mt"] is not None:
                summary_acc[name]["mt"].append(abs(deltas["delta_mt"]) / TOL["mt_mm"])
        sort_score = candidates[0]["overall_norm"] if candidates else None
        rows.append({
            "label_id": label_id,
            "image_id": image_id,
            "image_path": src.get("image_path", ""),
            "source": src.get("source", ""),
            "quality": score.get("quality", ""),
            "apo_pixels": score.get("apo_pixels", ""),
            "fasc_pixels": score.get("fasc_pixels", ""),
            "n_fascicles": score.get("n_fascicles", ""),
            "human": human,
            "candidates": candidates,
            "sort_score": sort_score,
            "review": load_review(review_dir / f"{label_id}.json"),
        })

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    summary = []
    for name, acc in summary_acc.items():
        summary.append({
            "name": name,
            "overall_norm": mean(acc["overall"]),
            "pa_norm": mean(acc["pa"]),
            "fl_norm": mean(acc["fl"]),
            "mt_norm": mean(acc["mt"]),
            "n": len(acc["overall"]),
        })
    rows.sort(key=lambda r: (-1 if r["sort_score"] is None else -r["sort_score"], r["image_id"]))
    summary.sort(key=lambda s: 999 if s["overall_norm"] is None else s["overall_norm"])
    return rows, summary


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    summary: list[dict] = []
    labels_dir: Path = ROOT / "results" / "human_benchmark" / "target_labels"
    review_dir: Path = ROOT / "results" / "human_benchmark" / "review_notes"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text: str, status: int = 200) -> None:
        self.send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def send_json(self, value, status: int = 200) -> None:
        self.send_bytes(json.dumps(value).encode("utf-8"), "application/json", status)

    def row_for(self, idx_text: str) -> dict | None:
        try:
            idx = int(idx_text)
        except ValueError:
            return None
        if not (0 <= idx < len(self.rows)):
            return None
        return self.rows[idx]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/review":
            self.send_json({"rows": self.rows, "summary": self.summary})
            return
        if len(parts) == 2 and parts[0] == "image":
            row = self.row_for(parts[1])
            if row is None:
                self.send_text("bad image index", 404)
                return
            path = Path(row["image_path"])
            if not path.exists():
                self.send_text(f"missing image: {path}", 404)
                return
            data, ctype = browser_image_bytes(path)
            self.send_bytes(data, ctype)
            return
        if len(parts) == 3 and parts[0] == "label":
            row = self.row_for(parts[1])
            layer = parts[2].replace(".png", "")
            if row is None or layer not in LAYERS:
                self.send_text("bad label path", 404)
                return
            path = self.labels_dir / row["label_id"] / f"{layer}.png"
            if not path.exists():
                self.send_text("not labeled yet", 404)
                return
            self.send_bytes(path.read_bytes(), "image/png")
            return
        self.send_text("not found", 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save_review":
            self.send_text("not found", 404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            idx = int(payload["index"])
            row = self.rows[idx]
        except Exception as exc:
            self.send_text(f"bad save payload: {exc}", 400)
            return
        self.review_dir.mkdir(parents=True, exist_ok=True)
        review = {
            "label_id": row["label_id"],
            "image_id": row["image_id"],
            "label_quality": payload.get("label_quality", ""),
            "failure_kind": payload.get("failure_kind", ""),
            "notes": payload.get("notes", ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.review_dir / f"{row['label_id']}.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
        row["review"] = review
        self.send_json({"ok": True, "review": review})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "results" / "human_benchmark" / "target_seed_manifest.csv"))
    ap.add_argument("--labels-dir", default=str(ROOT / "results" / "human_benchmark" / "target_labels"))
    ap.add_argument("--scores", default=str(ROOT / "results" / "human_benchmark" / "target_scores.csv"))
    ap.add_argument("--calibration", default=str(ROOT / "results" / "calibration_measurement_debug.csv"))
    ap.add_argument("--review-dir", default=str(ROOT / "results" / "human_benchmark" / "review_notes"))
    ap.add_argument("--pred-csv", action="append", default=[],
                    help="candidate as name=path or just path; can be repeated")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()

    candidate_csvs = parse_candidate_arg(args.pred_csv) if args.pred_csv else default_candidate_csvs()
    rows, summary = build_rows(
        Path(args.manifest),
        Path(args.labels_dir),
        Path(args.scores),
        Path(args.calibration),
        candidate_csvs,
        Path(args.review_dir),
    )
    Handler.rows = rows
    Handler.summary = summary
    Handler.labels_dir = Path(args.labels_dir)
    Handler.review_dir = Path(args.review_dir)
    Handler.review_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"UMUD label review: http://{args.host}:{args.port}")
    print(f"labeled rows: {len(rows)}")
    print("candidates:")
    for name, path in candidate_csvs:
        print(f"  {name}: {path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
