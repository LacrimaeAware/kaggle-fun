"""Tiny local labeling server for UMUD benchmark masks.

No Flask, no database. It serves a browser canvas and writes masks/metadata to disk.

Example:
    python benchmark_lab/label_server.py --manifest results/human_benchmark/manifest.csv
"""

from __future__ import annotations

import argparse
import base64
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
LAYERS = ("apo", "fasc", "ignore")


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>UMUD Benchmark Lab</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: #161616;
      color: #ededed;
      overflow: hidden;
    }
    #topbar {
      height: 54px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-bottom: 1px solid #333;
      background: #202020;
      white-space: nowrap;
    }
    button, select, input, textarea {
      font: inherit;
      border: 1px solid #444;
      background: #2b2b2b;
      color: #f0f0f0;
      border-radius: 6px;
      padding: 6px 8px;
    }
    button { cursor: pointer; }
    button.active { outline: 2px solid #7ab7ff; background: #123a57; }
    button.danger { background: #4a2222; }
    #wrap {
      height: calc(100vh - 54px);
      display: grid;
      grid-template-columns: 1fr 340px;
      min-width: 0;
    }
    #stage {
      position: relative;
      overflow: auto;
      background: #050505;
      touch-action: none;
    }
    #stack {
      position: relative;
      width: max-content;
      height: max-content;
      margin: 18px;
      image-rendering: auto;
    }
    canvas {
      position: absolute;
      inset: 0;
      transform-origin: top left;
    }
    #base { position: relative; z-index: 0; }
    #apo { z-index: 2; opacity: 0.75; }
    #fasc { z-index: 3; opacity: 0.75; }
    #ignore { z-index: 4; opacity: 0.45; }
    #drawShield { z-index: 10; }
    #side {
      border-left: 1px solid #333;
      background: #1e1e1e;
      overflow: auto;
      padding: 12px;
    }
    .row { display: flex; align-items: center; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
    .kv { font-size: 12px; color: #bbb; line-height: 1.45; word-break: break-word; }
    .kv b { color: #f5f5f5; }
    .section { border-top: 1px solid #333; margin-top: 12px; padding-top: 12px; }
    .legend { display: grid; gap: 6px; margin-top: 8px; }
    .legend div { display: flex; gap: 8px; align-items: flex-start; }
    .swatch { width: 13px; height: 13px; border-radius: 3px; flex: 0 0 auto; margin-top: 2px; }
    .swatch.apo { background: rgb(0,220,255); }
    .swatch.fasc { background: rgb(255,65,65); }
    .swatch.ignore { background: rgb(255,210,0); }
    textarea { width: 100%; height: 82px; resize: vertical; }
    input[type="range"] { width: 150px; }
    input.short { width: 88px; }
    #status { color: #9fe59f; }
    #warn { color: #ffcc7a; }
    #list {
      max-height: 220px;
      overflow: auto;
      border: 1px solid #333;
      border-radius: 6px;
    }
    .item {
      padding: 6px 8px;
      border-bottom: 1px solid #333;
      cursor: pointer;
      font-size: 12px;
    }
    .item.current { background: #123a57; }
    .item.saved::before { content: "saved "; color: #9fe59f; }
  </style>
</head>
<body>
  <div id="topbar">
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <span id="counter"></span>
    <button id="save">Save</button>
    <span id="status"></span>
    <span id="warn"></span>
  </div>
  <div id="wrap">
    <div id="stage"><div id="stack">
      <canvas id="base"></canvas>
      <canvas id="apo"></canvas>
      <canvas id="fasc"></canvas>
      <canvas id="ignore"></canvas>
      <canvas id="drawShield"></canvas>
    </div></div>
    <aside id="side">
      <div class="kv" id="meta"></div>
      <div class="section">
        <div class="row">
          <button class="layer active" data-layer="apo">apo</button>
          <button class="layer" data-layer="fasc">fasc</button>
          <button class="layer" data-layer="ignore">ignore</button>
          <button id="eraser">eraser</button>
        </div>
        <div class="legend kv">
          <div><span class="swatch apo"></span><span><b>apo</b>: visible upper/lower boundary bands.</span></div>
          <div><span class="swatch fasc"></span><span><b>fasc</b>: visible slanted fiber fragments only.</span></div>
          <div><span class="swatch ignore"></span><span><b>ignore</b>: text, shadows, or areas too ambiguous to trust.</span></div>
        </div>
        <div class="row">
          <button class="tool active" data-tool="brush">brush</button>
          <button class="tool" data-tool="line">dot line</button>
          <button class="tool" data-tool="curve">curve chain</button>
          <button id="resetPath">reset dots</button>
        </div>
        <div class="row">
          <label>width <input id="brush" type="range" min="1" max="32" value="5"></label>
          <span id="brushVal">5</span>
        </div>
        <div class="kv" id="pathHint"></div>
        <div class="row">
          <button id="undo">Undo</button>
          <button id="clear" class="danger">Clear active</button>
        </div>
      </div>
      <div class="section">
        <div class="row">
          <label>quality
            <select id="quality">
              <option value="ok">ok</option>
              <option value="uncertain">uncertain</option>
              <option value="bad_image">bad image</option>
              <option value="skip">skip</option>
            </select>
          </label>
        </div>
        <div class="kv"><b>Optional manual measurements.</b> Leave these blank unless you intentionally
          measured them somewhere else. The scorer derives PA/FL/MT from your masks.</div>
        <div class="row">
          <label>scale optional <input id="scale" class="short" type="text"></label>
          <label>PA optional <input id="pa" class="short" type="text"></label>
        </div>
        <div class="row">
          <label>FL optional <input id="fl" class="short" type="text"></label>
          <label>MT optional <input id="mt" class="short" type="text"></label>
        </div>
        <label>notes</label>
        <textarea id="notes"></textarea>
      </div>
      <div class="section">
        <div class="kv">
          Shortcuts: A/F/I layer, B brush, L dot line, C curve chain, E eraser, Esc reset dots,
          S save, arrows navigate, [ and ] width. Save writes local mask files only.
          Draw visible structures only. Do not hand-extrapolate fascicles.
        </div>
      </div>
      <div class="section">
        <div id="list"></div>
      </div>
    </aside>
  </div>
<script>
const state = {
  rows: [],
  idx: 0,
  layer: 'apo',
  tool: 'brush',
  erasing: false,
  drawing: false,
  last: null,
  pathPoint: null,
  curvePoints: [],
  curveBase: null,
  curveStyle: null,
  undo: [],
  brush: 5,
  scale: 1,
};
const colors = {apo: 'rgba(0,220,255,1)', fasc: 'rgba(255,65,65,1)', ignore: 'rgba(255,210,0,1)'};
const canvases = {};
for (const id of ['base','apo','fasc','ignore','drawShield']) canvases[id] = document.getElementById(id);
const ctx = Object.fromEntries(Object.entries(canvases).map(([k,c]) => [k, c.getContext('2d')]));

function qs(id){ return document.getElementById(id); }
function row(){ return state.rows[state.idx]; }
function labelUrl(layer){ return `/label/${state.idx}/${layer}.png?t=${Date.now()}`; }
function setStatus(txt){ qs('status').textContent = txt; setTimeout(()=>{ if(qs('status').textContent===txt) qs('status').textContent=''; }, 2500); }
function setWarn(txt){ qs('warn').textContent = txt || ''; }

function resizeAll(w, h) {
  for (const c of Object.values(canvases)) { c.width = w; c.height = h; c.style.width = w + 'px'; c.style.height = h + 'px'; }
  qs('stack').style.width = w + 'px';
  qs('stack').style.height = h + 'px';
}

function clearLayer(layer) {
  ctx[layer].clearRect(0,0,canvases[layer].width,canvases[layer].height);
}

function setLayer(layer) {
  state.layer = layer;
  resetPath();
  document.querySelectorAll('button.layer').forEach(b => b.classList.toggle('active', b.dataset.layer === layer));
}

function setTool(tool) {
  state.tool = tool;
  resetPath();
  document.querySelectorAll('button.tool').forEach(b => b.classList.toggle('active', b.dataset.tool === tool));
  setPathHint();
}

function resetPath() {
  state.pathPoint = null;
  state.curvePoints = [];
  state.curveBase = null;
  state.curveStyle = null;
  clearPreview();
  setPathHint();
}

function setPathHint(text) {
  const hint = qs('pathHint');
  if (!hint) return;
  if (text) {
    hint.textContent = text;
  } else if (state.tool === 'line') {
    hint.textContent = 'dot line: click points; each click connects from the previous point.';
  } else if (state.tool === 'curve') {
    hint.textContent = 'curve chain: click anchors along the visible path; new points smooth the previous segment.';
  } else {
    hint.textContent = 'brush: draw normally with pen or mouse.';
  }
}

function clearPreview() {
  ctx.drawShield.clearRect(0, 0, canvases.drawShield.width, canvases.drawShield.height);
}

function previewDot(p, label, color) {
  const c = ctx.drawShield;
  c.save();
  c.lineWidth = 2;
  c.strokeStyle = color;
  c.fillStyle = 'rgba(0,0,0,0.72)';
  c.beginPath();
  c.arc(p.x, p.y, 6, 0, Math.PI * 2);
  c.fill();
  c.stroke();
  c.fillStyle = color;
  c.font = '12px system-ui, sans-serif';
  c.fillText(label, p.x + 9, p.y - 8);
  c.restore();
}

function previewGuide(points) {
  if (points.length < 2) return;
  const c = ctx.drawShield;
  c.save();
  c.lineWidth = 1;
  c.setLineDash([6, 5]);
  c.strokeStyle = 'rgba(255,255,255,0.72)';
  c.beginPath();
  c.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) c.lineTo(points[i].x, points[i].y);
  c.stroke();
  c.restore();
}

function renderPreview() {
  clearPreview();
  if (state.tool === 'line' && state.pathPoint) {
    previewDot(state.pathPoint, 'last', '#7ab7ff');
  }
  if (state.tool === 'curve' && state.curvePoints.length) {
    previewGuide(state.curvePoints);
    state.curvePoints.forEach((p, i) => {
      const isLast = i === state.curvePoints.length - 1;
      const label = isLast ? `p${i + 1} last` : `p${i + 1}`;
      previewDot(p, label, isLast ? '#ffd36f' : '#7ab7ff');
    });
  }
}

function pushUndo() {
  state.undo.push({layer: state.layer, data: canvases[state.layer].toDataURL('image/png')});
  if (state.undo.length > 20) state.undo.shift();
}

function restoreDataUrl(layer, url) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => { clearLayer(layer); ctx[layer].drawImage(im, 0, 0); resolve(); };
    im.onerror = () => resolve();
    im.src = url;
  });
}

async function loadExistingLayer(layer) {
  clearLayer(layer);
  const im = new Image();
  await new Promise((resolve) => {
    im.onload = () => { ctx[layer].drawImage(im, 0, 0); resolve(); };
    im.onerror = () => resolve();
    im.src = labelUrl(layer);
  });
}

async function loadMeta() {
  qs('quality').value = 'ok';
  qs('notes').value = '';
  qs('scale').value = row().scale_px_per_mm || '';
  qs('pa').value = '';
  qs('fl').value = '';
  qs('mt').value = '';
  try {
    const r = await fetch(`/api/meta/${state.idx}?t=${Date.now()}`);
    if (!r.ok) return;
    const m = await r.json();
    qs('quality').value = m.quality || 'ok';
    qs('notes').value = m.notes || '';
    qs('scale').value = m.scale_px_per_mm ?? (row().scale_px_per_mm || '');
    qs('pa').value = m.pa_deg ?? '';
    qs('fl').value = m.fl_mm ?? '';
    qs('mt').value = m.mt_mm ?? '';
  } catch {}
}

async function loadImage() {
  setWarn('');
  const r = row();
  qs('counter').textContent = `${state.idx + 1}/${state.rows.length}`;
  qs('meta').innerHTML =
    `<b>${r.label_id}</b><br>` +
    `source: ${r.source}<br>` +
    `mode: ${r.label_mode}<br>` +
    `image: ${r.image_id}<br>` +
    `priority: ${r.priority || ''}<br>` +
    `notes: ${r.notes || ''}`;
  const im = new Image();
  im.onload = async () => {
    resizeAll(im.naturalWidth, im.naturalHeight);
    ctx.base.clearRect(0,0,im.naturalWidth,im.naturalHeight);
    ctx.base.drawImage(im, 0, 0);
    for (const layer of ['apo','fasc','ignore']) await loadExistingLayer(layer);
    await loadMeta();
    state.undo = [];
    resetPath();
    renderList();
  };
  im.onerror = () => setWarn('could not load image');
  im.src = `/image/${state.idx}?t=${Date.now()}`;
}

function pointerPos(ev) {
  const rect = canvases.drawShield.getBoundingClientRect();
  return {
    x: (ev.clientX - rect.left) * canvases.drawShield.width / rect.width,
    y: (ev.clientY - rect.top) * canvases.drawShield.height / rect.height,
  };
}

function prepareStroke() {
  const c = ctx[state.layer];
  c.save();
  c.lineCap = 'round';
  c.lineJoin = 'round';
  c.lineWidth = state.brush;
  c.globalCompositeOperation = state.erasing ? 'destination-out' : 'source-over';
  c.strokeStyle = colors[state.layer];
  c.fillStyle = colors[state.layer];
  return c;
}

function drawDot(p) {
  const c = prepareStroke();
  c.beginPath();
  c.arc(p.x, p.y, Math.max(1, state.brush / 2), 0, Math.PI * 2);
  c.fill();
  c.restore();
}

function drawLine(a, b) {
  const c = prepareStroke();
  c.beginPath();
  c.moveTo(a.x, a.y);
  c.lineTo(b.x, b.y);
  c.stroke();
  c.restore();
}

function ensureCurveBase() {
  if (state.curveBase && state.curveStyle && state.curveStyle.layer === state.layer) return;
  pushUndo();
  state.curveBase = ctx[state.layer].getImageData(0, 0, canvases[state.layer].width, canvases[state.layer].height);
  state.curveStyle = {layer: state.layer, width: state.brush, erasing: state.erasing};
}

function restoreCurveBase() {
  if (!state.curveBase || !state.curveStyle) return;
  ctx[state.curveStyle.layer].putImageData(state.curveBase, 0, 0);
}

function drawSmoothChain(points) {
  if (!state.curveStyle || points.length < 2) return;
  const c = ctx[state.curveStyle.layer];
  const style = state.curveStyle;
  c.save();
  c.lineCap = 'round';
  c.lineJoin = 'round';
  c.lineWidth = style.width;
  c.globalCompositeOperation = style.erasing ? 'destination-out' : 'source-over';
  c.strokeStyle = colors[style.layer];
  c.beginPath();
  c.moveTo(points[0].x, points[0].y);
  if (points.length === 2) {
    c.lineTo(points[1].x, points[1].y);
  } else {
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[Math.max(0, i - 1)];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = points[Math.min(points.length - 1, i + 2)];
      const cp1 = {x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6};
      const cp2 = {x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6};
      c.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, p2.x, p2.y);
    }
  }
  c.stroke();
  c.restore();
}

function redrawCurveChain() {
  if (!state.curveBase || !state.curveStyle) return;
  restoreCurveBase();
  drawSmoothChain(state.curvePoints);
}

function drawTo(p) {
  drawLine(state.last, p);
  state.last = p;
}

function handlePointTool(p) {
  if (state.tool === 'line') {
    pushUndo();
    if (!state.pathPoint) {
      drawDot(p);
      state.pathPoint = p;
      setPathHint('line start set; click the next point to connect.');
    } else {
      drawLine(state.pathPoint, p);
      state.pathPoint = p;
      setPathHint('connected; keep clicking points or reset dots.');
    }
    renderPreview();
    return;
  }

  if (state.tool === 'curve') {
    ensureCurveBase();
    state.curvePoints.push(p);
    if (state.curvePoints.length === 1) {
      setPathHint('curve start set; click the next visible point.');
    } else {
      redrawCurveChain();
      setPathHint('curve smoothed; keep clicking anchors along the visible path.');
    }
    renderPreview();
  }
}

function startBrush(p) {
  state.drawing = true;
  state.last = p;
  pushUndo();
  drawDot(p);
}

canvases.drawShield.addEventListener('pointerdown', ev => {
  ev.preventDefault();
  canvases.drawShield.setPointerCapture(ev.pointerId);
  const p = pointerPos(ev);
  if (state.tool === 'brush') {
    startBrush(p);
  } else {
    handlePointTool(p);
  }
});
canvases.drawShield.addEventListener('pointermove', ev => {
  if (!state.drawing || state.tool !== 'brush') return;
  ev.preventDefault();
  drawTo(pointerPos(ev));
});
for (const name of ['pointerup','pointercancel','pointerleave']) {
  canvases.drawShield.addEventListener(name, () => { state.drawing = false; state.last = null; });
}

async function save() {
  function layerHasPixels(layer) {
    const data = ctx[layer].getImageData(0, 0, canvases[layer].width, canvases[layer].height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
    return false;
  }
  const hasAnyPixels = ['apo','fasc','ignore'].some(layerHasPixels);
  if (!hasAnyPixels && !qs('notes').value.trim()) {
    if (!confirm('No drawn pixels or notes on this image. Save a blank label anyway?')) return;
  }
  const payload = {
    index: state.idx,
    quality: qs('quality').value,
    notes: qs('notes').value,
    scale_px_per_mm: qs('scale').value,
    pa_deg: qs('pa').value,
    fl_mm: qs('fl').value,
    mt_mm: qs('mt').value,
    layers: {
      apo: canvases.apo.toDataURL('image/png'),
      fasc: canvases.fasc.toDataURL('image/png'),
      ignore: canvases.ignore.toDataURL('image/png'),
    },
  };
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (!res.ok) { setWarn(await res.text()); return; }
  state.rows[state.idx].saved = true;
  setStatus('saved');
  renderList();
}

function renderList() {
  const box = qs('list');
  box.innerHTML = '';
  state.rows.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'item' + (i === state.idx ? ' current' : '') + (r.saved ? ' saved' : '');
    d.textContent = `${i+1}. ${r.label_id}`;
    d.onclick = () => { state.idx = i; loadImage(); };
    box.appendChild(d);
  });
}

document.querySelectorAll('button.layer').forEach(b => b.onclick = () => setLayer(b.dataset.layer));
document.querySelectorAll('button.tool').forEach(b => b.onclick = () => setTool(b.dataset.tool));
qs('eraser').onclick = () => { resetPath(); state.erasing = !state.erasing; qs('eraser').classList.toggle('active', state.erasing); };
qs('brush').oninput = ev => { resetPath(); state.brush = Number(ev.target.value); qs('brushVal').textContent = state.brush; };
qs('undo').onclick = async () => {
  resetPath();
  const u = state.undo.pop();
  if (u) await restoreDataUrl(u.layer, u.data);
};
qs('clear').onclick = () => { pushUndo(); clearLayer(state.layer); resetPath(); };
qs('resetPath').onclick = () => { resetPath(); setStatus('dots reset'); };
qs('save').onclick = save;
qs('prev').onclick = () => { if (state.idx > 0) { state.idx--; loadImage(); } };
qs('next').onclick = () => { if (state.idx + 1 < state.rows.length) { state.idx++; loadImage(); } };

window.addEventListener('keydown', ev => {
  if (ev.target && ['TEXTAREA','INPUT','SELECT'].includes(ev.target.tagName)) return;
  if (ev.key === 'a' || ev.key === 'A') setLayer('apo');
  if (ev.key === 'f' || ev.key === 'F') setLayer('fasc');
  if (ev.key === 'i' || ev.key === 'I') setLayer('ignore');
  if (ev.key === 'b' || ev.key === 'B') setTool('brush');
  if (ev.key === 'l' || ev.key === 'L') setTool('line');
  if (ev.key === 'c' || ev.key === 'C') setTool('curve');
  if (ev.key === 'e' || ev.key === 'E') qs('eraser').click();
  if (ev.key === 'Escape') { resetPath(); setStatus('dots reset'); }
  if (ev.key === 's' || ev.key === 'S') save();
  if (ev.key === 'ArrowLeft') qs('prev').click();
  if (ev.key === 'ArrowRight') qs('next').click();
  if (ev.key === '[') { qs('brush').value = Math.max(1, Number(qs('brush').value) - 1); qs('brush').dispatchEvent(new Event('input')); }
  if (ev.key === ']') { qs('brush').value = Math.min(32, Number(qs('brush').value) + 1); qs('brush').dispatchEvent(new Event('input')); }
});

async function init() {
  const res = await fetch('/api/manifest');
  state.rows = await res.json();
  if (!state.rows.length) { setWarn('manifest has no rows'); return; }
  await loadImage();
}
init();
</script>
</body>
</html>
"""


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def browser_image_bytes(path: Path) -> tuple[bytes, str]:
    """Return browser-displayable image bytes.

    Chrome/Edge do not reliably display TIFF, so convert TIFF/BMP through OpenCV when available.
    """
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


def read_manifest(path: Path, labels_dir: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        label_id = safe_id(row["label_id"])
        row["label_id"] = label_id
        row["saved"] = (labels_dir / label_id / "meta.json").exists()
    return rows


class Handler(BaseHTTPRequestHandler):
    rows: list[dict[str, str]] = []
    labels_dir: Path = ROOT / "results" / "human_benchmark" / "labels"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value, status: int = 200) -> None:
        self.send_bytes(json.dumps(value).encode("utf-8"), "application/json", status)

    def send_text(self, text: str, status: int = 200) -> None:
        self.send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def row_for(self, idx_text: str) -> dict[str, str] | None:
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
        if parsed.path == "/api/manifest":
            self.send_json(self.rows)
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
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "meta":
            row = self.row_for(parts[2])
            if row is None:
                self.send_text("bad meta index", 404)
                return
            path = self.labels_dir / row["label_id"] / "meta.json"
            if not path.exists():
                self.send_json({})
                return
            self.send_bytes(path.read_bytes(), "application/json")
            return
        self.send_text("not found", 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
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
        out = self.labels_dir / row["label_id"]
        out.mkdir(parents=True, exist_ok=True)
        for layer in LAYERS:
            data_url = payload.get("layers", {}).get(layer, "")
            if not data_url.startswith("data:image/png;base64,"):
                continue
            raw = base64.b64decode(data_url.split(",", 1)[1])
            (out / f"{layer}.png").write_bytes(raw)
        meta = {
            "label_id": row["label_id"],
            "image_id": row.get("image_id", ""),
            "source": row.get("source", ""),
            "label_mode": row.get("label_mode", ""),
            "image_path": row.get("image_path", ""),
            "quality": payload.get("quality", "ok"),
            "notes": payload.get("notes", ""),
            "scale_px_per_mm": payload.get("scale_px_per_mm", ""),
            "pa_deg": payload.get("pa_deg", ""),
            "fl_mm": payload.get("fl_mm", ""),
            "mt_mm": payload.get("mt_mm", ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        row["saved"] = "true"
        self.send_json({"ok": True, "label_id": row["label_id"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "results" / "human_benchmark" / "manifest.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "human_benchmark" / "labels"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    labels_dir = Path(args.out_dir)
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}\nRun make_manifest.py first.")
    labels_dir.mkdir(parents=True, exist_ok=True)
    Handler.labels_dir = labels_dir
    Handler.rows = read_manifest(manifest, labels_dir)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"UMUD benchmark labeler: http://{args.host}:{args.port}")
    print(f"manifest: {manifest}")
    print(f"labels:   {labels_dir}")
    print("Ctrl+C stops the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
