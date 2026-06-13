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
import sys
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert_consensus import robust_mean_for_row  # noqa: E402

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
LAYERS = ("apo", "fasc", "ignore", "diag")
VIEWER_V2_DIR = Path(__file__).resolve().parent / "viewer_v2"
VIEWER_V2_SESSION_DIR = ROOT / "results" / "viewer_v2_sessions"

MODEL_INFO = {
    "our_pipeline_true_scale": {
        "label": "Pipeline true scale",
        "color": "#5b8def",
        "kind": "baseline",
        "description": "Original benchmark pipeline with true benchmark scale.",
    },
    "robust_triangle_anchor": {
        "label": "Robust triangle",
        "color": "#d665d6",
        "kind": "geometry",
        "description": "Upper boundary uses the robust three-anchor triangle convention.",
    },
    "story_stack": {
        "label": "Story stack",
        "color": "#14a276",
        "kind": "class-aware",
        "description": "EXP48 class-aware stack: scan-support FL, gated PA, vertical MT.",
    },
    "weighted_story_fl": {
        "label": "Weighted story FL",
        "color": "#10b981",
        "kind": "weighted stack",
        "description": "EXP49: PA median, FL weighted by area times ultrasound-field support, MT vertical center.",
    },
    "story_weight_grid_best": {
        "label": "Story weight grid best",
        "color": "#34d399",
        "kind": "weighted stack",
        "description": "EXP50: PA median, FL weighted-trimmed by support and local trajectory residual, MT vertical center.",
    },
    "story_weight_same_story": {
        "label": "Story weight same-story",
        "color": "#2dd4bf",
        "kind": "weighted stack",
        "description": "EXP50: PA and FL both use the same area x ultrasound-field x trajectory residual weighting.",
    },
    "median_weight_blend_best": {
        "label": "Median/weight blend best",
        "color": "#a3e635",
        "kind": "weighted stack",
        "description": "EXP53: median anchor blended with weighted PA/FL support reducers.",
    },
    "fl_scan_support": {
        "label": "FL scan support",
        "color": "#f59e0b",
        "kind": "component",
        "description": "Strict scan-region and visible-support weighted FL component.",
    },
    "pa_per_band": {
        "label": "PA per-band",
        "color": "#8b5cf6",
        "kind": "component",
        "description": "Fragment-count weighted per-band PA component.",
    },
    "pa_conflict_gate": {
        "label": "PA local conflict gate",
        "color": "#06b6d4",
        "kind": "component",
        "description": "Older local-median PA conflict gate. This is not the wave non-crossing trial.",
    },
    "wave_non_crossing_trial": {
        "label": "Wave non-crossing trial",
        "color": "#22d3ee",
        "kind": "visual trial",
        "description": "User-proposed rotate-only wave correction: avoid projected fragment crossings, then remeasure PA/FL.",
    },
    "wave_non_crossing_updating_trial": {
        "label": "Wave non-crossing updating",
        "color": "#2dd4bf",
        "kind": "visual trial",
        "description": "Same rotate-only wave correction, but recomputes the consensus direction after accepted corrections.",
    },
    "mt_vertical_center": {
        "label": "MT vertical center",
        "color": "#ef4444",
        "kind": "component",
        "description": "Vertical center-gap MT convention.",
    },
    "best_local_all_features": {
        "label": "Best local stack",
        "color": "#84cc16",
        "kind": "stack",
        "description": "Best full-feature stack from EXP44, before EXP48 class-aware story gating.",
    },
    "DLTrack": {
        "label": "DLTrack",
        "color": "#94a3b8",
        "kind": "reference",
        "description": "Reference benchmark method from the expert spreadsheet.",
    },
    "SMA": {
        "label": "SMA",
        "color": "#64748b",
        "kind": "reference",
        "description": "Reference benchmark method from the expert spreadsheet.",
    },
}

GEOMETRY_MODEL_IDS = {
    "our_pipeline_true_scale",
    "robust_triangle_anchor",
    "story_stack",
    "fl_scan_support",
    "pa_per_band",
    "best_local_all_features",
    "weighted_story_fl",
    "story_weight_grid_best",
    "story_weight_same_story",
    "median_weight_blend_best",
}


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
    #deltaStrip {
      position: sticky;
      top: 0;
      z-index: 30;
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 58px;
      padding: 9px 16px;
      background: rgba(18, 18, 18, 0.96);
      border-bottom: 1px solid #333;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
      white-space: nowrap;
    }
    .deltaTitle {
      color: #d8d8d8;
      font-size: 13px;
      font-weight: 650;
      margin-right: 2px;
    }
    .deltaChip {
      display: inline-flex;
      align-items: baseline;
      gap: 6px;
      border: 1px solid #404040;
      border-radius: 7px;
      padding: 7px 10px;
      background: #202020;
      font-weight: 750;
      font-size: 21px;
      line-height: 1;
    }
    .deltaChip .label {
      color: #f2f2f2;
      font-size: 13px;
      letter-spacing: 0;
    }
    .deltaChip .dir {
      font-size: 12px;
      text-transform: uppercase;
      color: #d5d5d5;
    }
    .deltaChip.over { color: #ff8a7a; border-color: #7d3b34; }
    .deltaChip.under { color: #79b7ff; border-color: #345176; }
    .deltaChip.neutral { color: #b8f0bd; border-color: #426946; }
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
    #diag { z-index: 1; opacity: 0.75; }
    #toolCanvas {
      position: absolute;
      inset: 0;
      z-index: 8;
      transform-origin: top left;
      cursor: crosshair;
    }
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
    .help { font-size: 12px; color: #d4d4d4; line-height: 1.4; }
    .muted { color: #999; }
    .kv b { color: #f5f5f5; }
    .section { border-top: 1px solid #333; margin-top: 12px; padding-top: 12px; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
    .pill { border: 1px solid #444; border-radius: 999px; padding: 2px 7px; color: #ddd; font-size: 12px; }
    .diagBox { border: 1px solid #3a3a3a; border-radius: 7px; padding: 9px; background: #181818; }
    .diagGuess { color: #f4f4f4; font-size: 13px; font-weight: 700; line-height: 1.35; margin-bottom: 8px; }
    .diagGrid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 8px; }
    .diagStat { border: 1px solid #3b3b3b; border-radius: 6px; padding: 6px; background: #202020; }
    .diagStat b { display: block; color: #f2f2f2; font-size: 14px; }
    .diagStat span { color: #aaa; font-size: 11px; }
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
      <div id="deltaStrip"></div>
      <div id="stack">
        <img id="base">
        <img id="apo">
        <img id="fasc">
        <img id="ignore">
        <img id="diag">
        <canvas id="toolCanvas"></canvas>
      </div>
    </main>
    <aside id="side">
      <div id="meta" class="kv"></div>
      <div id="fragmentDiag" class="section"></div>
      <div class="section">
        <div class="row">
          <button class="layer active" data-layer="apo">apo</button>
          <button class="layer active" data-layer="fasc">fasc</button>
          <button class="layer" data-layer="ignore">ignore</button>
          <button class="layer" data-layer="diag">old diag</button>
        </div>
        <div class="row">
          <label>overlay <input id="opacity" type="range" min="0" max="100" value="75"></label>
          <span id="opacityVal">75%</span>
        </div>
      </div>
      <div class="section">
        <div class="help"><b>Measurement scratch pad</b><br>
          Straight ruler: click two points. Trial FL: click two endpoints for a trial line; the viewer extends that line to the two apo boundaries and updates the median.
        </div>
        <div class="row">
          <button class="tool active" data-tool="off">inspect</button>
          <button class="tool" data-tool="sup">upper boundary</button>
          <button class="tool" data-tool="deep">lower boundary</button>
          <button class="tool" data-tool="ruler">straight ruler</button>
          <button class="tool" data-tool="trial">trial FL</button>
          <button id="undoTool">undo</button>
          <button id="clearTools">clear</button>
        </div>
        <div id="scaleBox" class="kv"></div>
        <div id="toolReadout" class="kv"></div>
      </div>
      <div class="section">
        <div class="help"><b>Disagreement is not out of 10.</b> 1.0 means one competition tolerance on average: PA 6 deg, FL 12 mm, MT 3 mm. Lower is closer to your mask.</div>
        <div class="kv" style="margin-top:8px;"><b>Candidate disagreement on labeled rows</b></div>
        <div id="summary"></div>
      </div>
      <div class="section">
        <div class="kv"><b>Current image: human mask vs candidate</b></div>
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
        Shortcuts: arrows navigate, S saves review note, A/F/I/D toggle layers, Ctrl/Cmd +/-/0 zoom.
      </div>
    </aside>
  </div>
<script>
const state = {
  rows: [],
  summary: [],
  idx: 0,
  zoom: 1,
  visible: {apo: true, fasc: true, ignore: false, diag: false},
  tool: 'off',
  pending: [],
  ruler: null,
  trialLines: [],
  manualApo: {superficial: null, deep: null},
  drag: null,
  dragMoved: false
};
const imgs = {base: document.getElementById('base'), apo: document.getElementById('apo'), fasc: document.getElementById('fasc'), ignore: document.getElementById('ignore'), diag: document.getElementById('diag')};
const toolCanvas = document.getElementById('toolCanvas');
const toolCtx = toolCanvas.getContext('2d');
function qs(id){ return document.getElementById(id); }
function row(){ return state.rows[state.idx]; }
function fmt(v, digits=2){ if(v===null || v===undefined || v==='') return ''; const n=Number(v); return Number.isFinite(n) ? n.toFixed(digits) : ''; }
function signed(v, digits=2){ if(v===null || v===undefined || v==='') return ''; const n=Number(v); return Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(digits) : ''; }
function num(v){ const n = Number(v); return Number.isFinite(n) ? n : null; }
function clsNorm(n){ if(n === null || n === undefined || n === '') return 'ok'; n=Number(n); if(n >= 1) return 'bad'; if(n <= 0.35) return 'good'; return 'ok'; }
function setStatus(txt){ qs('status').textContent = txt; setTimeout(()=>{ if(qs('status').textContent===txt) qs('status').textContent=''; }, 2500); }
function dist(a, b){ return Math.hypot(a.x - b.x, a.y - b.y); }
function median(vals){
  const xs = vals.filter(v => Number.isFinite(v)).slice().sort((a,b)=>a-b);
  if (!xs.length) return null;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}
function pxToMm(px){
  const scale = Number(row()?.scale_px_per_mm);
  return Number.isFinite(px) && Number.isFinite(scale) && scale > 0 ? px / scale : null;
}
function lengthText(px){
  if (!Number.isFinite(px)) return '';
  const mm = pxToMm(px);
  return `${fmt(px,1)} px` + (mm === null ? '' : ` / ${fmt(mm,2)} mm`);
}

function applyZoom(){
  const w = imgs.base.naturalWidth || 1;
  const h = imgs.base.naturalHeight || 1;
  const sw = Math.round(w * state.zoom);
  const sh = Math.round(h * state.zoom);
  for (const im of Object.values(imgs)) {
    im.style.width = sw + 'px';
    im.style.height = sh + 'px';
  }
  toolCanvas.style.width = sw + 'px';
  toolCanvas.style.height = sh + 'px';
  qs('stack').style.width = sw + 'px';
  qs('stack').style.height = sh + 'px';
  qs('zoomReset').textContent = Math.round(state.zoom * 100) + '%';
  drawTools();
}
function setZoom(z){ state.zoom = Math.min(4, Math.max(0.35, z)); applyZoom(); }

function resizeToolCanvas(){
  const w = imgs.base.naturalWidth || 1;
  const h = imgs.base.naturalHeight || 1;
  if (toolCanvas.width !== w || toolCanvas.height !== h) {
    toolCanvas.width = w;
    toolCanvas.height = h;
  }
  applyZoom();
}

function hasLayer(layer){
  const r = row();
  if (!r) return false;
  return r.label_available !== false || !!(r.overlay_paths && r.overlay_paths[layer]);
}

function updateLayerVisibility(){
  for (const layer of ['apo','fasc','ignore','diag']) {
    imgs[layer].style.display = hasLayer(layer) && state.visible[layer] ? 'block' : 'none';
    document.querySelector(`button[data-layer="${layer}"]`).classList.toggle('active', state.visible[layer]);
  }
  const op = Number(qs('opacity').value) / 100;
  qs('opacityVal').textContent = qs('opacity').value + '%';
  imgs.apo.style.opacity = op;
  imgs.fasc.style.opacity = op;
  imgs.ignore.style.opacity = Math.min(op, 0.55);
  imgs.diag.style.opacity = op;
}

function setTool(tool){
  state.tool = tool;
  state.pending = [];
  document.querySelectorAll('.tool').forEach(btn => btn.classList.toggle('active', btn.dataset.tool === tool));
  drawTools();
}

function canvasPoint(ev){
  const rect = toolCanvas.getBoundingClientRect();
  return {
    x: (ev.clientX - rect.left) * toolCanvas.width / Math.max(rect.width, 1),
    y: (ev.clientY - rect.top) * toolCanvas.height / Math.max(rect.height, 1)
  };
}

function lineY(line, x){ return line.slope * x + line.intercept; }
function lineFromPoints(p1, p2){
  const dx = p2.x - p1.x;
  if (Math.abs(dx) < 1e-6) return {vertical: true, x: p1.x};
  const slope = (p2.y - p1.y) / dx;
  return {vertical: false, slope, intercept: p1.y - slope * p1.x};
}
function boundaryFromEndpoints(p1, p2){
  const line = lineFromPoints(p1, p2);
  if (line.vertical) return null;
  return {...line, p1, p2, manual: true};
}
function currentApoLines(){
  const r = row() || {};
  const base = r.apo_lines || {};
  const boundary = r.candidate_boundary || {};
  return {
    superficial: state.manualApo.superficial || base.superficial || boundary.top || null,
    deep: state.manualApo.deep || base.deep || boundary.deep || null
  };
}
function isPiecewiseBoundary(boundary){
  return boundary && boundary.type === 'piecewise' && Array.isArray(boundary.points) && boundary.points.length >= 2;
}
function boundarySegmentLine(a, b){
  return lineFromPoints(a, b);
}
function boundaryY(boundary, x){
  if (!boundary) return null;
  if (!isPiecewiseBoundary(boundary)) return lineY(boundary, x);
  const pts = boundary.points;
  const mid = pts[Math.floor(pts.length / 2)];
  const seg = x <= mid.x ? boundarySegmentLine(pts[0], pts[1]) : boundarySegmentLine(pts[1], pts[2]);
  return lineY(seg, x);
}
function intersectLineWithLine(testLine, apoLine){
  if (!apoLine) return null;
  if (testLine.vertical) return {x: testLine.x, y: lineY(apoLine, testLine.x)};
  const denom = testLine.slope - apoLine.slope;
  if (Math.abs(denom) < 1e-6) return null;
  const x = (apoLine.intercept - testLine.intercept) / denom;
  return {x, y: testLine.slope * x + testLine.intercept};
}
function intersectLineWithApo(testLine, apoBoundary, xref=null){
  if (!apoBoundary) return null;
  if (!isPiecewiseBoundary(apoBoundary)) return intersectLineWithLine(testLine, apoBoundary);
  const pts = apoBoundary.points;
  const hits = [];
  for (let i = 0; i + 1 < pts.length; i++) {
    const segLine = boundarySegmentLine(pts[i], pts[i + 1]);
    if (segLine.vertical) continue;
    const hit = intersectLineWithLine(testLine, segLine);
    if (!hit) continue;
    const lo = Math.min(pts[i].x, pts[i + 1].x);
    const hi = Math.max(pts[i].x, pts[i + 1].x);
    const onSegment = hit.x >= lo - 10 && hit.x <= hi + 10;
    const ref = xref === null ? hit.x : xref;
    hits.push({hit, onSegment, dist: Math.abs(hit.x - ref)});
  }
  if (!hits.length) return null;
  hits.sort((a, b) => (a.onSegment === b.onSegment ? a.dist - b.dist : (a.onSegment ? -1 : 1)));
  return hits[0].hit;
}
function trialFromEndpoints(p1, p2){
  const apo = currentApoLines();
  const sup = apo.superficial;
  const deep = apo.deep;
  const testLine = lineFromPoints(p1, p2);
  const upper = intersectLineWithApo(testLine, sup, (p1.x + p2.x) / 2);
  const lower = intersectLineWithApo(testLine, deep);
  if (!upper || !lower) return {p1, p2, visible_px: dist(p1, p2), error: 'no boundary intersection'};
  const visiblePx = dist(p1, p2);
  const fullPx = dist(upper, lower);
  let angleDeg = null;
  if (!testLine.vertical && deep) {
    angleDeg = Math.abs((Math.atan(testLine.slope) - Math.atan(deep.slope)) * 180 / Math.PI);
    if (angleDeg > 90) angleDeg = 180 - angleDeg;
  }
  return {
    p1, p2, upper, lower,
    visible_px: visiblePx,
    extrapolated_px: fullPx,
    support_ratio: fullPx > 0 ? visiblePx / fullPx : null,
    angle_deg: angleDeg,
    error: fullPx < 10 || fullPx > 4000 ? 'outside scorer length range' : ''
  };
}

function drawPoint(p, color){
  toolCtx.fillStyle = color;
  toolCtx.beginPath();
  toolCtx.arc(p.x, p.y, Math.max(2, 4 / state.zoom), 0, Math.PI * 2);
  toolCtx.fill();
}
function drawSegment(a, b, color, width=3, dashed=false){
  toolCtx.save();
  toolCtx.strokeStyle = color;
  toolCtx.lineWidth = Math.max(1, width / state.zoom);
  if (dashed) toolCtx.setLineDash([9 / state.zoom, 7 / state.zoom]);
  toolCtx.beginPath();
  toolCtx.moveTo(a.x, a.y);
  toolCtx.lineTo(b.x, b.y);
  toolCtx.stroke();
  toolCtx.restore();
}
function drawApoFit(line, color){
  if (!line || !toolCanvas.width) return;
  if (isPiecewiseBoundary(line)) {
    for (let i = 0; i + 1 < line.points.length; i++) {
      drawSegment(line.points[i], line.points[i + 1], color, 5, false);
    }
    for (const p of line.points) drawPoint(p, color);
    return;
  }
  drawSegment({x: 0, y: lineY(line, 0)}, {x: toolCanvas.width, y: lineY(line, toolCanvas.width)}, color, 2, true);
  if (line.manual && line.p1 && line.p2) {
    drawSegment(line.p1, line.p2, color, 5, false);
    drawPoint(line.p1, color);
    drawPoint(line.p2, color);
  }
}
function scratchStats(){
  const pa = median(state.trialLines.map(t => t.angle_deg));
  const flPx = median(state.trialLines.map(t => t.extrapolated_px));
  const flMm = pxToMm(flPx);
  const vals = [];
  const h = row()?.human || {};
  if (pa !== null && Number.isFinite(h.pa_deg)) vals.push(Math.abs(pa - h.pa_deg) / 6);
  if (flMm !== null && Number.isFinite(h.fl_mm)) vals.push(Math.abs(flMm - h.fl_mm) / 12);
  return {
    pa_deg: pa,
    fl_px: flPx,
    fl_mm: flMm,
    delta_pa: pa === null || !Number.isFinite(h.pa_deg) ? null : pa - h.pa_deg,
    delta_fl: flMm === null || !Number.isFinite(h.fl_mm) ? null : flMm - h.fl_mm,
    overall_norm: vals.length ? vals.reduce((a,b)=>a+b,0) / vals.length : null
  };
}
function drawTools(){
  toolCtx.clearRect(0, 0, toolCanvas.width, toolCanvas.height);
  const r = row();
  if (!r) return;
  const apo = currentApoLines();
  for (const span of (r.candidate_projection_spans || [])) {
    drawSegment({x: span.x1, y: span.y1}, {x: span.x2, y: span.y2}, 'rgba(99, 255, 121, 0.92)', 2, false);
  }
  drawApoFit(apo.superficial, isPiecewiseBoundary(apo.superficial) ? 'rgba(255, 80, 216, 0.95)' : 'rgba(111, 199, 255, 0.85)');
  drawApoFit(apo.deep, 'rgba(255, 214, 102, 0.85)');
  if (state.ruler) {
    drawSegment(state.ruler.p1, state.ruler.p2, '#6fffd2', 4);
    drawPoint(state.ruler.p1, '#6fffd2');
    drawPoint(state.ruler.p2, '#6fffd2');
  }
  for (const t of state.trialLines) {
    if (t.upper && t.lower) drawSegment(t.upper, t.lower, '#ffd966', 3, true);
    drawSegment(t.p1, t.p2, '#7aff8a', 4);
    drawPoint(t.p1, '#7aff8a');
    drawPoint(t.p2, '#7aff8a');
    if (t.upper) drawPoint(t.upper, '#ffd966');
    if (t.lower) drawPoint(t.lower, '#ffd966');
  }
  for (const p of state.pending) drawPoint(p, '#ff7ab7');
  renderToolReadout();
  renderCandidateTable();
}
function renderScaleBox(){
  const r = row();
  const scale = Number(r.scale_px_per_mm);
  if (Number.isFinite(scale) && scale > 0) {
    qs('scaleBox').innerHTML = `<b>scale used here:</b> ${fmt(scale,3)} px/mm (${fmt(scale * 10,1)} px/cm)` +
      (r.calibration_method ? `<br><span class="muted">method: ${r.calibration_method}</span>` : '');
  } else {
    qs('scaleBox').innerHTML = '<b>scale used here:</b> missing, so scratch lengths are pixels only';
  }
}
function renderToolReadout(){
  const parts = [];
  const apo = currentApoLines();
  if (!apo.superficial || !apo.deep) {
    parts.push('<b>boundary needed:</b> set upper and lower boundary before trial FL can extrapolate.');
  } else if (isPiecewiseBoundary(apo.superficial)) {
    parts.push('<b>candidate overlay:</b> magenta = robust triangle upper boundary; yellow = lower boundary; green = robust-triangle projected FL spans. The optional old diag layer is separate.');
  }
  if (state.tool === 'sup') parts.push(state.pending.length ? 'upper boundary: click the second point' : 'upper boundary: click two points along the upper line');
  if (state.tool === 'deep') parts.push(state.pending.length ? 'lower boundary: click the second point' : 'lower boundary: click two points along the lower line');
  if (state.tool === 'ruler') parts.push(state.pending.length ? 'ruler: click the second point' : 'ruler: click two points');
  if (state.tool === 'trial') parts.push(state.pending.length ? 'trial FL: click the second endpoint' : 'trial FL: click two endpoints for each segment');
  if (state.ruler) parts.push(`<b>ruler:</b> ${lengthText(dist(state.ruler.p1, state.ruler.p2))}`);
  if (state.trialLines.length) {
    const s = scratchStats();
    parts.push(`<b>scratch trial median:</b> PA ${fmt(s.pa_deg,1)} deg, FL ${fmt(s.fl_mm,2)} mm (${fmt(s.fl_px,1)} px), disagreement ${fmt(s.overall_norm,2)}`);
    parts.push(state.trialLines.map((t, i) => {
      const err = t.error ? ` <span class="bad">${t.error}</span>` : '';
      return `${i + 1}. full ${lengthText(t.extrapolated_px)}, clicked segment ${lengthText(t.visible_px)}, clicked/full ${fmt(t.support_ratio,2)}, angle ${fmt(t.angle_deg,1)}${err}`;
    }).join('<br>'));
  }
  qs('toolReadout').innerHTML = parts.length ? parts.join('<br>') : '<span class="muted">Select a tool above to measure without changing the saved label.</span>';
}
function handleToolClick(ev){
  if (state.dragMoved) {
    state.dragMoved = false;
    return;
  }
  if (state.tool === 'off') return;
  const p = canvasPoint(ev);
  if (state.tool === 'sup' || state.tool === 'deep') {
    if (!state.pending.length) {
      state.pending = [p];
    } else {
      const line = boundaryFromEndpoints(state.pending[0], p);
      if (line) state.manualApo[state.tool === 'sup' ? 'superficial' : 'deep'] = line;
      state.pending = [];
    }
  } else if (state.tool === 'ruler') {
    if (!state.pending.length) state.pending = [p];
    else { state.ruler = {p1: state.pending[0], p2: p}; state.pending = []; }
  } else if (state.tool === 'trial') {
    if (!state.pending.length) state.pending = [p];
    else { state.trialLines.push(trialFromEndpoints(state.pending[0], p)); state.pending = []; }
  }
  drawTools();
}

function handleAt(p, target){
  if (!target) return false;
  return dist(p, target) <= Math.max(8 / state.zoom, 4);
}
function findHandle(p){
  if (state.ruler) {
    if (handleAt(p, state.ruler.p1)) return {kind: 'ruler', point: 'p1'};
    if (handleAt(p, state.ruler.p2)) return {kind: 'ruler', point: 'p2'};
  }
  for (let i = state.trialLines.length - 1; i >= 0; i--) {
    const t = state.trialLines[i];
    if (handleAt(p, t.p1)) return {kind: 'trial', index: i, point: 'p1'};
    if (handleAt(p, t.p2)) return {kind: 'trial', index: i, point: 'p2'};
  }
  for (const role of ['superficial', 'deep']) {
    const line = state.manualApo[role];
    if (!line) continue;
    if (handleAt(p, line.p1)) return {kind: 'apo', role, point: 'p1'};
    if (handleAt(p, line.p2)) return {kind: 'apo', role, point: 'p2'};
  }
  return null;
}
function applyDrag(handle, p){
  if (handle.kind === 'ruler') {
    state.ruler[handle.point] = p;
  } else if (handle.kind === 'trial') {
    const old = state.trialLines[handle.index];
    const p1 = handle.point === 'p1' ? p : old.p1;
    const p2 = handle.point === 'p2' ? p : old.p2;
    state.trialLines[handle.index] = trialFromEndpoints(p1, p2);
  } else if (handle.kind === 'apo') {
    const old = state.manualApo[handle.role];
    const p1 = handle.point === 'p1' ? p : old.p1;
    const p2 = handle.point === 'p2' ? p : old.p2;
    state.manualApo[handle.role] = boundaryFromEndpoints(p1, p2) || old;
    state.trialLines = state.trialLines.map(t => trialFromEndpoints(t.p1, t.p2));
  }
}

function renderList(){
  const box = qs('list');
  box.innerHTML = '';
  const intro = document.createElement('div');
  intro.className = 'kv';
  intro.style.padding = '0 6px 8px';
  intro.innerHTML = '<b>Worst first</b><br><span class="muted">sorted by the first candidate error units</span>';
  box.appendChild(intro);
  state.rows.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'item' + (i === state.idx ? ' current' : '') + (r.review && r.review.updated_at ? ' reviewed' : '');
    d.innerHTML = `<b>${i+1}. ${r.image_id}</b><br><span class="score">disagree ${fmt(r.sort_score,2)}</span> ` +
      `<span class="kv">fasc ${r.n_fascicles || ''}</span><br>` +
      `<span class="kv">${r.review?.label_quality || ''} ${r.review?.failure_kind || ''}</span>`;
    d.onclick = () => { state.idx = i; loadCurrent(); };
    box.appendChild(d);
  });
}

function renderSummary(){
  let html = '<table class="summaryTable"><tr><th>candidate</th><th>overall err</th><th>PA err</th><th>FL err</th><th>MT err</th><th>n</th></tr>';
  for (const s of state.summary) {
    html += `<tr><td>${s.name}</td><td class="${clsNorm(s.overall_norm)}">${fmt(s.overall_norm,2)}</td>` +
      `<td>${fmt(s.pa_norm,2)}</td><td>${fmt(s.fl_norm,2)}</td><td>${fmt(s.mt_norm,2)}</td><td>${s.n}</td></tr>`;
  }
  html += '</table>';
  qs('summary').innerHTML = html;
}

function renderCandidateTable(){
  const r = row();
  let html = '<table><tr><th>source</th><th>PA deg</th><th>FL mm</th><th>MT mm</th><th>error units</th></tr>';
  html += `<tr><td>${r.truth_label || 'human mask'}</td><td>${fmt(r.human.pa_deg)}</td><td>${fmt(r.human.fl_mm)}</td><td>${fmt(r.human.mt_mm)}</td><td></td></tr>`;
  if (state.trialLines.length) {
    const s = scratchStats();
    html += `<tr><td>scratch trial median<br><span class="muted">live, not saved</span></td>` +
      `<td>${fmt(s.pa_deg)}<br><span class="muted">${signed(s.delta_pa)}</span></td>` +
      `<td>${fmt(s.fl_mm)}<br><span class="muted">${signed(s.delta_fl)}</span></td>` +
      `<td></td><td class="${clsNorm(s.overall_norm)}">${fmt(s.overall_norm,2)}</td></tr>`;
  }
  for (const c of r.candidates) {
    html += `<tr><td>${c.name}</td><td>${fmt(c.pa_deg)}</td><td>${fmt(c.fl_mm)}</td><td>${fmt(c.mt_mm)}</td><td class="${clsNorm(c.overall_norm)}">${fmt(c.overall_norm,2)}</td></tr>`;
    html += `<tr><td class="kv">candidate - human</td><td class="${clsNorm(Math.abs(c.delta_pa)/6)}">${signed(c.delta_pa)}</td>` +
      `<td class="${clsNorm(Math.abs(c.delta_fl)/12)}">${signed(c.delta_fl)}</td>` +
      `<td class="${clsNorm(Math.abs(c.delta_mt)/3)}">${signed(c.delta_mt)}</td><td></td></tr>`;
  }
  html += '</table>';
  qs('candidateTable').innerHTML = html;
}

function deltaChip(label, value, unit){
  if (!Number.isFinite(Number(value))) {
    return `<span class="deltaChip neutral"><span class="label">${label}</span> n/a</span>`;
  }
  const n = Number(value);
  const cls = Math.abs(n) < 0.005 ? 'neutral' : (n > 0 ? 'over' : 'under');
  const dir = Math.abs(n) < 0.005 ? 'even' : (n > 0 ? 'over' : 'under');
  return `<span class="deltaChip ${cls}"><span class="label">${label}</span>${signed(n,2)}${unit}<span class="dir">${dir}</span></span>`;
}

function renderDeltaStrip(){
  const r = row();
  const c = r?.candidates?.[0];
  if (!c) {
    qs('deltaStrip').innerHTML = '<span class="deltaTitle">No candidate deltas</span>';
    return;
  }
  qs('deltaStrip').innerHTML =
    `<span class="deltaTitle">${c.name} - ${r.truth_label || 'truth'}</span>` +
    deltaChip('PA', c.delta_pa, ' deg') +
    deltaChip('FL', c.delta_fl, ' mm') +
    deltaChip('MT', c.delta_mt, ' mm');
}

function renderFragmentDiagnostics(){
  const r = row();
  const tax = r?.taxonomy || {};
  const p50 = num(tax.projected_fl_p50_mm);
  if (p50 === null) {
    qs('fragmentDiag').innerHTML = '';
    return;
  }
  const truth = num(tax.truth_fl_mm);
  const truthPct = num(tax.projected_fl_truth_percentile);
  const p25Delta = num(tax.projected_fl_p25_delta_mm);
  const p25Improve = num(tax.projected_fl_p25_improvement_mm);
  const highOut = num(tax.projected_fl_tukey_high_count) || 0;
  const lowOut = num(tax.projected_fl_tukey_low_count) || 0;
  const tailLine = tax.projected_fl_tail_read || '';
  let guess = tailLine || 'No projection-tail read available.';
  if (truthPct !== null && truthPct <= 0.35 && p50 > truth) {
    guess = `Best guess: median projected FL is too high here; expert FL sits at about the ${fmt(truthPct * 100,0)}th percentile of our projected spans. `;
    guess += (lowOut || highOut) ? 'There are statistical tails too.' : 'This is not just one obvious statistical tail.';
  } else if (truthPct !== null && truthPct >= 0.65 && p50 < truth) {
    guess = `Best guess: median projected FL is too low here; expert FL sits at about the ${fmt(truthPct * 100,0)}th percentile.`;
  }
  qs('fragmentDiag').innerHTML = `
    <div class="diagBox">
      <div class="diagGuess">${guess}</div>
      <div class="kv"><b>Projected FL distribution</b> from the cyan full-length spans, in mm.</div>
      <div class="diagGrid">
        <div class="diagStat"><b>${fmt(tax.projected_fl_p10_mm,1)} / ${fmt(tax.projected_fl_p25_mm,1)}</b><span>p10 / p25</span></div>
        <div class="diagStat"><b>${fmt(tax.projected_fl_p50_mm,1)}</b><span>median used now</span></div>
        <div class="diagStat"><b>${fmt(tax.projected_fl_p75_mm,1)} / ${fmt(tax.projected_fl_p90_mm,1)}</b><span>p75 / p90</span></div>
        <div class="diagStat"><b>${fmt(truth,1)}</b><span>expert consensus</span></div>
        <div class="diagStat"><b>${truthPct === null ? '' : fmt(truthPct * 100,0) + '%'}</b><span>expert percentile inside our spans</span></div>
        <div class="diagStat"><b>${signed(p25Delta,1)} mm</b><span>p25 - expert; local gain ${fmt(p25Improve,1)} mm</span></div>
      </div>
      <div class="kv" style="margin-top:8px;"><b>Tail read:</b> ${tailLine}</div>
    </div>`;
}

function loadCurrent(){
  const r = row();
  state.pending = [];
  state.ruler = null;
  state.trialLines = [];
  state.manualApo = {superficial: null, deep: null};
  state.drag = null;
  state.dragMoved = false;
  qs('counter').textContent = `${state.idx + 1}/${state.rows.length}`;
  const tax = r.taxonomy || {};
  const taxHtml = tax.tags ? `<br><b>tags:</b> ${tax.tags}<br><b>likely:</b> ${tax.diagnosis || ''}` : '';
  qs('meta').innerHTML = `<b>${r.label_id}</b><br>image: ${r.image_id}<br>quality: ${r.quality || ''}<br>` +
    `pixels: apo ${r.apo_pixels}, fasc ${r.fasc_pixels}<br>measured fragments: ${r.n_fascicles || ''}<br>` +
    `sort disagreement: ${fmt(r.sort_score,2)}<br>` +
    (r.candidate_boundary?.mode ? `<b>candidate boundary:</b> ${r.candidate_boundary.mode}; magenta overlay on canvas<br>` : '') +
    `<span class="muted">${r.truth_note || 'FL here = median of drawn fascicle lines after straight extrapolation to the two apo boundaries.'}</span>${taxHtml}`;
  imgs.base.onload = resizeToolCanvas;
  imgs.base.src = `/image/${state.idx}?t=${Date.now()}`;
  for (const layer of ['apo','fasc','ignore','diag']) {
    if (hasLayer(layer)) imgs[layer].src = `/label/${state.idx}/${layer}.png?t=${Date.now()}`;
    else {
      imgs[layer].removeAttribute('src');
      imgs[layer].style.display = 'none';
    }
  }
  qs('labelQuality').value = r.review?.label_quality || '';
  qs('failureKind').value = r.review?.failure_kind || '';
  qs('notes').value = r.review?.notes || '';
  renderScaleBox();
  renderList();
  renderDeltaStrip();
  renderFragmentDiagnostics();
  renderCandidateTable();
  renderToolReadout();
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
document.querySelectorAll('button.tool').forEach(b => b.onclick = () => setTool(b.dataset.tool));
toolCanvas.addEventListener('click', handleToolClick);
toolCanvas.addEventListener('pointerdown', ev => {
  const p = canvasPoint(ev);
  const handle = findHandle(p);
  if (!handle) return;
  state.drag = handle;
  state.dragMoved = false;
  toolCanvas.setPointerCapture(ev.pointerId);
  ev.preventDefault();
});
toolCanvas.addEventListener('pointermove', ev => {
  if (!state.drag) return;
  applyDrag(state.drag, canvasPoint(ev));
  state.dragMoved = true;
  drawTools();
});
toolCanvas.addEventListener('pointerup', ev => {
  if (!state.drag) return;
  try { toolCanvas.releasePointerCapture(ev.pointerId); } catch {}
  state.drag = null;
});
qs('undoTool').onclick = () => {
  if (state.pending.length) state.pending = [];
  else if (state.trialLines.length) state.trialLines.pop();
  else if (state.manualApo.deep) state.manualApo.deep = null;
  else if (state.manualApo.superficial) state.manualApo.superficial = null;
  else state.ruler = null;
  drawTools();
};
qs('clearTools').onclick = () => {
  state.pending = [];
  state.ruler = null;
  state.trialLines = [];
  state.manualApo = {superficial: null, deep: null};
  drawTools();
};
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
  if (ev.key === 'd' || ev.key === 'D') { state.visible.diag = !state.visible.diag; updateLayerVisibility(); }
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


def xlsx_col_index(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref)
    if not m:
        return 0
    out = 0
    for ch in m.group(1):
        out = out * 26 + ord(ch) - ord("A") + 1
    return out - 1


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            vals: list[str] = []
            for cell in row.findall("m:c", ns):
                idx = xlsx_col_index(cell.attrib.get("r", "A"))
                while len(vals) <= idx:
                    vals.append("")
                kind = cell.attrib.get("t", "")
                value_el = cell.find("m:v", ns)
                if kind == "inlineStr":
                    vals[idx] = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                elif value_el is None:
                    vals[idx] = ""
                elif kind == "s":
                    vals[idx] = shared[int(value_el.text or "0")]
                else:
                    vals[idx] = value_el.text or ""
            if vals:
                rows.append(vals)
        if not rows:
            return []
        headers = rows[0]
        return [{headers[i]: (vals[i] if i < len(vals) else "") for i in range(len(headers))} for vals in rows[1:]]


def parse_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_mask(path: Path) -> np.ndarray | None:
    if cv2 is None or np is None or not path.exists():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 4:
        return (arr[:, :, 3] > 0).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return (arr > 0).astype(np.uint8)


def fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    m = np.polyfit(xs.astype(float), ys.astype(float), 1)
    return float(m[0]), float(m[1])


def apo_boundary_groups(apo_mask: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]] | None:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, 4])
        if area < 5:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 2:
            continue
        comps.append({"mean_y": float(np.mean(ys)), "xs": xs, "ys": ys, "area": area})
    if len(comps) < 2:
        return None
    comps.sort(key=lambda c: c["mean_y"])
    if len(comps) == 2:
        split = 1
    else:
        gaps = [comps[i + 1]["mean_y"] - comps[i]["mean_y"] for i in range(len(comps) - 1)]
        split = int(np.argmax(gaps)) + 1
    groups = [comps[:split], comps[split:]]
    if not groups[0] or not groups[1]:
        return None
    out = []
    for group in groups:
        xs = np.concatenate([g["xs"] for g in group])
        ys = np.concatenate([g["ys"] for g in group])
        out.append((float(np.average([g["mean_y"] for g in group], weights=[g["area"] for g in group])), xs, ys))
    return out


def candidate_apo_boundary_groups(apo_mask: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]] | None:
    """Select the primary measured band from model apo masks.

    Multi-band predictions often contain three or more long boundary traces.
    The old largest-gap split could merge the first two traces and pick a lower
    trace/artifact as the deep boundary. For candidate geometry, use the first
    two substantial horizontal boundary clusters instead.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, 4])
        if area < 5:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 2:
            continue
        comps.append({
            "mean_y": float(np.mean(ys)),
            "xs": xs,
            "ys": ys,
            "area": area,
            "y_min": int(np.min(ys)),
            "y_max": int(np.max(ys)),
            "x_span": int(np.max(xs) - np.min(xs) + 1),
        })
    if len(comps) < 2:
        return None
    comps.sort(key=lambda c: c["mean_y"])
    clusters: list[list[dict]] = []
    for comp in comps:
        if not clusters:
            clusters.append([comp])
            continue
        prev = clusters[-1]
        prev_ymax = max(c["y_max"] for c in prev)
        prev_mean = float(np.average([c["mean_y"] for c in prev], weights=[c["area"] for c in prev]))
        if comp["y_min"] <= prev_ymax + 20 or abs(comp["mean_y"] - prev_mean) <= 45:
            prev.append(comp)
        else:
            clusters.append([comp])
    cluster_rows = []
    for group in clusters:
        xs = np.concatenate([g["xs"] for g in group])
        ys = np.concatenate([g["ys"] for g in group])
        area = int(sum(g["area"] for g in group))
        cluster_rows.append({
            "mean_y": float(np.average([g["mean_y"] for g in group], weights=[g["area"] for g in group])),
            "xs": xs,
            "ys": ys,
            "area": area,
            "x_span": int(np.max(xs) - np.min(xs) + 1),
        })
    max_area = max(c["area"] for c in cluster_rows)
    max_span = max(c["x_span"] for c in cluster_rows)
    substantial = [
        c for c in cluster_rows
        if c["area"] >= max(80, 0.08 * max_area) and c["x_span"] >= max(40, 0.18 * max_span)
    ]
    if len(substantial) < 2:
        substantial = cluster_rows
    substantial.sort(key=lambda c: c["mean_y"])
    pair = substantial[:2]
    if len(pair) < 2:
        return None
    return [(c["mean_y"], c["xs"], c["ys"]) for c in pair]


def apo_line_fits(labels_dir: Path, label_id: str) -> dict[str, dict[str, float]] | None:
    apo = load_mask(labels_dir / label_id / "apo.png")
    if apo is None:
        return None
    groups = apo_boundary_groups(np.ascontiguousarray(apo, np.uint8))
    if groups is None:
        return None
    groups.sort(key=lambda item: item[0])
    out: dict[str, dict[str, float]] = {}
    for role, (_, xs, ys) in zip(("superficial", "deep"), groups):
        ux, inv = np.unique(xs, return_inverse=True)
        if len(ux) < 2:
            return None
        if role == "superficial":
            edge_y = np.full(len(ux), -1.0)
            np.maximum.at(edge_y, inv, ys.astype(float))
        else:
            edge_y = np.full(len(ux), 1e18)
            np.minimum.at(edge_y, inv, ys.astype(float))
        slope, intercept = fit_line(ux.astype(float), edge_y)
        out[role] = {"slope": slope, "intercept": intercept}
    return out


def robust_triangle_points(edge_x: np.ndarray, edge_y: np.ndarray) -> list[dict[str, float]] | None:
    if np is None or len(edge_x) < 12:
        return None
    q25, q75 = np.percentile(edge_x, [25, 75])
    left = np.where(edge_x <= q25)[0]
    center = np.where((edge_x >= q25) & (edge_x <= q75))[0]
    right = np.where(edge_x >= q75)[0]
    if len(left) == 0 or len(center) == 0 or len(right) == 0:
        return None

    def low(indices: np.ndarray) -> dict[str, float]:
        cutoff = np.percentile(edge_y[indices], 95)
        keep = indices[edge_y[indices] >= cutoff]
        return {"x": float(np.median(edge_x[keep])), "y": float(np.median(edge_y[keep]))}

    def high(indices: np.ndarray) -> dict[str, float]:
        cutoff = np.percentile(edge_y[indices], 5)
        keep = indices[edge_y[indices] <= cutoff]
        return {"x": float(np.median(edge_x[keep])), "y": float(np.median(edge_y[keep]))}

    return [low(left), high(center), low(right)]


def py_line_from_points(p1: dict[str, float], p2: dict[str, float]) -> dict[str, float] | None:
    dx = p2["x"] - p1["x"]
    if abs(dx) < 1e-9:
        return None
    slope = (p2["y"] - p1["y"]) / dx
    return {"slope": float(slope), "intercept": float(p1["y"] - slope * p1["x"])}


def py_line_intersection(a: dict[str, float], b: dict[str, float]) -> tuple[float, float] | None:
    denom = a["slope"] - b["slope"]
    if abs(denom) < 1e-9:
        return None
    x = (b["intercept"] - a["intercept"]) / denom
    return float(x), float(a["slope"] * x + a["intercept"])


def py_piecewise_intersection(line: dict[str, float], top: dict, xref: float) -> tuple[float, float] | None:
    if top.get("type") != "piecewise":
        return py_line_intersection(line, top)
    points = top.get("points") or []
    hits = []
    for p1, p2 in zip(points, points[1:]):
        seg = py_line_from_points(p1, p2)
        if seg is None:
            continue
        hit = py_line_intersection(line, seg)
        if hit is None:
            continue
        lo, hi = sorted((p1["x"], p2["x"]))
        on_segment = lo - 10.0 <= hit[0] <= hi + 10.0
        hits.append((0 if on_segment else 1, abs(hit[0] - xref), hit))
    if not hits:
        return None
    return sorted(hits, key=lambda item: (item[0], item[1]))[0][2]


def py_pca_line(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    ctr = pts.mean(axis=0)
    pts0 = pts - ctr
    try:
        _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    vx, vy = vh[0]
    if abs(vx) < 1e-9:
        return None
    slope = vy / vx
    return {"slope": float(slope), "intercept": float(ctr[1] - slope * ctr[0])}


def py_signed_angle_to_deep(slope: float, deep_slope: float) -> float:
    angle = np.degrees(np.arctan(slope) - np.arctan(deep_slope))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(angle)


def py_weighted_median(vals: list[float], wts: list[float]) -> float | None:
    if not vals:
        return None
    pairs = sorted(
        (float(v), float(w))
        for v, w in zip(vals, wts)
        if np.isfinite(v) and np.isfinite(w) and w > 0
    )
    if not pairs:
        return None
    total = sum(w for _v, w in pairs)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2.0:
            return value
    return pairs[-1][0]


def fragment_visible_length(xs: np.ndarray, ys: np.ndarray, slope: float) -> float:
    ux = 1.0 / np.sqrt(1.0 + slope * slope)
    uy = slope * ux
    proj = xs.astype(float) * ux + ys.astype(float) * uy
    return float(np.ptp(proj)) if len(proj) else 0.0


def line_segment_from_center(cx: float, cy: float, slope: float, length: float) -> dict[str, float]:
    half = max(18.0, min(float(length) / 2.0, 95.0))
    ux = 1.0 / np.sqrt(1.0 + slope * slope)
    uy = slope * ux
    return {
        "x1": float(cx - ux * half),
        "y1": float(cy - uy * half),
        "x2": float(cx + ux * half),
        "y2": float(cy + uy * half),
    }


def segment_intersection(a: dict[str, float], b: dict[str, float]) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    p1 = (float(a["x1"]), float(a["y1"]))
    p2 = (float(a["x2"]), float(a["y2"]))
    q1 = (float(b["x1"]), float(b["y1"]))
    q2 = (float(b["x2"]), float(b["y2"]))
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def project_center_slope(
    cx: float,
    cy: float,
    slope: float,
    boundary: dict,
    scale_px_per_mm: float | None,
) -> dict | None:
    line = {"slope": float(slope), "intercept": float(cy - slope * cx)}
    top = py_piecewise_intersection(line, boundary.get("top"), cx)
    deep = py_line_intersection(line, boundary.get("deep"))
    if top is None or deep is None:
        return None
    fl_px = float(np.hypot(top[0] - deep[0], top[1] - deep[1]))
    angle = abs(py_signed_angle_to_deep(slope, float(boundary["deep"]["slope"])))
    out = {
        "x1": top[0],
        "y1": top[1],
        "x2": deep[0],
        "y2": deep[1],
        "fl_px": fl_px,
        "angle_deg": angle,
    }
    if scale_px_per_mm and scale_px_per_mm > 0:
        out["fl_mm"] = fl_px / scale_px_per_mm
    return out


def angle_delta_rad(a: float, b: float) -> float:
    d = float(a - b)
    while d <= -np.pi / 2:
        d += np.pi
    while d > np.pi / 2:
        d -= np.pi
    return d


def angle_abs_delta_rad(a: float, b: float) -> float:
    return abs(angle_delta_rad(a, b))


@lru_cache(maxsize=128)
def wave_non_crossing_diagnostics_cached(
    image_id: str,
    boundary_text: str,
    scale_px_per_mm: float | None,
    max_degrees: float = 42.0,
    update_consensus: bool = False,
) -> str:
    boundary = json.loads(boundary_text) if boundary_text else None
    orientations = pa_orientation_diagnostics(image_id, boundary, deg_gate=999.0)
    if not orientations or boundary is None:
        return json.dumps({"order": "none", "items": [], "median_fl_mm": None, "median_pa_deg": None, "n_changed": 0}, default=json_numpy_default)
    raw_items = []
    for frag in orientations:
        raw_span = project_center_slope(frag["cx"], frag["cy"], frag["raw_slope"], boundary, scale_px_per_mm)
        if raw_span is None:
            continue
        raw_items.append({**frag, "raw_span": raw_span})
    if not raw_items:
        return json.dumps({"order": "none", "items": [], "median_fl_mm": None, "median_pa_deg": None, "n_changed": 0}, default=json_numpy_default)

    median_dx = float(np.median([item["raw_span"]["x1"] - item["raw_span"]["x2"] for item in raw_items]))
    left_to_right = median_dx >= 0.0
    wave_theta = py_weighted_median(
        [float(np.arctan(item["raw_slope"])) for item in raw_items],
        [item["area"] for item in raw_items],
    )
    if wave_theta is None:
        wave_theta = float(np.median([np.arctan(item["raw_slope"]) for item in raw_items]))

    out = []
    for item in raw_items:
        raw_theta = float(np.arctan(item["raw_slope"]))
        out.append({
            **item,
            "current_theta": raw_theta,
            "raw_crosses": 0,
            "corrected_slope": item["raw_slope"],
            "corrected_angle": item["raw_angle"],
            "corrected_span": item["raw_span"],
            "changed": False,
            "delta_angle_deg": 0.0,
            "cascade_steps": 0,
            "failed_to_resolve": False,
        })

    def crossings(items: list[dict]) -> list[tuple[int, int]]:
        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if segment_intersection(items[i]["corrected_span"], items[j]["corrected_span"]):
                    pairs.append((i, j))
        return pairs

    raw_pairs = crossings(out)
    raw_counts = [0 for _ in out]
    for i, j in raw_pairs:
        raw_counts[i] += 1
        raw_counts[j] += 1
    for idx, count in enumerate(raw_counts):
        out[idx]["raw_crosses"] = int(count)

    wave_slope = float(np.tan(wave_theta))
    seed_opposite_sign = 0
    for item in out:
        raw_theta = float(item["current_theta"])
        raw_dev = angle_abs_delta_rad(raw_theta, wave_theta)
        if (
            abs(wave_slope) > 0.03
            and float(item["raw_slope"]) * wave_slope < 0.0
            and raw_dev > np.radians(15.0)
        ):
            slope = wave_slope
            span = project_center_slope(item["cx"], item["cy"], slope, boundary, scale_px_per_mm)
            if span is None:
                continue
            item["current_theta"] = float(wave_theta)
            item["corrected_slope"] = float(slope)
            item["corrected_angle"] = abs(py_signed_angle_to_deep(float(slope), float(boundary["deep"]["slope"])))
            item["corrected_span"] = span
            item["changed"] = True
            item["delta_angle_deg"] = float(np.degrees(angle_delta_rad(wave_theta, raw_theta)))
            item["correction_reason"] = "opposite_consensus_sign_seed"
            seed_opposite_sign += 1

    max_iter = max(20, len(out) * 8)
    cascade_log = []
    for step in range(max_iter):
        if update_consensus:
            updated_wave = py_weighted_median(
                [float(item["current_theta"]) for item in out],
                [float(item.get("area", 1.0)) for item in out],
            )
            if updated_wave is not None:
                wave_theta = float(updated_wave)
        pairs = crossings(out)
        if not pairs:
            break
        offender_counts = {idx: 0 for idx in range(len(out))}
        offender_devs = {idx: angle_abs_delta_rad(out[idx]["current_theta"], wave_theta) for idx in range(len(out))}
        for i, j in pairs:
            offender = i if offender_devs[i] >= offender_devs[j] else j
            offender_counts[offender] += 1
        offender = max(
            offender_counts,
            key=lambda idx: (offender_counts[idx], offender_devs[idx], out[idx].get("area", 0.0)),
        )
        if offender_counts[offender] <= 0:
            break

        item = out[offender]
        start_theta = float(item["current_theta"])
        to_consensus = angle_delta_rad(wave_theta, start_theta)
        candidate_thetas = [start_theta]
        if abs(to_consensus) > np.radians(0.05):
            # Clamp to the consensus direction. This trial is allowed to rotate
            # a fragment toward the wave, but not past it.
            for frac in np.linspace(0.03, 1.0, 80):
                candidate_thetas.append(float(start_theta + to_consensus * frac))

        best = None
        best_count = 10**9
        best_dev = 10**9
        for theta in candidate_thetas:
            slope = float(np.tan(theta))
            span = project_center_slope(item["cx"], item["cy"], slope, boundary, scale_px_per_mm)
            if span is None:
                continue
            count = 0
            for other_idx, other in enumerate(out):
                if other_idx == offender:
                    continue
                if segment_intersection(span, other["corrected_span"]):
                    count += 1
            dev = angle_abs_delta_rad(theta, wave_theta)
            if (count, dev) < (best_count, best_dev):
                best = (theta, slope, span)
                best_count = count
                best_dev = dev
            if count == 0:
                break
        if best is None:
            out[offender]["failed_to_resolve"] = True
            break
        theta, slope, span = best
        # Stop if this pass cannot improve the offender's conflicts.
        if best_count >= offender_counts[offender] and abs(theta - start_theta) < np.radians(0.1):
            out[offender]["failed_to_resolve"] = True
            break
        out[offender]["current_theta"] = float(theta)
        out[offender]["corrected_slope"] = float(slope)
        out[offender]["corrected_angle"] = abs(py_signed_angle_to_deep(float(slope), float(boundary["deep"]["slope"])))
        out[offender]["corrected_span"] = span
        raw_theta = float(np.arctan(out[offender]["raw_slope"]))
        out[offender]["changed"] = angle_abs_delta_rad(theta, raw_theta) > np.radians(0.5)
        out[offender]["delta_angle_deg"] = float(np.degrees(angle_delta_rad(theta, raw_theta)))
        out[offender]["cascade_steps"] = int(out[offender].get("cascade_steps", 0) + 1)
        cascade_log.append({
            "step": step + 1,
            "index": offender,
            "frag_id": out[offender].get("frag_id"),
            "remaining_before": len(pairs),
            "offender_crossings_before": offender_counts[offender],
            "offender_crossings_after": best_count,
            "delta_angle_deg": out[offender]["delta_angle_deg"],
            "consensus_angle_deg": float(np.degrees(wave_theta)),
        })

    remaining_pairs = crossings(out)
    for item in out:
        raw_theta = float(np.arctan(item["raw_slope"]))
        current_theta = float(item.get("current_theta", raw_theta))
        item["raw_theta_deg"] = float(np.degrees(raw_theta))
        item["corrected_theta_deg"] = float(np.degrees(current_theta))
        item["raw_consensus_dev_deg"] = float(np.degrees(angle_abs_delta_rad(raw_theta, wave_theta)))
        item["corrected_consensus_dev_deg"] = float(np.degrees(angle_abs_delta_rad(current_theta, wave_theta)))
    out_sorted = sorted(out, key=lambda item: item["cx"], reverse=not left_to_right)
    fls = [item["corrected_span"].get("fl_mm") for item in out_sorted if item["corrected_span"].get("fl_mm") is not None]
    pas = [item["corrected_angle"] for item in out_sorted if item.get("corrected_angle") is not None]
    return json.dumps({
        "order": "left_to_right" if left_to_right else "right_to_left",
        "mode": "updating_consensus" if update_consensus else "fixed_consensus",
        "consensus_angle_deg": float(np.degrees(wave_theta)),
        "items": out_sorted,
        "median_fl_mm": float(np.median(fls)) if fls else None,
        "median_pa_deg": float(np.median(pas)) if pas else None,
        "n_changed": int(sum(1 for item in out if item.get("changed"))),
        "n_seed_opposite_sign": int(seed_opposite_sign),
        "n_raw_crossing": int(sum(1 for item in out if item.get("raw_crosses", 0) > 0)),
        "raw_crossing_pairs": len(raw_pairs),
        "remaining_crossing_pairs": len(remaining_pairs),
        "cascade_steps": len(cascade_log),
        "cascade_log": cascade_log[:120],
    }, default=json_numpy_default)


def wave_non_crossing_diagnostics(
    image_id: str,
    boundary: dict | None,
    scale_px_per_mm: float | None,
    max_degrees: float = 42.0,
    update_consensus: bool = False,
) -> dict:
    boundary_text = json.dumps(boundary, sort_keys=True) if boundary else ""
    return json.loads(wave_non_crossing_diagnostics_cached(image_id, boundary_text, scale_px_per_mm, max_degrees, update_consensus))


def segment_fraction_in_rect(span: dict | None, rect: dict | None) -> float | None:
    if not span or not rect:
        return None
    x0 = float(rect["x"])
    y0 = float(rect["y"])
    x1 = x0 + float(rect["w"])
    y1 = y0 + float(rect["h"])
    ax = float(span["x1"])
    ay = float(span["y1"])
    bx = float(span["x2"])
    by = float(span["y2"])
    dx = bx - ax
    dy = by - ay
    if float(np.hypot(dx, dy)) <= 1e-9:
        return 0.0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if abs(p) < 1e-12:
            if q < 0:
                return 0.0
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return 0.0
            t0 = max(t0, r)
        else:
            if r < t0:
                return 0.0
            t1 = min(t1, r)
    return float(np.clip(t1 - t0, 0.0, 1.0))


def add_support_fractions_to_wave_items(items: list[dict], row: dict) -> list[dict]:
    out = []
    for item in items:
        item = dict(item)
        span = item.get("corrected_span")
        raw_span = item.get("raw_span")
        fl_px = float((span or {}).get("fl_px") or 0.0)
        raw_fl_px = float((raw_span or {}).get("fl_px") or 0.0)
        visible_len = float(item.get("visible_len") or 0.0)
        visible_frac = float(np.clip(visible_len / fl_px, 0.0, 1.0)) if fl_px > 0 else None
        raw_visible_frac = float(np.clip(visible_len / raw_fl_px, 0.0, 1.0)) if raw_fl_px > 0 else None
        us_frac = segment_fraction_in_rect(span, row.get("us_field"))
        raw_us_frac = segment_fraction_in_rect(raw_span, row.get("us_field"))
        area = float(item.get("area") or 1.0)
        raw_crosses = float(item.get("raw_crosses") or 0.0)
        cross_down = 1.0 / (1.0 + raw_crosses)
        item["visible_frac"] = visible_frac
        item["raw_visible_frac"] = raw_visible_frac
        item["us_field_frac"] = us_frac
        item["raw_us_field_frac"] = raw_us_frac
        item["area_us_weight"] = area * (us_frac if us_frac is not None else 1.0)
        item["raw_area_us_weight"] = area * (raw_us_frac if raw_us_frac is not None else 1.0)
        item["area_us_visible_weight"] = area * (us_frac if us_frac is not None else 1.0) * (visible_frac if visible_frac is not None else 1.0)
        item["area_us_cross_weight"] = area * (us_frac if us_frac is not None else 1.0) * cross_down
        out.append(item)
    return out


def wave_projection_spans_from_items(items: list[dict]) -> list[dict]:
    return [
        {
            **item["corrected_span"],
            "component_id": item.get("component_id"),
            "frag_id": item.get("frag_id"),
            "raw_x1": item["raw_span"]["x1"],
            "raw_y1": item["raw_span"]["y1"],
            "raw_x2": item["raw_span"]["x2"],
            "raw_y2": item["raw_span"]["y2"],
            "changed": item.get("changed", False),
            "raw_crosses": item.get("raw_crosses", 0),
            "raw_theta_deg": item.get("raw_theta_deg"),
            "corrected_theta_deg": item.get("corrected_theta_deg"),
            "raw_consensus_dev_deg": item.get("raw_consensus_dev_deg"),
            "corrected_consensus_dev_deg": item.get("corrected_consensus_dev_deg"),
            "delta_angle_deg": item.get("delta_angle_deg", 0.0),
            "cascade_steps": item.get("cascade_steps", 0),
            "failed_to_resolve": item.get("failed_to_resolve", False),
            "correction_reason": item.get("correction_reason"),
            "raw_fl_mm": item.get("raw_span", {}).get("fl_mm"),
            "raw_angle_deg": item.get("raw_angle"),
            "corrected_angle_deg": item.get("corrected_angle"),
            "visible_frac": item.get("visible_frac"),
            "raw_visible_frac": item.get("raw_visible_frac"),
            "us_field_frac": item.get("us_field_frac"),
            "raw_us_field_frac": item.get("raw_us_field_frac"),
            "area_us_weight": item.get("area_us_weight"),
            "raw_area_us_weight": item.get("raw_area_us_weight"),
            "area_us_visible_weight": item.get("area_us_visible_weight"),
            "area_us_cross_weight": item.get("area_us_cross_weight"),
        }
        for item in items
        if item.get("corrected_span") is not None
    ]


@lru_cache(maxsize=256)
def pa_orientation_diagnostics_cached(image_id: str, boundary_text: str, deg_gate: float = 7.0, width: float = 180.0) -> tuple[dict, ...]:
    boundary = json.loads(boundary_text) if boundary_text else None
    if cv2 is None or np is None or boundary is None or boundary.get("deep") is None:
        return tuple()
    fasc = load_mask(ROOT / "results" / "visual_review" / f"{image_id}_fasc.png")
    if fasc is None:
        return []
    deep_slope = float(boundary["deep"]["slope"])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc, np.uint8), 8)
    raw = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        line = py_pca_line(xs, ys)
        if line is None:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        slope = float(line["slope"])
        angle = abs(py_signed_angle_to_deep(slope, deep_slope))
        if not (6.0 <= angle <= 75.0):
            continue
        visible = fragment_visible_length(xs, ys, slope)
        raw.append({
            "component_id": int(i),
            "frag_id": f"F{i}",
            "cx": cx,
            "cy": cy,
            "area": float(area),
            "raw_slope": slope,
            "raw_angle": angle,
            "visible_len": visible,
        })
    out = []
    for frag in raw:
        neigh = [
            other for other in raw
            if 0 < abs(other["cx"] - frag["cx"]) <= width
        ]
        local_theta = None
        if len(neigh) >= 2:
            local_theta = py_weighted_median(
                [float(np.arctan(other["raw_slope"])) for other in neigh],
                [other["area"] for other in neigh],
            )
        raw_theta = float(np.arctan(frag["raw_slope"]))
        corrected_theta = raw_theta
        changed = False
        if local_theta is not None and abs(np.degrees(raw_theta - local_theta)) >= deg_gate:
            corrected_theta = float(local_theta)
            changed = True
        corrected_slope = float(np.tan(corrected_theta))
        row = {
            **frag,
            "local_slope": None if local_theta is None else float(np.tan(local_theta)),
            "corrected_slope": corrected_slope,
            "corrected_angle": abs(py_signed_angle_to_deep(corrected_slope, deep_slope)),
            "changed": changed,
            "raw_segment": line_segment_from_center(frag["cx"], frag["cy"], frag["raw_slope"], frag["visible_len"]),
            "corrected_segment": line_segment_from_center(frag["cx"], frag["cy"], corrected_slope, frag["visible_len"]),
        }
        out.append(row)
    return tuple(out)


def pa_orientation_diagnostics(image_id: str, boundary: dict | None, deg_gate: float = 7.0, width: float = 180.0) -> list[dict]:
    boundary_text = json.dumps(boundary, sort_keys=True) if boundary else ""
    return list(pa_orientation_diagnostics_cached(image_id, boundary_text, float(deg_gate), float(width)))


def robust_projection_spans(fasc_path: Path, boundary: dict | None, scale_px_per_mm: float | None) -> list[dict[str, float]]:
    fasc = load_mask(fasc_path)
    if fasc is None or boundary is None or boundary.get("top") is None or boundary.get("deep") is None:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc, np.uint8), 8)
    spans = []
    deep = boundary["deep"]
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        line = py_pca_line(xs, ys)
        if line is None:
            continue
        cx = float(np.mean(xs))
        upper = py_piecewise_intersection(line, boundary["top"], cx)
        lower = py_line_intersection(line, deep)
        if upper is None or lower is None:
            continue
        fl_px = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
        angle = abs(py_signed_angle_to_deep(line["slope"], deep["slope"]))
        if not (10.0 <= fl_px <= 4000.0 and 6.0 <= angle <= 75.0):
            continue
        row = {
            "component_id": int(i),
            "frag_id": f"F{i}",
            "x1": upper[0],
            "y1": upper[1],
            "x2": lower[0],
            "y2": lower[1],
            "angle_deg": float(angle),
            "fl_px": fl_px,
            "area": float(area),
        }
        if scale_px_per_mm and scale_px_per_mm > 0:
            row["fl_mm"] = fl_px / scale_px_per_mm
        spans.append(row)
    return spans


def candidate_boundary_from_apo_mask(path: Path) -> dict | None:
    apo = load_mask(path)
    if apo is None:
        return None
    groups = candidate_apo_boundary_groups(np.ascontiguousarray(apo, np.uint8))
    if groups is None:
        return None
    groups.sort(key=lambda item: item[0])
    (_, sup_xs, sup_ys), (_, deep_xs, deep_ys) = groups[:2]

    sup_ux, sup_inv = np.unique(sup_xs, return_inverse=True)
    deep_ux, deep_inv = np.unique(deep_xs, return_inverse=True)
    if len(sup_ux) < 2 or len(deep_ux) < 2:
        return None

    # Upper boundary uses the muscle-facing/lower edge of the superficial band.
    sup_edge_y = np.full(len(sup_ux), -1.0)
    np.maximum.at(sup_edge_y, sup_inv, sup_ys.astype(float))
    # Lower boundary uses the muscle-facing/upper edge of the deep band.
    deep_edge_y = np.full(len(deep_ux), 1e18)
    np.minimum.at(deep_edge_y, deep_inv, deep_ys.astype(float))

    deep_slope, deep_intercept = fit_line(deep_ux.astype(float), deep_edge_y)
    points = robust_triangle_points(sup_ux.astype(float), sup_edge_y.astype(float))
    if points is None:
        sup_slope, sup_intercept = fit_line(sup_ux.astype(float), sup_edge_y)
        top = {"type": "line", "slope": sup_slope, "intercept": sup_intercept}
    else:
        top = {"type": "piecewise", "points": points}
    return {
        "mode": "robust_triangle",
        "top": top,
        "deep": {"type": "line", "slope": deep_slope, "intercept": deep_intercept},
    }


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
    ]
    return [(name, path) for name, path in candidates if path.exists()]


def default_expert_candidate_csvs() -> list[tuple[str, Path]]:
    exp48 = ROOT / "results" / "exp48_geometry_class_feature_matrix"
    exp49 = ROOT / "results" / "exp49_weighted_story_aggregators"
    exp50 = ROOT / "results" / "exp50_story_weight_grid"
    exp53 = ROOT / "results" / "exp53_median_weight_blends"
    candidates = [
        ("robust_triangle_anchor", ROOT / "results" / "benchmark_pred_robust_triangle.csv"),
        ("story_stack", exp48 / "story_FL_scan_on_low_support_PA_per_band_on_multi_band_else_conflict_all_MT_vertical_all.csv"),
        ("weighted_story_fl", exp49 / "best_combo.csv"),
        ("story_weight_grid_best", exp50 / "best_combo.csv"),
        ("story_weight_same_story", exp50 / "best_same_story.csv"),
        ("median_weight_blend_best", exp53 / "best_combo.csv"),
        ("fl_scan_support", ROOT / "results" / "exp40_untested_feature_benchmark" / "strict_scan_region_linear_support_weighted_FL_only.csv"),
        ("pa_per_band", ROOT / "results" / "exp42_per_band_pa_mt_isolation" / "fragment_count_average_all_detected_bands_pa_only_keep_FL_baseline.csv"),
        ("pa_conflict_gate", ROOT / "results" / "exp39_pa_lower_boundary_ablation" / "pa_conflict_gated_7deg.csv"),
        ("mt_vertical_center", ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "MT_only_vertical_center_gap_keep_PA_FL.csv"),
        ("best_local_all_features", ROOT / "results" / "exp44_best_local_feature_stack" / "FL_scan_region_linear_plus_PA_conflict_gate_plus_MT_vertical_center.csv"),
    ]
    return [(name, path) for name, path in candidates if path.exists()]


def rejected_candidate_csvs() -> list[tuple[str, Path]]:
    candidates = [
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


def dedupe_candidate_csvs(candidates: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    out = []
    for name, path in candidates:
        key = safe_id(name)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, path))
    return out


def read_exp48_classes() -> dict[str, dict[str, bool]]:
    path = ROOT / "results" / "exp48_geometry_class_feature_matrix" / "geometry_classes.csv"
    out: dict[str, dict[str, bool]] = {}
    for row in read_csv(path):
        image_id = stem(row.get("image_id", ""))
        out[image_id] = {
            key: str(value).strip().lower() == "true"
            for key, value in row.items()
            if key != "image_id"
        }
    return out


def read_exp48_table(name: str) -> list[dict[str, str]]:
    return read_csv(ROOT / "results" / "exp48_geometry_class_feature_matrix" / name)


@lru_cache(maxsize=8)
def read_json_file(path_text: str) -> dict:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def json_numpy_default(value):
    if np is not None and isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def exp40_geometry(variant: str, image_id: str) -> dict:
    bundle = read_json_file(str(ROOT / "results" / "exp40_untested_feature_benchmark" / "geometry_bundle.json"))
    return bundle.get(variant, {}).get(image_id, {})


def exp42_geometry(image_id: str) -> dict:
    bundle = read_json_file(str(ROOT / "results" / "exp42_per_band_pa_mt_isolation" / "geometry_bundle.json"))
    return bundle.get(image_id, {})


def exp43_geometry(variant: str, image_id: str) -> dict:
    bundle = read_json_file(str(ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "geometry_bundle.json"))
    return bundle.get(variant, {}).get(image_id, {})


def exp39_geometry(image_id: str) -> dict:
    bundle = read_json_file(str(ROOT / "results" / "exp39_pa_lower_boundary_ablation" / "geometry_bundle.json"))
    return bundle.get(image_id, {})


def boundary_from_points(points: list[dict[str, float]] | None) -> dict | None:
    if not points:
        return None
    return {"type": "piecewise", "points": points}


def normalize_span(span: dict) -> dict:
    out = {
        "x1": span.get("x1"),
        "y1": span.get("y1"),
        "x2": span.get("x2"),
        "y2": span.get("y2"),
    }
    for key in ("fl_mm", "angle_deg", "visible_frac", "strict_scan_region_frac", "gap_index", "frag_id", "component_id"):
        if key in span:
            out[key] = span[key]
    return out


def attach_span_ids(spans: list[dict], reference_spans: list[dict]) -> list[dict]:
    if not spans or not reference_spans:
        return spans
    refs = [ref for ref in reference_spans if ref.get("frag_id")]
    if not refs:
        return spans
    used: set[int] = set()
    for idx, span in enumerate(spans):
        if span.get("frag_id"):
            continue
        best_idx = None
        best_score = float("inf")
        for ref_idx, ref in enumerate(refs):
            if ref_idx in used:
                continue
            try:
                endpoint_dist = (
                    abs(float(span.get("x1", 0)) - float(ref.get("x1", 0)))
                    + abs(float(span.get("y1", 0)) - float(ref.get("y1", 0)))
                    + abs(float(span.get("x2", 0)) - float(ref.get("x2", 0)))
                    + abs(float(span.get("y2", 0)) - float(ref.get("y2", 0)))
                )
                angle_dist = abs(float(span.get("angle_deg", 0)) - float(ref.get("angle_deg", 0))) * 8.0
                fl_dist = abs(float(span.get("fl_mm", 0)) - float(ref.get("fl_mm", 0))) * 2.0
            except (TypeError, ValueError):
                continue
            score = endpoint_dist + angle_dist + fl_dist
            if score < best_score:
                best_score = score
                best_idx = ref_idx
        if best_idx is not None and best_score < 160.0:
            ref = refs[best_idx]
            used.add(best_idx)
            span["frag_id"] = ref.get("frag_id")
            span["component_id"] = ref.get("component_id")
            span["frag_match"] = "geometry_nearest"
        elif idx < len(refs):
            ref = refs[idx]
            span["frag_id"] = ref.get("frag_id")
            span["component_id"] = ref.get("component_id")
            span["frag_match"] = "order_fallback"
    return spans


def detect_scan_region(image_id: str, image_path_text: str) -> dict | None:
    if cv2 is None or np is None:
        return None
    preferred = ROOT / "results" / "visual_review" / f"{image_id}_base.jpg"
    path = preferred if preferred.exists() else Path(image_path_text)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    nonblack = (gray > 8).astype(np.uint8)
    if int(nonblack.sum()) == 0:
        return None
    closed = cv2.morphologyEx(nonblack, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        ys, xs = np.where(nonblack > 0)
        if len(xs) == 0:
            return None
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        area = int(nonblack.sum())
    else:
        keep = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        x = int(stats[keep, cv2.CC_STAT_LEFT])
        y = int(stats[keep, cv2.CC_STAT_TOP])
        w = int(stats[keep, cv2.CC_STAT_WIDTH])
        h = int(stats[keep, cv2.CC_STAT_HEIGHT])
        area = int(stats[keep, cv2.CC_STAT_AREA])
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "image_w": int(gray.shape[1]),
        "image_h": int(gray.shape[0]),
        "method": "largest_nonblack_connected_component",
    }


def longest_true_run(mask: np.ndarray, min_len: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, ok in enumerate(mask.tolist() + [False]):
        if ok and start is None:
            start = idx
        if not ok and start is not None:
            end = idx
            if end - start >= min_len and (best is None or end - start > best[1] - best[0]):
                best = (start, end)
            start = None
    return best


def detect_image_field(image_id: str, image_path_text: str) -> dict | None:
    if cv2 is None or np is None:
        return None
    preferred = ROOT / "results" / "visual_review" / f"{image_id}_base.jpg"
    path = preferred if preferred.exists() else Path(image_path_text)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    # Broad field estimate: find long rows/columns that are not pure black.
    # This is intentionally not the dense texture core; it is the visible image
    # field boundary used for on-screen/off-screen projection reasoning.
    active = (gray > 8).astype(np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1).astype(bool)
    row_density = active.mean(axis=1)
    col_density = active.mean(axis=0)
    row_density = np.convolve(row_density, np.ones(21) / 21.0, mode="same")
    col_density = np.convolve(col_density, np.ones(21) / 21.0, mode="same")
    rows = longest_true_run(row_density > 0.05, max(20, gray.shape[0] // 10))
    cols = longest_true_run(col_density > 0.05, max(40, gray.shape[1] // 10))
    if rows is None or cols is None:
        return None
    y0, y1 = rows
    x0, x1 = cols
    if (x1 - x0) * (y1 - y0) < gray.shape[0] * gray.shape[1] * 0.08:
        return None
    return {
        "x": int(x0),
        "y": int(y0),
        "w": int(x1 - x0),
        "h": int(y1 - y0),
        "image_w": int(gray.shape[1]),
        "image_h": int(gray.shape[0]),
        "method": "broad_nonblack_field",
    }


def mask_anchor_for_field(image_id: str) -> tuple[int, int] | None:
    masks = []
    for suffix in ("fasc", "apo"):
        mask = load_mask(ROOT / "results" / "visual_review" / f"{image_id}_{suffix}.png")
        if mask is not None and int(mask.sum()) > 0:
            masks.append(mask.astype(bool))
    if not masks:
        return None
    merged = np.logical_or.reduce(masks)
    ys, xs = np.where(merged)
    if len(xs) == 0:
        return None
    return int(np.median(xs)), int(np.median(ys))


def uniform_vertical(gray: np.ndarray, x: int, y: int, run: int, tol: int) -> bool:
    y0 = max(0, y - run)
    y1 = min(gray.shape[0], y + run + 1)
    vals = gray[y0:y1, x]
    return len(vals) >= run and int(vals.max()) - int(vals.min()) <= tol


def uniform_horizontal(gray: np.ndarray, x: int, y: int, run: int, tol: int) -> bool:
    x0 = max(0, x - run)
    x1 = min(gray.shape[1], x + run + 1)
    vals = gray[y, x0:x1]
    return len(vals) >= run and int(vals.max()) - int(vals.min()) <= tol


def detect_ultrasound_field(image_id: str, image_path_text: str) -> dict | None:
    if cv2 is None or np is None:
        return None
    preferred = ROOT / "results" / "visual_review" / f"{image_id}_base.jpg"
    path = preferred if preferred.exists() else Path(image_path_text)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    anchor = mask_anchor_for_field(image_id)
    if anchor is None:
        return None
    cx, cy = anchor
    masks = []
    for suffix in ("fasc", "apo"):
        mask = load_mask(ROOT / "results" / "visual_review" / f"{image_id}_{suffix}.png")
        if mask is not None and int(mask.sum()) > 0:
            masks.append(mask.astype(bool))
    if not masks:
        return None
    merged = np.logical_or.reduce(masks)
    ys, xs = np.where(merged)
    if len(xs) == 0:
        return None
    mx0, mx1 = int(xs.min()), int(xs.max())
    my0, my1 = int(ys.min()), int(ys.max())
    run = max(10, min(gray.shape[:2]) // 40)
    tol = 4
    y_samples = [int(v) for v in np.linspace(max(0, my0 - 30), min(gray.shape[0] - 1, my1 + 30), 13)]
    x_samples = [int(v) for v in np.linspace(max(0, mx0 - 45), min(gray.shape[1] - 1, mx1 + 45), 15)]

    def vertical_background_vote(x: int) -> float:
        votes = [uniform_vertical(gray, x, y, run, tol) for y in y_samples]
        return float(sum(votes) / max(len(votes), 1))

    def horizontal_background_vote(y: int) -> float:
        votes = [uniform_horizontal(gray, x, y, run, tol) for x in x_samples]
        return float(sum(votes) / max(len(votes), 1))

    def sustained_edge(values: list[tuple[int, float]], reverse: bool) -> int | None:
        run = 0
        best = None
        ordered = values[::-1] if reverse else values
        for pos, vote in ordered:
            if vote >= 0.62:
                run += 1
                if run >= 4:
                    best = pos
                    break
            else:
                run = 0
        return best

    left_values = [(x, vertical_background_vote(x)) for x in range(max(0, mx0 - 320), max(1, mx0 + 1))]
    right_values = [(x, vertical_background_vote(x)) for x in range(min(gray.shape[1] - 1, mx1), min(gray.shape[1], mx1 + 321))]
    top_values = [(y, horizontal_background_vote(y)) for y in range(max(0, my0 - 260), max(1, my0 + 1))]
    bottom_values = [(y, horizontal_background_vote(y)) for y in range(min(gray.shape[0] - 1, my1), min(gray.shape[0], my1 + 261))]
    left_bg = sustained_edge(left_values, reverse=True)
    right_bg = sustained_edge(right_values, reverse=False)
    top_bg = sustained_edge(top_values, reverse=True)
    bottom_bg = sustained_edge(bottom_values, reverse=False)
    left = 0 if left_bg is None else min(gray.shape[1] - 1, left_bg + 3)
    right = gray.shape[1] - 1 if right_bg is None else max(0, right_bg - 3)
    top = 0 if top_bg is None else min(gray.shape[0] - 1, top_bg + 3)
    bottom = gray.shape[0] - 1 if bottom_bg is None else max(0, bottom_bg - 3)
    if right <= left or bottom <= top:
        return None
    return {
        "x": left,
        "y": top,
        "w": right - left + 1,
        "h": bottom - top + 1,
        "anchor_x": cx,
        "anchor_y": cy,
        "mask_box": {"x": mx0, "y": my0, "w": mx1 - mx0 + 1, "h": my1 - my0 + 1},
        "uniform_run": run,
        "uniform_tol": tol,
        "vote_threshold": 0.62,
        "image_w": int(gray.shape[1]),
        "image_h": int(gray.shape[0]),
        "method": "mask_bounds_uniform_run_vote",
    }


def model_display(name: str) -> dict[str, str]:
    key = safe_id(name)
    info = MODEL_INFO.get(key, {})
    return {
        "id": key,
        "label": info.get("label", name.replace("_", " ")),
        "color": info.get("color", "#f4f4f5"),
        "kind": info.get("kind", "candidate"),
        "description": info.get("description", ""),
    }


def story_gate_reason(model_id: str, flags: dict[str, bool]) -> str:
    if model_id == "story_stack":
        pa = "per-band PA" if flags.get("multi_band_risk") else "conflict-gated PA"
        fl = "scan-support FL" if flags.get("low_support_any") else "robust-triangle FL"
        return f"{fl}; {pa}; vertical-center MT."
    if model_id == "weighted_story_fl":
        return "EXP49 benchmark: PA median, FL weighted by area x ultrasound-field support, vertical-center MT."
    if model_id == "story_weight_grid_best":
        return "EXP50 benchmark best: PA median; FL raw-span weighted trim using area x ultrasound-field support x local trajectory residual; vertical-center MT."
    if model_id == "story_weight_same_story":
        return "EXP50 coherent reducer: PA and FL both use area x ultrasound-field support x local trajectory residual; vertical-center MT."
    if model_id == "median_weight_blend_best":
        return "EXP53 benchmark best: PA median blended 25% toward saturating support/position PA; FL median blended 85% toward weighted-trim support FL; vertical-center MT."
    if model_id == "fl_scan_support":
        return "Component check: FL is weighted by scan-region support."
    if model_id == "pa_per_band":
        return "Component check: PA is averaged within bands before global aggregation."
    if model_id == "pa_conflict_gate":
        return "Component check: PA uses conflict-gated orientation selection."
    if model_id == "mt_vertical_center":
        return "Component check: MT uses vertical center gap."
    if model_id == "robust_triangle_anchor":
        return "Anchor model: robust upper triangle plus linear lower boundary."
    if model_id == "best_local_all_features":
        return "Prior best local stack before class-aware EXP48 gating."
    return ""


def model_from_candidate(
    cand: dict,
    boundary: dict | None,
    spans: list[dict[str, float]],
    flags: dict[str, bool],
    row: dict,
) -> dict:
    display = model_display(cand.get("name", "candidate"))
    model_id = display["id"]
    has_geometry = model_id in GEOMETRY_MODEL_IDS and boundary is not None
    model_boundary = boundary if has_geometry else None
    model_spans = spans if has_geometry else []
    diagnostics: dict = {}
    geometry_note = "No model-specific geometry bundle; numeric candidate only."

    if model_id == "robust_triangle_anchor" and model_boundary is not None:
        diagnostics["pa_orientation"] = pa_orientation_diagnostics(row.get("image_id", ""), boundary, deg_gate=999.0)
        geometry_note = "Model-specific robust-triangle boundary and projected spans."
    if model_id == "our_pipeline_true_scale" and model_boundary is not None:
        diagnostics["pa_orientation"] = pa_orientation_diagnostics(row.get("image_id", ""), boundary, deg_gate=999.0)
        geometry_note = "Numeric true-scale pipeline CSV; spans shown are reconstructed from current masks for visual reference, not an archived exact production geometry bundle."
    if model_id in {"fl_scan_support", "best_local_all_features"}:
        g = exp40_geometry("strict_scan_region_linear_support_weighted_FL_only", row.get("image_id", ""))
        if g:
            model_boundary = {
                "mode": "exp40_scan_support",
                "top": boundary_from_points(g.get("upper", {}).get("points")) or (boundary or {}).get("top"),
                "deep": (boundary or {}).get("deep"),
            }
            model_spans = attach_span_ids([normalize_span(s) for s in g.get("spans", [])], row.get("candidate_projection_spans") or [])
            diagnostics.update({k: g.get(k) for k in ("n_fragments", "n_projected", "median_visible_frac", "median_strict_scan_region_frac")})
            geometry_note = "Model-specific exp40 scan-support spans; lower line remains the robust deep line."
    if model_id == "story_stack":
        diagnostics["pa_orientation"] = pa_orientation_diagnostics(row.get("image_id", ""), boundary, deg_gate=(999.0 if flags.get("multi_band_risk") else 7.0))
        if flags.get("low_support_any"):
            g = exp40_geometry("strict_scan_region_linear_support_weighted_FL_only", row.get("image_id", ""))
            if g:
                model_boundary = {
                    "mode": "story_low_support_scan",
                    "top": boundary_from_points(g.get("upper", {}).get("points")) or (boundary or {}).get("top"),
                    "deep": (boundary or {}).get("deep"),
                }
                model_spans = attach_span_ids([normalize_span(s) for s in g.get("spans", [])], row.get("candidate_projection_spans") or [])
                diagnostics.update({k: g.get(k) for k in ("n_fragments", "n_projected", "median_visible_frac", "median_strict_scan_region_frac")})
                geometry_note = "Story stack used scan-support FL for this low-support image."
        elif model_boundary is not None:
            geometry_note = "Story stack uses robust FL geometry for this image."
    if model_id in {"weighted_story_fl", "story_weight_grid_best", "story_weight_same_story", "median_weight_blend_best"} and boundary is not None:
        diag = wave_non_crossing_diagnostics(
            row.get("image_id", ""),
            boundary,
            row.get("scale_px_per_mm"),
        )
        diag["items"] = add_support_fractions_to_wave_items(diag.get("items") or [], row)
        diagnostics["wave_non_crossing"] = diag
        diagnostics["pa_orientation"] = [
            {
                **item,
                "corrected_segment": item.get("corrected_segment") or line_segment_from_center(
                    item["cx"],
                    item["cy"],
                    item.get("corrected_slope", item["raw_slope"]),
                    item.get("visible_len", 60.0),
                ),
            }
            for item in diag.get("items") or []
        ]
        model_boundary = boundary
        model_spans = wave_projection_spans_from_items(diag.get("items") or [])
        geometry_note = "Weighted story stack: wave spans displayed with ultrasound-field support weights."
    if model_id == "pa_per_band":
        diagnostics["pa_orientation"] = pa_orientation_diagnostics(row.get("image_id", ""), boundary, deg_gate=999.0)
        g = exp42_geometry(row.get("image_id", ""))
        band_spans = []
        for gap in g.get("gaps", []):
            for span in gap.get("spans", []):
                band_spans.append(normalize_span({**span, "gap_index": gap.get("gap_index")}))
        if band_spans:
            model_boundary = boundary
            model_spans = attach_span_ids(band_spans, row.get("candidate_projection_spans") or [])
            diagnostics.update({"n_bands": g.get("n_bands"), "n_gaps": g.get("n_gaps")})
            geometry_note = "Model-specific exp42 per-band grouped projected spans."
    if model_id == "pa_conflict_gate":
        diagnostics["pa_orientation"] = pa_orientation_diagnostics(row.get("image_id", ""), boundary, deg_gate=7.0)
        g = exp39_geometry(row.get("image_id", ""))
        if g:
            model_boundary = boundary
            diagnostics.update({
                "n_fragments": g.get("n_fragments"),
                "lower_smooth": g.get("lower_smooth"),
                "lower_quartile": g.get("lower_quartile"),
            })
            geometry_note = "PA conflict is numeric; lower-boundary diagnostic curves are available."
    if model_id in {"mt_vertical_center", "best_local_all_features", "story_stack"}:
        mt = exp43_geometry("MT_only_vertical_center_gap_keep_PA_FL", row.get("image_id", ""))
        if mt.get("mt"):
            model_boundary = model_boundary or boundary
            diagnostics["mt_positions"] = mt["mt"].get("positions", [])
            diagnostics["mt_values"] = mt["mt"].get("mt_values", [])
            if model_id == "mt_vertical_center":
                geometry_note = "Model-specific exp43 vertical-center MT position."

    return {
        **display,
        "predictions": {
            "pa_deg": cand.get("pa_deg"),
            "fl_mm": cand.get("fl_mm"),
            "mt_mm": cand.get("mt_mm"),
        },
        "deltas": {
            "pa_deg": cand.get("delta_pa"),
            "fl_mm": cand.get("delta_fl"),
            "mt_mm": cand.get("delta_mt"),
        },
        "overall_norm": cand.get("overall_norm"),
        "gate_reason": story_gate_reason(model_id, flags),
        "geometry_note": geometry_note,
        "diagnostics": diagnostics,
        "boundary": model_boundary,
        "projection_spans": model_spans,
    }


def wave_trial_model(
    row: dict,
    boundary: dict | None,
    flags: dict[str, bool],
    model_id: str = "wave_non_crossing_trial",
    update_consensus: bool = False,
) -> dict | None:
    if boundary is None:
        return None
    display = model_display(model_id)
    diag = wave_non_crossing_diagnostics(
        row.get("image_id", ""),
        boundary,
        row.get("scale_px_per_mm"),
        update_consensus=update_consensus,
    )
    items = diag.get("items") or []
    items = add_support_fractions_to_wave_items(items, row)
    diag["items"] = items
    spans = wave_projection_spans_from_items(items)
    pred = {
        "pa_deg": diag.get("median_pa_deg"),
        "fl_mm": diag.get("median_fl_mm"),
        "mt_mm": None,
    }
    deltas, overall = norm_delta(row.get("human", {}), pred)
    return {
        **display,
        "predictions": pred,
        "deltas": {
            "pa_deg": deltas.get("delta_pa"),
            "fl_mm": deltas.get("delta_fl"),
            "mt_mm": deltas.get("delta_mt"),
        },
        "overall_norm": overall,
        "summary": {},
        "gate_reason": (
            f"Visual trial only ({diag.get('mode')}). Order {diag.get('order')}; "
            f"changed {diag.get('n_changed')} of {len(items)} fragments; "
            f"raw crossing fragments {diag.get('n_raw_crossing')}."
        ),
        "geometry_note": "Rotate-only non-crossing projected-span trial, clamped at consensus; not yet a production candidate.",
        "diagnostics": {
            "wave_non_crossing": diag,
            "pa_orientation": [
                {
                    **item,
                    "corrected_segment": item.get("corrected_segment") or line_segment_from_center(
                        item["cx"],
                        item["cy"],
                        item.get("corrected_slope", item["raw_slope"]),
                        item.get("visible_len", 60.0),
                    ),
                }
                for item in items
            ],
        },
        "boundary": boundary,
        "projection_spans": spans,
    }


def enrich_rows_for_v2(rows: list[dict], summary: list[dict]) -> dict:
    class_map = read_exp48_classes()
    summary_by_name = {safe_id(row.get("name", "")): row for row in summary}
    for row in rows:
        flags = class_map.get(row.get("image_id", ""), {})
        row["class_flags"] = flags
        row["classes"] = [name for name, enabled in flags.items() if enabled]
        row["scan_region"] = detect_scan_region(row.get("image_id", ""), row.get("image_path", ""))
        row["image_field"] = detect_image_field(row.get("image_id", ""), row.get("image_path", ""))
        row["us_field"] = detect_ultrasound_field(row.get("image_id", ""), row.get("image_path", ""))
        models = [
            model_from_candidate(
                cand,
                row.get("candidate_boundary"),
                row.get("candidate_projection_spans") or [],
                flags,
                row,
            )
            for cand in row.get("candidates", [])
        ]
        wave = wave_trial_model(row, row.get("candidate_boundary"), flags)
        wave_updating = wave_trial_model(
            row,
            row.get("candidate_boundary"),
            flags,
            model_id="wave_non_crossing_updating_trial",
            update_consensus=True,
        )
        wave_models = [model for model in (wave, wave_updating) if model is not None]
        if wave_models:
            insert_at = next((idx + 1 for idx, model in enumerate(models) if model["id"] == "pa_conflict_gate"), len(models))
            for offset, model in enumerate(wave_models):
                models.insert(insert_at + offset, model)
        row["models"] = models
        for model in row["models"]:
            model["summary"] = summary_by_name.get(model["id"], {})
    computed: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        for model in row.get("models", []):
            acc = computed.setdefault(model["id"], {"overall": [], "pa": [], "fl": [], "mt": []})
            if model.get("overall_norm") is not None:
                acc["overall"].append(float(model["overall_norm"]))
            deltas = model.get("deltas") or {}
            if deltas.get("pa_deg") is not None:
                acc["pa"].append(abs(float(deltas["pa_deg"])) / TOL["pa_deg"])
            if deltas.get("fl_mm") is not None:
                acc["fl"].append(abs(float(deltas["fl_mm"])) / TOL["fl_mm"])
            if deltas.get("mt_mm") is not None:
                acc["mt"].append(abs(float(deltas["mt_mm"])) / TOL["mt_mm"])

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    for model_id, acc in computed.items():
        summary_by_name[model_id] = {
            **summary_by_name.get(model_id, {}),
            "name": model_id,
            "overall_norm": mean(acc["overall"]),
            "pa_norm": mean(acc["pa"]),
            "fl_norm": mean(acc["fl"]),
            "mt_norm": mean(acc["mt"]),
            "n": len(acc["overall"]),
        }
    for row in rows:
        for model in row.get("models", []):
            model["summary"] = summary_by_name.get(model["id"], {})
    return {
        "tolerances": TOL,
        "model_info": MODEL_INFO,
        "story_summary": read_exp48_table("story_candidate_summary.csv"),
        "class_feature_scores": read_exp48_table("class_feature_scores.csv"),
        "class_names": list(next(iter(class_map.values())).keys()) if class_map else [],
    }


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


def robust_expert_human(row: dict[str, str]) -> tuple[dict[str, float | None], list[str]]:
    out: dict[str, float | None] = {}
    notes: list[str] = []
    for col, suffix in (("pa_deg", "PA"), ("fl_mm", "FL"), ("mt_mm", "MT")):
        value, dropped = robust_mean_for_row(row, suffix)
        out[col] = value
        if dropped is not None:
            notes.append(
                f"{suffix}: dropped {dropped.rater}={dropped.value:.2f} "
                f"(raw mean {dropped.raw_mean:.2f} -> {dropped.robust_mean:.2f})"
            )
    return out, notes


def summarize_candidates(summary_acc: dict[str, dict[str, list[float]]]) -> list[dict]:
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
    summary.sort(key=lambda s: 999 if s["overall_norm"] is None else s["overall_norm"])
    return summary


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
        cal_row = cal.get(image_id, {})
        scale = parse_float(cal_row.get("px_per_mm"))
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
            "scale_px_per_mm": scale,
            "scale_px_per_cm": None if scale is None else scale * 10.0,
            "calibration_method": cal_row.get("method", "") or cal_row.get("calibration_method", ""),
            "calibration_confidence": parse_float(cal_row.get("confidence") or cal_row.get("calibration_confidence")),
            "apo_lines": apo_line_fits(labels_dir, label_id),
            "label_available": True,
            "truth_label": "human mask",
            "truth_note": "Target-image human mask measurement; useful for triage, not official ground truth.",
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


def find_expert_benchmark_dir() -> Path:
    hits = list((ROOT / "data").glob("**/Results_benchmark_architecture*.xlsx"))
    if not hits:
        raise FileNotFoundError("Results_benchmark_architecture*.xlsx not found under data/")
    return hits[0].parent


def benchmark_candidate_data(truth_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, dict[str, str]]]]:
    out: list[tuple[str, dict[str, dict[str, str]]]] = []
    pred_path = ROOT / "results" / "benchmark_pred_truescale.csv"
    if pred_path.exists():
        out.append(("our_pipeline_true_scale", read_keyed(pred_path)))
    for name, prefix in (("DLTrack", "DLTrack"), ("SMA", "SMA")):
        data: dict[str, dict[str, str]] = {}
        for row in truth_rows:
            image_id = stem(row.get("ImageID", ""))
            data[image_id] = {
                "image_id": image_id,
                "pa_deg": row.get(f"{prefix}_PA", ""),
                "fl_mm": row.get(f"{prefix}_FL", ""),
                "mt_mm": row.get(f"{prefix}_MT", ""),
            }
        out.append((name, data))
    return out


def build_expert_benchmark_rows(
    primary_candidate_csvs: list[tuple[str, Path]],
    extra_candidate_csvs: list[tuple[str, Path]],
    review_dir: Path,
) -> tuple[list[dict], list[dict]]:
    bench_dir = find_expert_benchmark_dir()
    truth_rows = read_xlsx_first_sheet(next(bench_dir.glob("Results_benchmark_architecture*.xlsx")))
    taxonomy = read_keyed(ROOT / "results" / "benchmark_error_taxonomy.csv")
    candidate_data = (
        [(name, read_keyed(path)) for name, path in primary_candidate_csvs]
        + benchmark_candidate_data(truth_rows)
        + [(name, read_keyed(path)) for name, path in extra_candidate_csvs]
    )
    summary_acc: dict[str, dict[str, list[float]]] = {
        name: {"pa": [], "fl": [], "mt": [], "overall": []} for name, _ in candidate_data
    }
    rows = []
    for src in truth_rows:
        image_id = stem(src.get("ImageID", ""))
        label_id = f"benchmark_{safe_id(image_id)}"
        scale_cm = parse_float(src.get("Scale_pixel_per_cm"))
        human, drop_notes = robust_expert_human(src)
        boundary = candidate_boundary_from_apo_mask(ROOT / "results" / "visual_review" / f"{image_id}_apo.png")
        projection_spans = robust_projection_spans(
            ROOT / "results" / "visual_review" / f"{image_id}_fasc.png",
            boundary,
            None if scale_cm is None else scale_cm / 10.0,
        )
        candidates = []
        for name, data in candidate_data:
            pred_row = data.get(image_id, {})
            pred = {
                "pa_deg": parse_float(pred_row.get("pa_deg")),
                "fl_mm": parse_float(pred_row.get("fl_mm")),
                "mt_mm": parse_float(pred_row.get("mt_mm")),
            }
            deltas, overall = norm_delta(human, pred)
            candidates.append({"name": name, **pred, **deltas, "overall_norm": overall})
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
            "image_path": str(bench_dir / f"{image_id}.tif"),
            "source": "osf_expert_benchmark",
            "quality": "expert consensus",
            "apo_pixels": "",
            "fasc_pixels": "",
            "n_fascicles": "3 expert fascicle measurements averaged",
            "scale_px_per_mm": None if scale_cm is None else scale_cm / 10.0,
            "scale_px_per_cm": scale_cm,
            "calibration_method": "expert_xlsx_true_scale",
            "calibration_confidence": 1.0,
            "apo_lines": None,
            "candidate_boundary": boundary,
            "candidate_projection_spans": projection_spans,
            "label_available": False,
            "overlay_available": any((ROOT / "results" / "visual_review" / f"{image_id}_{layer}.png").exists()
                                     for layer in ("apo", "fasc", "ignore")),
            "overlay_paths": {
                **{
                    layer: str(ROOT / "results" / "visual_review" / f"{image_id}_{layer}.png")
                    for layer in ("apo", "fasc", "ignore")
                    if (ROOT / "results" / "visual_review" / f"{image_id}_{layer}.png").exists()
                },
                **({"diag": str(ROOT / "results" / "benchmark_overlay" / f"{image_id}.jpg")}
                   if (ROOT / "results" / "benchmark_overlay" / f"{image_id}.jpg").exists() else {}),
            },
            "truth_label": "expert consensus",
            "truth_note": "35-image benchmark: robust expert consensus, true scale, final PA/FL/MT only."
                          + (f" Dropped tail(s): {'; '.join(drop_notes)}." if drop_notes else ""),
            "human": human,
            "candidates": candidates,
            "taxonomy": taxonomy.get(image_id, {}),
            "sort_score": sort_score,
            "review": load_review(review_dir / f"{label_id}.json"),
        })
    rows.sort(key=lambda r: (-1 if r["sort_score"] is None else -r["sort_score"], r["image_id"]))
    return rows, summarize_candidates(summary_acc)


def build_synthetic_rows(
    synthetic_dir: Path,
    extra_candidate_csvs: list[tuple[str, Path]],
    review_dir: Path,
) -> tuple[list[dict], list[dict]]:
    truth_rows = read_csv(synthetic_dir / "truth.csv")
    score_rows = read_keyed(synthetic_dir / "measure_light_scores.csv")
    labels_dir = synthetic_dir / "labels"
    candidate_data: list[tuple[str, dict[str, dict[str, str]]]] = []
    generated_pred: dict[str, dict[str, str]] = {}
    for row in read_csv(synthetic_dir / "measure_light_scores.csv"):
        image_id = stem(row.get("image_id", ""))
        generated_pred[image_id] = {
            "image_id": image_id,
            "pa_deg": row.get("pred_pa_deg", ""),
            "fl_mm": row.get("pred_fl_mm", ""),
            "mt_mm": row.get("pred_mt_mm", ""),
        }
    candidate_data.append(("current_straight_scorer", generated_pred))
    candidate_data.extend((name, read_keyed(path)) for name, path in extra_candidate_csvs)
    summary_acc: dict[str, dict[str, list[float]]] = {
        name: {"pa": [], "fl": [], "mt": [], "overall": []} for name, _ in candidate_data
    }
    rows = []
    for src in truth_rows:
        image_id = stem(src.get("image_id", src.get("case_id", "")))
        label_id = safe_id(src.get("case_id", image_id))
        human = {
            "pa_deg": parse_float(src.get("pa_deg")),
            "fl_mm": parse_float(src.get("fl_mm")),
            "mt_mm": parse_float(src.get("mt_mm")),
        }
        candidates = []
        for name, data in candidate_data:
            pred_row = data.get(image_id, {})
            pred = {
                "pa_deg": parse_float(pred_row.get("pa_deg")),
                "fl_mm": parse_float(pred_row.get("fl_mm")),
                "mt_mm": parse_float(pred_row.get("mt_mm")),
            }
            deltas, overall = norm_delta(human, pred)
            candidates.append({"name": name, **pred, **deltas, "overall_norm": overall})
            if overall is not None:
                summary_acc[name]["overall"].append(overall)
            if deltas["delta_pa"] is not None:
                summary_acc[name]["pa"].append(abs(deltas["delta_pa"]) / TOL["pa_deg"])
            if deltas["delta_fl"] is not None:
                summary_acc[name]["fl"].append(abs(deltas["delta_fl"]) / TOL["fl_mm"])
            if deltas["delta_mt"] is not None:
                summary_acc[name]["mt"].append(abs(deltas["delta_mt"]) / TOL["mt_mm"])
        score_row = score_rows.get(image_id, {})
        sort_score = candidates[0]["overall_norm"] if candidates else None
        rows.append({
            "label_id": label_id,
            "image_id": image_id,
            "image_path": src.get("image_path", ""),
            "source": f"synthetic_geometry/{src.get('family', '')}",
            "quality": "synthetic exact",
            "apo_pixels": "",
            "fasc_pixels": "",
            "n_fascicles": src.get("n_strands", ""),
            "scale_px_per_mm": parse_float(src.get("px_per_mm")),
            "scale_px_per_cm": None if parse_float(src.get("px_per_mm")) is None else parse_float(src.get("px_per_mm")) * 10.0,
            "calibration_method": "synthetic_known_scale",
            "calibration_confidence": 1.0,
            "apo_lines": apo_line_fits(labels_dir, label_id),
            "label_available": True,
            "truth_label": "synthetic exact",
            "truth_note": (
                f"Family {src.get('family', '')}: exact generated target. "
                f"Curve bow {src.get('mean_curve_bow_px', '')} px; boundary curve {src.get('boundary_curve_px', '')} px; "
                f"current scorer error {score_row.get('overall_error_units', '')} units."
            ),
            "human": human,
            "candidates": candidates,
            "sort_score": sort_score,
            "review": load_review(review_dir / f"{label_id}.json"),
        })
    rows.sort(key=lambda r: (-1 if r["sort_score"] is None else -r["sort_score"], r["image_id"]))
    return rows, summarize_candidates(summary_acc)


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    summary: list[dict] = []
    metadata: dict = {}
    labels_dir: Path = ROOT / "results" / "human_benchmark" / "target_labels"
    review_dir: Path = ROOT / "results" / "human_benchmark" / "review_notes"
    session_dir: Path = VIEWER_V2_SESSION_DIR

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

    def send_static_v2(self, rel_path: str) -> None:
        rel = rel_path.strip("/") or "index.html"
        path = (VIEWER_V2_DIR / rel).resolve()
        try:
            path.relative_to(VIEWER_V2_DIR.resolve())
        except ValueError:
            self.send_text("bad static path", 404)
            return
        if not path.exists() or not path.is_file():
            self.send_text("static file not found", 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in {".html", ".css", ".js"}:
            ctype += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), ctype)

    def session_path(self) -> Path:
        return self.session_dir / "session.json"

    def load_session(self) -> dict:
        path = self.session_path()
        if not path.exists():
            return {"items": {}, "updated_at": None}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"items": {}, "updated_at": None}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/v2" or parsed.path == "/v2/":
            self.send_static_v2("index.html")
            return
        if parsed.path.startswith("/viewer_v2/"):
            self.send_static_v2(parsed.path.removeprefix("/viewer_v2/"))
            return
        if parsed.path == "/api/review":
            self.send_json({"rows": self.rows, "summary": self.summary, "metadata": self.metadata})
            return
        if parsed.path == "/api/session":
            self.send_json(self.load_session())
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
            overlay_paths = row.get("overlay_paths") or {}
            path = Path(overlay_paths.get(layer, "")) if overlay_paths.get(layer) else self.labels_dir / row["label_id"] / f"{layer}.png"
            if not path.exists():
                self.send_text("not labeled yet", 404)
                return
            self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png")
            return
        self.send_text("not found", 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/save_session":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("session payload must be an object")
            except Exception as exc:
                self.send_text(f"bad session payload: {exc}", 400)
                return
            self.session_dir.mkdir(parents=True, exist_ok=True)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.session_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.send_json({"ok": True, "updated_at": payload["updated_at"]})
            return
        if path != "/api/save_review":
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
    ap.add_argument("--primary-pred-csv", action="append", default=[],
                    help="expert-benchmark only: candidate(s) to prepend, making the first one control sorting/deltas")
    ap.add_argument("--include-rejected", action="store_true",
                    help="also show rejected historical candidates such as tail-bar and old segmentation")
    ap.add_argument("--expert-benchmark", action="store_true",
                    help="show the 35-image OSF expert benchmark instead of target human labels")
    ap.add_argument("--synthetic-dir", default="",
                    help="show a synthetic geometry benchmark directory generated by generate_synthetic_geometry.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()

    if args.synthetic_dir:
        candidate_csvs = parse_candidate_arg(args.pred_csv)
        rows, summary = build_synthetic_rows(Path(args.synthetic_dir), candidate_csvs, Path(args.review_dir))
        Handler.labels_dir = Path(args.synthetic_dir) / "labels"
    elif args.expert_benchmark:
        primary_candidate_csvs = parse_candidate_arg(args.primary_pred_csv)
        candidate_csvs = dedupe_candidate_csvs(default_expert_candidate_csvs() + parse_candidate_arg(args.pred_csv))
        rows, summary = build_expert_benchmark_rows(primary_candidate_csvs, candidate_csvs, Path(args.review_dir))
        Handler.labels_dir = Path(args.labels_dir)
    else:
        candidate_csvs = parse_candidate_arg(args.pred_csv) if args.pred_csv else default_candidate_csvs()
        if args.include_rejected:
            candidate_csvs.extend(rejected_candidate_csvs())
        rows, summary = build_rows(
            Path(args.manifest),
            Path(args.labels_dir),
            Path(args.scores),
            Path(args.calibration),
            candidate_csvs,
            Path(args.review_dir),
        )
        Handler.labels_dir = Path(args.labels_dir)
    metadata = enrich_rows_for_v2(rows, summary)
    Handler.rows = rows
    Handler.summary = summary
    Handler.metadata = metadata
    Handler.review_dir = Path(args.review_dir)
    Handler.session_dir = VIEWER_V2_SESSION_DIR / safe_id("expert_benchmark" if args.expert_benchmark else (Path(args.synthetic_dir).name if args.synthetic_dir else "target_review"))
    Handler.review_dir.mkdir(parents=True, exist_ok=True)
    Handler.session_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"UMUD label review: http://{args.host}:{args.port}")
    print(f"Viewer v2: http://{args.host}:{args.port}/v2")
    print(f"labeled rows: {len(rows)}")
    print("candidates:")
    if args.synthetic_dir:
        print("  built-in: current_straight_scorer")
        for name, path in candidate_csvs:
            print(f"  {name}: {path}")
    elif args.expert_benchmark:
        print("  built-in: our_pipeline_true_scale, DLTrack, SMA")
        for name, path in candidate_csvs:
            print(f"  {name}: {path}")
    else:
        for name, path in candidate_csvs:
            print(f"  {name}: {path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
