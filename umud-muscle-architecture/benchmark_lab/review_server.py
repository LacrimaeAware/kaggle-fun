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
          <button class="layer active" data-layer="diag">lines</button>
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
  visible: {apo: true, fasc: true, ignore: false, diag: true},
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
    parts.push('<b>candidate boundary visible:</b> magenta = robust triangle upper boundary; yellow = lower boundary. Cyan overlay spans are old projection diagnostics, not the robust-triangle boundary.');
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


def candidate_boundary_from_apo_mask(path: Path) -> dict | None:
    apo = load_mask(path)
    if apo is None:
        return None
    groups = apo_boundary_groups(np.ascontiguousarray(apo, np.uint8))
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
            overlay_paths = row.get("overlay_paths") or {}
            path = Path(overlay_paths.get(layer, "")) if overlay_paths.get(layer) else self.labels_dir / row["label_id"] / f"{layer}.png"
            if not path.exists():
                self.send_text("not labeled yet", 404)
                return
            self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png")
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
        candidate_csvs = parse_candidate_arg(args.pred_csv)
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
    Handler.rows = rows
    Handler.summary = summary
    Handler.review_dir = Path(args.review_dir)
    Handler.review_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"UMUD label review: http://{args.host}:{args.port}")
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
