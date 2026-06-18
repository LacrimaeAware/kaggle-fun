"use strict";
const $ = s => document.querySelector(s);
const canvas = $("#canvas"), ctx = canvas.getContext("2d");
const DEG = 180 / Math.PI, LS = "umud_corr_";
const PRIORITY = new Set(["IMG_00001.tif", "IMG_00015.tif", "IMG_00029.tif", "IMG_00206.tif", "IMG_00220.tif", "IMG_00234.tif", "IMG_00037.tif", "IMG_00070.tif", "IMG_00150.tif", "IMG_00164.tif", "IMG_00252.png", "IMG_00266.png", "IMG_00280.png", "IMG_00294.png", "IMG_00056.tif", "IMG_00112.tif", "IMG_00128.tif", "IMG_00074.tif", "IMG_00093.tif", "IMG_00171.tif", "IMG_00036.tif"]);
const isPriority = r => PRIORITY.has(r.image_id);
const viewMode = () => { const el = document.getElementById("view"); return el ? el.value : "all"; };
function passFilter(r) { const v = viewMode(); if (v === "priority") return isPriority(r); if (v === "multiband") return !!r.multiband; if (v === "noscale") return r.calibration_method === "none"; return true; }
const filteredIndices = () => state.rows.map((r, i) => i).filter(i => passFilter(state.rows[i]));

const BANNER = {
  scale: "SCALE - click two ADJACENT ruler ticks, then pick spacing. Wheel to zoom, middle-mouse to pan. Skip if scale looks right.",
  apo: "APO LINES - drag the handles of the cyan (superficial) and yellow (deep) lines. Tick 'curved top' to bend the superficial into 3 points. Type an angle or 'snap 0/90'.",
  reject: "REJECT - click any GREEN fascicle that is wrong or shouldn't count. Click again to bring it back.",
  add: "ADD - click the SUPERFICIAL end then the DEEP end of a missed fascicle. It auto-selects: drag its dots, type an angle, or snap. Click an existing one to re-select it.",
  blind: "BLIND ANGLE - model hidden. Click the two ends of the SINGLE dominant fascicle. Then drag its dots, type the angle, or snap.",
};

const state = {
  rows: [], index: 0, prefill: null, corr: {}, tool: "reject",
  img: new Image(), loadingId: null, ready: false,
  pending: null, cursor: null, drag: null, pan: null, scalePts: [], saveTimer: null,
  overlay: 1, zoom: 1, layers: { apo: true, fasc: true, mt: true }, active: null,
};

const W = () => state.prefill.geometry.width;
const Hh = () => state.prefill.geometry.height;
const curRow = () => state.rows[state.index];
const isBlind = () => curRow() && String(curRow().blind_angle) === "1";
const supCoef = () => (state.corr.apo && state.corr.apo.superficial_coef) || state.prefill.geometry.apo.superficial_coef;
const deepCoef = () => (state.corr.apo && state.corr.apo.deep_coef) || state.prefill.geometry.apo.deep_coef;
const topOverride = () => (state.corr.apo && state.corr.apo.top_boundary && state.corr.apo.top_boundary.points) ? state.corr.apo.top_boundary : null;
const yAt = (c, x) => c[0] * x + c[1];
const keptFrags = () => state.prefill.geometry.fragments.filter(f => f.kept);
const isRejected = id => (state.corr.reject || []).includes(id);
const scaleNow = () => (state.corr.scale_px_per_mm != null) ? state.corr.scale_px_per_mm : state.prefill.scale.px_per_mm;
const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const grabTol = () => { const sc = (canvas.getBoundingClientRect().width / canvas.width) || 1; return Math.max(8, 18 / sc); };

function interp(a, b, x) { return a[1] + (b[1] - a[1]) * ((x - a[0]) / ((b[0] - a[0]) || 1e-9)); }
function topYat(x) { const tb = topOverride(); if (!tb) return yAt(supCoef(), x); const p = tb.points; return x <= p[1][0] ? interp(p[0], p[1], x) : interp(p[1], p[2], x); }

function canvasPoint(e) { const r = canvas.getBoundingClientRect(); return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height }; }
function distToSeg(p, a, b) { const dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy || 1; let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / L2; t = Math.max(0, Math.min(1, t)); return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy)); }
function fragSegment(f) { const s = f.slope, L = Math.max(f.visible_len || 30, 30), ux = 1 / Math.sqrt(1 + s * s), uy = s * ux, cx = f.centroid[0], cy = f.centroid[1]; return [{ x: cx - ux * L / 2, y: cy - uy * L / 2 }, { x: cx + ux * L / 2, y: cy + uy * L / 2 }]; }
function nearestFrag(p) { let best = null, bd = grabTol(); for (const f of keptFrags()) { const [a, b] = fragSegment(f), d = distToSeg(p, a, b); if (d < bd) { bd = d; best = f; } } return best; }
function nearestAdd(p) { const add = state.corr.add || []; let bi = -1, bd = grabTol(); for (let i = 0; i < add.length; i++) { const d = distToSeg(p, { x: add[i].p1[0], y: add[i].p1[1] }, { x: add[i].p2[0], y: add[i].p2[1] }); if (d < bd) { bd = d; bi = i; } } return bi; }
function imgAngle(p1, p2) { let a = Math.atan2(p2.y - p1.y, p2.x - p1.x) * DEG; while (a > 90) a -= 180; while (a <= -90) a += 180; return a; }
const deepImgAngle = () => Math.atan(deepCoef()[0]) * DEG;
function vsDeep(a) { let d = Math.abs(a - deepImgAngle()); return d > 90 ? 180 - d : d; }
function ensureApo() { if (!state.corr.apo) state.corr.apo = { superficial_coef: [...state.prefill.geometry.apo.superficial_coef], deep_coef: [...state.prefill.geometry.apo.deep_coef] }; }

function activeGetPts() {
  const a = state.active; if (!a) return null;
  if (a.kind === "add") { const d = state.corr.add[a.idx]; return d ? [{ x: d.p1[0], y: d.p1[1] }, { x: d.p2[0], y: d.p2[1] }] : null; }
  if (a.kind === "blind") { const d = state.corr.blind_angle_line; return d ? [{ x: d.p1[0], y: d.p1[1] }, { x: d.p2[0], y: d.p2[1] }] : null; }
  if (a.kind === "apoSup") { const c = supCoef(); return [{ x: 0, y: yAt(c, 0) }, { x: W(), y: yAt(c, W()) }]; }
  if (a.kind === "apoDeep") { const c = deepCoef(); return [{ x: 0, y: yAt(c, 0) }, { x: W(), y: yAt(c, W()) }]; }
  return null;
}
function activeSetPts(p1, p2) { const a = state.active; if (a.kind === "add") state.corr.add[a.idx] = { p1: [p1.x, p1.y], p2: [p2.x, p2.y] }; else if (a.kind === "blind") state.corr.blind_angle_line = { p1: [p1.x, p1.y], p2: [p2.x, p2.y] }; }
function setActive(a) { state.active = a; updateHud(); }

// ---------------- drawing ----------------
function fullLine(c, color, dash, wdt) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = wdt || 2; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(0, yAt(c, 0)); ctx.lineTo(W(), yAt(c, W())); ctx.stroke(); ctx.restore(); }
function seg(a, b, color, wdt, dash) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = wdt || 2; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.restore(); }
function handle(x, y) { const r = Math.max(7, grabTol() * 0.5); ctx.save(); ctx.globalAlpha = 1; ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore(); }
function dot(p, color) { ctx.save(); ctx.globalAlpha = 1; ctx.fillStyle = color; ctx.strokeStyle = "#000"; ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore(); }

function draw() {
  if (!state.ready) return;
  canvas.width = W(); canvas.height = Hh();
  ctx.drawImage(state.img, 0, 0, W(), Hh());
  const blind = isBlind(), tb = topOverride();

  ctx.save(); ctx.globalAlpha = state.overlay;
  if (state.layers.fasc && !blind) {
    for (const f of state.prefill.geometry.fragments) {
      const [a, b] = fragSegment(f);
      if (!f.kept) {  // pipeline already excluded this from the CSV (out_of_band / apo_parallel / etc.)
        seg(a, b, "#ff6a3d", 2, [4, 4]);
        if (f.reject_reason) { ctx.save(); ctx.fillStyle = "#ff6a3d"; ctx.font = `${Math.max(12, grabTol())}px sans-serif`; ctx.fillText(f.reject_reason, b.x + 6, b.y); ctx.restore(); }
        continue;
      }
      const rej = isRejected(f.id), s = f.slope, c = [s, f.centroid[1] - s * f.centroid[0]];
      fullLine(c, rej ? "rgba(255,80,80,.30)" : "rgba(120,255,150,.40)", [6, 6], 1);
      seg(a, b, rej ? "#ff5050" : "#5dff8a", rej ? 2 : 3, rej ? [4, 4] : []);
    }
    (state.corr.add || []).forEach((ad, i) => {
      const p1 = { x: ad.p1[0], y: ad.p1[1] }, p2 = { x: ad.p2[0], y: ad.p2[1] }, s = (p2.y - p1.y) / ((p2.x - p1.x) || 1e-9);
      fullLine([s, p1.y - s * p1.x], "rgba(255,174,66,.35)", [6, 6], 1);
      seg(p1, p2, (state.active && state.active.kind === "add" && state.active.idx === i) ? "#ffd083" : "#ffae42", 3);
    });
  }
  if (state.layers.apo) {
    if (tb) { const p = tb.points; seg({ x: p[0][0], y: p[0][1] }, { x: p[1][0], y: p[1][1] }, "#37c8ff", 2); seg({ x: p[1][0], y: p[1][1] }, { x: p[2][0], y: p[2][1] }, "#37c8ff", 2); }
    else fullLine(supCoef(), "#37c8ff", [], 2);
    fullLine(deepCoef(), "#ffe14d", [], 2);
  }
  if (state.layers.mt) { const xc = W() / 2; seg({ x: xc, y: topYat(xc) }, { x: xc, y: yAt(deepCoef(), xc) }, "#ff66d8", 2, [5, 4]); }
  if (blind && state.corr.blind_angle_line) { const b = state.corr.blind_angle_line; seg({ x: b.p1[0], y: b.p1[1] }, { x: b.p2[0], y: b.p2[1] }, "#ffae42", 4); }
  ctx.restore();

  // always-visible handles
  if (state.tool === "apo" && state.layers.apo && !blind) {
    if (tb) tb.points.forEach(pt => handle(pt[0], pt[1])); else { const s = supCoef(); handle(0, yAt(s, 0)); handle(W(), yAt(s, W())); }
    const dc = deepCoef(); handle(0, yAt(dc, 0)); handle(W(), yAt(dc, W()));
  }
  if (state.active && (state.active.kind === "add" || state.active.kind === "blind")) { const pts = activeGetPts(); if (pts) pts.forEach(p => handle(p.x, p.y)); }
  if (state.corr.scale_line) { const sl = state.corr.scale_line, a = { x: sl.p1[0], y: sl.p1[1] }, b = { x: sl.p2[0], y: sl.p2[1] }; seg(a, b, "#ffd83a", 3); dot(a, "#ffd83a"); dot(b, "#ffd83a"); }
  for (const p of state.scalePts) dot(p, "#ffd83a");
  if (state.scalePts.length === 2) seg(state.scalePts[0], state.scalePts[1], "#ffd83a", 2);
  if (state.pending) { dot(state.pending, "#ffae42"); if (state.cursor) seg(state.pending, state.cursor, "#ffae42", 2, [4, 4]); }
}

// ---------------- zoom / pan ----------------
function applyZoom() { if (!state.prefill || !state.prefill.geometry) return; const stage = $("#stage"), fitW = Math.max(50, Math.min(W(), stage.clientWidth - 24)); canvas.style.maxWidth = "none"; canvas.style.width = (fitW * state.zoom) + "px"; canvas.style.height = "auto"; }
canvas.addEventListener("wheel", e => {
  if (!state.ready) return; e.preventDefault();
  if (e.shiftKey) { state.overlay = Math.max(0, Math.min(1, state.overlay - Math.sign(e.deltaY) * 0.1)); $("#opacity").value = Math.round(state.overlay * 100); $("#opval").textContent = Math.round(state.overlay * 100) + "%"; draw(); return; }
  const stage = $("#stage"), rect = canvas.getBoundingClientRect();
  const fx = (e.clientX - rect.left) / rect.width, fy = (e.clientY - rect.top) / rect.height;
  state.zoom = Math.max(1, Math.min(10, state.zoom * (e.deltaY < 0 ? 1.2 : 1 / 1.2))); applyZoom();
  const nr = canvas.getBoundingClientRect(), sr = stage.getBoundingClientRect();
  stage.scrollLeft = fx * nr.width - (e.clientX - sr.left); stage.scrollTop = fy * nr.height - (e.clientY - sr.top);
}, { passive: false });
window.addEventListener("resize", applyZoom);
canvas.addEventListener("mousedown", e => { if (e.button === 1) e.preventDefault(); });
canvas.addEventListener("auxclick", e => e.preventDefault());
canvas.addEventListener("contextmenu", e => e.preventDefault());

// ---------------- tool / banner / hud ----------------
function setTool(t) {
  if (isBlind()) t = "blind";
  state.tool = t; state.pending = null; state.scalePts = []; hideChoose();
  document.querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b.dataset.tool === t));
  const ban = $("#banner"); ban.textContent = BANNER[t] || ""; ban.classList.toggle("blind", t === "blind"); draw();
}
function updateHud() {
  const a = state.active, el = $("#angleInfo");
  if (!a) { el.textContent = topOverride() ? "curved top: drag the 3 points" : "none selected"; return; }
  if (a.kind === "apoSup" || a.kind === "apoDeep") { const c = a.kind === "apoSup" ? supCoef() : deepCoef(), ia = Math.atan(c[0]) * DEG; el.textContent = `${a.kind === "apoSup" ? "superficial" : "deep"} apo: ${ia.toFixed(1)} deg`; $("#angleInput").value = ia.toFixed(1); }
  else { const pts = activeGetPts(); if (!pts) { el.textContent = "none selected"; return; } const ia = imgAngle(pts[0], pts[1]); el.textContent = `fascicle: ${ia.toFixed(1)} deg (pennation vs deep: ${vsDeep(ia).toFixed(1)} deg)`; $("#angleInput").value = ia.toFixed(1); }
}
function applyAngle() {
  const v = parseFloat($("#angleInput").value), a = state.active; if (isNaN(v) || !a) return;
  const rad = v / DEG;
  if (a.kind === "apoSup" || a.kind === "apoDeep") { const key = a.kind === "apoSup" ? "superficial_coef" : "deep_coef", c = a.kind === "apoSup" ? supCoef() : deepCoef(), midX = W() / 2, midY = yAt(c, midX), slope = Math.tan(rad); ensureApo(); state.corr.apo[key] = [slope, midY - slope * midX]; }
  else { const pts = activeGetPts(), mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 }, len = d2(pts[0], pts[1]) || 60, dx = Math.cos(rad) * len / 2, dy = Math.sin(rad) * len / 2; activeSetPts({ x: mid.x - dx, y: mid.y - dy }, { x: mid.x + dx, y: mid.y + dy }); }
  edited(); updateHud();
}
function snapNearest() {
  const a = state.active; if (!a) return; let ia;
  if (a.kind === "apoSup" || a.kind === "apoDeep") ia = Math.atan((a.kind === "apoSup" ? supCoef() : deepCoef())[0]) * DEG;
  else { const pts = activeGetPts(); if (!pts) return; ia = imgAngle(pts[0], pts[1]); }
  $("#angleInput").value = Math.abs(ia) < 45 ? 0 : 90; applyAngle();
}
function showChoose() { $("#scalechoose").classList.remove("hidden"); }
function hideChoose() { $("#scalechoose").classList.add("hidden"); }

// ---------------- interaction ----------------
canvas.addEventListener("pointerdown", e => {
  if (!state.ready) return;
  const stage = $("#stage");
  if (e.button === 1) { state.pan = { x: e.clientX, y: e.clientY, sl: stage.scrollLeft, st: stage.scrollTop }; try { canvas.setPointerCapture(e.pointerId); } catch (x) {} e.preventDefault(); return; }
  if (e.button !== 0) return;
  const p = canvasPoint(e), blind = isBlind(), tol = grabTol();
  // grab an active add/blind endpoint first
  if (state.active && (state.active.kind === "add" || state.active.kind === "blind")) { const pts = activeGetPts(); for (let i = 0; i < 2; i++) if (pts && d2(p, pts[i]) < tol) { state.drag = { kind: "pt", end: i }; try { canvas.setPointerCapture(e.pointerId); } catch (x) {} return; } }
  if (blind) { if (state.corr.blind_angle_line) return; twoClick(p, pts => { state.corr.blind_angle_line = { p1: [pts[0].x, pts[0].y], p2: [pts[1].x, pts[1].y] }; setActive({ kind: "blind" }); edited(); }); return; }
  if (state.tool === "scale") {
    if (state.scalePts.length >= 2) return;
    let sp = p;
    if (state.scalePts.length === 1) { const a = state.scalePts[0]; sp = Math.abs(p.x - a.x) >= Math.abs(p.y - a.y) ? { x: p.x, y: a.y } : { x: a.x, y: p.y }; }  // snap to horizontal or vertical (ruler axis)
    state.scalePts.push(sp);
    if (state.scalePts.length === 2) showChoose();
    draw(); return;
  }
  if (state.tool === "apo") {
    const tb = topOverride();
    if (tb) { for (let i = 0; i < 3; i++) if (d2(p, { x: tb.points[i][0], y: tb.points[i][1] }) < tol) { state.drag = { kind: "tbpt", i }; setActive(null); try { canvas.setPointerCapture(e.pointerId); } catch (x) {} return; } }
    else { const s = supCoef(); if (d2(p, { x: 0, y: yAt(s, 0) }) < tol) { ensureApo(); setActive({ kind: "apoSup" }); state.drag = { kind: "apo", key: "superficial_coef", end: 0, c: [...s] }; return; } if (d2(p, { x: W(), y: yAt(s, W()) }) < tol) { ensureApo(); setActive({ kind: "apoSup" }); state.drag = { kind: "apo", key: "superficial_coef", end: 1, c: [...s] }; return; } }
    const dc = deepCoef(); if (d2(p, { x: 0, y: yAt(dc, 0) }) < tol) { ensureApo(); setActive({ kind: "apoDeep" }); state.drag = { kind: "apo", key: "deep_coef", end: 0, c: [...dc] }; return; } if (d2(p, { x: W(), y: yAt(dc, W()) }) < tol) { ensureApo(); setActive({ kind: "apoDeep" }); state.drag = { kind: "apo", key: "deep_coef", end: 1, c: [...dc] }; return; }
    return;
  }
  if (state.tool === "reject") { const f = nearestFrag(p); if (!f) return; state.corr.reject = state.corr.reject || []; const i = state.corr.reject.indexOf(f.id); if (i >= 0) state.corr.reject.splice(i, 1); else state.corr.reject.push(f.id); edited(); return; }
  if (state.tool === "add") { const ai = nearestAdd(p); if (ai >= 0) { setActive({ kind: "add", idx: ai }); draw(); return; } twoClick(p, pts => { state.corr.add = state.corr.add || []; state.corr.add.push({ p1: [pts[0].x, pts[0].y], p2: [pts[1].x, pts[1].y] }); setActive({ kind: "add", idx: state.corr.add.length - 1 }); edited(); }); return; }
});
function twoClick(p, done) { if (!state.pending) { state.pending = p; draw(); return; } const pts = [state.pending, p]; state.pending = null; state.cursor = null; done(pts); }
canvas.addEventListener("pointermove", e => {
  if (state.pan) { const stage = $("#stage"); stage.scrollLeft = state.pan.sl - (e.clientX - state.pan.x); stage.scrollTop = state.pan.st - (e.clientY - state.pan.y); return; }
  if (!state.ready) return;
  const p = canvasPoint(e);
  if (state.drag && state.drag.kind === "apo") { const d = state.drag, other = d.end === 0 ? { x: W(), y: yAt(d.c, W()) } : { x: 0, y: yAt(d.c, 0) }, moved = { x: d.end === 0 ? 0 : W(), y: p.y }, a = d.end === 0 ? moved : other, b = d.end === 0 ? other : moved, s = (b.y - a.y) / (b.x - a.x || 1e-9); ensureApo(); state.corr.apo[d.key] = [s, a.y - s * a.x]; updateHud(); draw(); return; }
  if (state.drag && state.drag.kind === "tbpt") { state.corr.apo.top_boundary.points[state.drag.i] = [p.x, p.y]; draw(); return; }
  if (state.drag && state.drag.kind === "pt") { const pts = activeGetPts(); pts[state.drag.end] = p; activeSetPts(pts[0], pts[1]); updateHud(); draw(); return; }
  if (state.pending) { state.cursor = p; draw(); }
});
canvas.addEventListener("pointerup", e => { if (state.pan) { state.pan = null; try { canvas.releasePointerCapture(e.pointerId); } catch (x) {} return; } if (state.drag) { state.drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch (x) {} edited(); } });

document.querySelectorAll("#scalechoose button[data-mm]").forEach(b => b.onclick = () => { const mm = parseFloat(b.dataset.mm), p = state.scalePts; if (p.length === 2 && mm > 0) { state.corr.scale_px_per_mm = d2(p[0], p[1]) / mm; state.corr.scale_line = { p1: [p[0].x, p[0].y], p2: [p[1].x, p[1].y], mm }; updateScaleRead(); edited(); } state.scalePts = []; hideChoose(); draw(); });
$("#scaleCancel").onclick = () => { state.scalePts = []; hideChoose(); draw(); };
$("#angleApply").onclick = applyAngle;
$("#snapBtn").onclick = snapNearest;
$("#apoCurve").onchange = e => {
  if (e.target.checked) { ensureApo(); const s = supCoef(); state.corr.apo.top_boundary = { type: "piecewise", points: [[0, yAt(s, 0)], [W() / 2, yAt(s, W() / 2)], [W(), yAt(s, W())]] }; }
  else if (state.corr.apo) delete state.corr.apo.top_boundary;
  setActive(null); edited();
};

// ---------------- persistence ----------------
function edited() { draw(); try { localStorage.setItem(LS + curRow().image_id, JSON.stringify(state.corr)); } catch (e) {} setStatus("unsaved edit...", "saving"); scheduleSave(); }
function scheduleSave() { clearTimeout(state.saveTimer); state.saveTimer = setTimeout(save, 500); }
async function save() {
  const row = curRow(); if (!row) return false;
  try { localStorage.setItem(LS + row.image_id, JSON.stringify(state.corr)); } catch (e) {}
  row.done = true; renderList(); setStatus("saving...", "saving");
  try { const r = await fetch("/api/corrections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_id: row.image_id, corrections: state.corr, blind_angle: isBlind(), updated_at: new Date().toISOString() }) }); const j = await r.json(); if (!j.ok) throw 0; setStatus("saved to disk ✓", "ok"); if (j.score) renderScore(j.score); }
  catch (e) { setStatus("saved locally ✓ (server down) - Export backup to be safe", "err"); }
  return true;
}
function setStatus(t, c) { const s = $("#status"); s.textContent = t; s.className = "pill " + (c || ""); }
function updateScaleRead() { const px = scaleNow(); $("#scaleread").textContent = px ? `scale ${(px * 10).toFixed(1)} px/cm${state.corr.scale_px_per_mm != null ? " (yours)" : ""}` : "scale none"; }
$("#exportBtn").onclick = () => { const out = {}; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); if (k.startsWith(LS)) out[k.slice(LS.length)] = JSON.parse(localStorage.getItem(k)); } const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `umud_corrections_${Object.keys(out).length}.json`; a.click(); };

// ---------------- score ----------------
const fmt = m => m ? `PA ${m.pa_deg}  FL ${m.fl_mm}  MT ${m.mt_mm}` : "-";
function renderScore(sc) {
  if (!sc || sc.error) { $("#sc-base").textContent = (sc && sc.error) || "-"; $("#sc-corr").textContent = "-"; $("#sc-contrib").textContent = "-"; return; }
  $("#sc-base").textContent = fmt(sc.baseline); $("#sc-corr").textContent = fmt(sc.corrected);
  const ov = sc.contribution.overall, max = Math.max(...Object.values(ov).map(Math.abs), 1e-9);
  $("#sc-contrib").innerHTML = Object.keys(ov).map(k => `<span class="ch ${Math.abs(ov[k]) >= max - 1e-9 && max > 1e-4 ? "hi" : ""}">${k} ${ov[k] >= 0 ? "+" : ""}${ov[k].toFixed(3)}</span>`).join("");
}

// ---------------- navigation ----------------
async function loadRow(i) {
  if (i < 0 || i >= state.rows.length) return;
  state.index = i; state.ready = false; state.pending = null; state.scalePts = []; state.active = null; hideChoose();
  const row = curRow(), id = row.image_id; state.loadingId = id;
  $("#imgtitle").textContent = `${id}  (${i + 1}/${state.rows.length})  ${row.calibration_method}`; renderList();
  const [pf, cr] = await Promise.all([
    fetch("/api/prefill?id=" + encodeURIComponent(id)).then(r => r.json()),
    fetch("/api/corrections?id=" + encodeURIComponent(id)).then(r => r.json()).catch(() => ({})),
  ]);
  if (state.loadingId !== id) return;
  state.prefill = pf;
  const serverCorr = (cr && cr.corrections) || {};
  let local = null; try { local = JSON.parse(localStorage.getItem(LS + id)); } catch (e) {}
  state.corr = Object.keys(serverCorr).length ? serverCorr : (local || {});
  document.querySelectorAll(".tool").forEach(b => b.classList.toggle("disabled", isBlind()));
  $("#apoCurve").checked = !!topOverride(); curRow().multiband = !!state.corr.multiband; $("#flagBtn").classList.toggle("on", !!state.corr.multiband); updateScaleRead(); updateHud();
  if (!pf.geometry) setStatus("no geometry (measure failed) - skip this one", "err"); else setStatus("idle", "");
  state.img.onload = () => { if (state.loadingId === id) { state.ready = true; applyZoom(); setTool(isBlind() ? "blind" : state.tool); draw(); } };
  state.img.src = "/image?id=" + encodeURIComponent(id) + "&v=" + (row.done ? "1" : "0");
  fetch("/api/score?id=" + encodeURIComponent(id)).then(r => r.json()).then(s => { if (state.loadingId === id) renderScore(s); }).catch(() => {});
}
function go(dir) { const idxs = filteredIndices(); if (!idxs.length) return; const pos = idxs.indexOf(state.index); const np = pos === -1 ? idxs[0] : idxs[Math.max(0, Math.min(idxs.length - 1, pos + dir))]; loadRow(np); }
async function saveNext() { await save(); await nextUndoneFrom(state.index + 1); }
async function nextUndoneFrom(start) {
  const idxs = filteredIndices(); if (!idxs.length) return;
  let cand = idxs.find(i => i >= start && !state.rows[i].done);
  if (cand == null) cand = idxs.find(i => i >= start);
  if (cand == null) cand = idxs[0];
  return loadRow(cand);
}

// ---------------- list ----------------
function renderList() {
  const onlyUndone = $("#undoneOnly").checked, ul = $("#list"); ul.innerHTML = "";
  state.rows.forEach((r, i) => { if (!passFilter(r)) return; if (onlyUndone && r.done && i !== state.index) return; const li = document.createElement("li"); li.className = (i === state.index ? "active " : "") + (r.done ? "done " : "") + (String(r.blind_angle) === "1" ? "blind " : ""); li.innerHTML = `<span class="name">${r.image_id}</span><span class="tag">${r.multiband ? "⚑ " : ""}${r.n_fragments_kept}f</span>`; li.onclick = () => loadRow(i); ul.appendChild(li); });
  const v = viewMode(), shown = state.rows.filter(passFilter);
  $("#counter").textContent = v === "multiband" ? `${shown.length} flagged multiband`
    : v === "noscale" ? `${shown.length} no-scale rows`
    : v === "priority" ? `${shown.filter(r => r.done).length}/${shown.length} batch done`
    : `${state.rows.filter(r => r.done).length}/${state.rows.length} reviewed`;
}

// ---------------- wiring ----------------
document.querySelectorAll(".tool").forEach(b => b.onclick = () => setTool(b.dataset.tool));
$("#fitBtn").onclick = () => { state.zoom = 1; applyZoom(); $("#stage").scrollTo(0, 0); };
$("#resetBtn").onclick = () => { state.corr = {}; state.scalePts = []; state.active = null; $("#apoCurve").checked = false; hideChoose(); updateHud(); updateScaleRead(); edited(); };
$("#prevBtn").onclick = () => go(-1);
$("#skipBtn").onclick = () => go(1);
$("#nextBtn").onclick = () => saveNext();
$("#undoneOnly").onchange = renderList;
$("#view").onchange = () => { renderList(); if (curRow() && !passFilter(curRow())) nextUndoneFrom(0); };
function toggleFlag() { if (!curRow()) return; state.corr.multiband = !state.corr.multiband; curRow().multiband = !!state.corr.multiband; $("#flagBtn").classList.toggle("on", !!state.corr.multiband); renderList(); edited(); }
$("#flagBtn").onclick = toggleFlag;
$("#status").title = "Autosaves to your browser (localStorage) AND to disk on every edit. The green check in the left list means saved. Use Export backup for a file copy.";
$("#opacity").oninput = e => { state.overlay = e.target.value / 100; $("#opval").textContent = e.target.value + "%"; draw(); };
$("#lyApo").onchange = e => { state.layers.apo = e.target.checked; draw(); };
$("#lyFasc").onchange = e => { state.layers.fasc = e.target.checked; draw(); };
$("#lyMt").onchange = e => { state.layers.mt = e.target.checked; draw(); };
document.addEventListener("keydown", e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "1") setTool("scale"); else if (k === "2") setTool("apo"); else if (k === "3") setTool("reject"); else if (k === "4") setTool("add");
  else if (k === "q" || k === "d" || k === "enter") saveNext();
  else if (k === "a" || k === "arrowleft") go(-1); else if (k === "arrowright") go(1);
  else if (k === "r") $("#resetBtn").click(); else if (k === "f") $("#fitBtn").click(); else if (k === "m") toggleFlag();
  else if (k === "escape") { state.pending = null; state.scalePts = []; hideChoose(); draw(); }
  else return;
  e.preventDefault();
});

(async function init() {
  const m = await fetch("/api/manifest").then(r => r.json());
  state.rows = m.rows || [];
  if (!state.rows.length) { $("#counter").textContent = "no manifest - run bake_prefill.py"; return; }
  renderList(); await nextUndoneFrom(0);
})();
