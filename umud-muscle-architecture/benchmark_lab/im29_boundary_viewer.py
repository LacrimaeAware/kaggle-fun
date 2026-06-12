"""Single-image boundary viewer for im_29_arch.

This is intentionally a one-off inspection tool: no text is drawn over the
image, and the user's exact deepest-left/right upper-boundary line is the
default candidate.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402
from experiments.exp35_benchmark_error_taxonomy import (  # noqa: E402
    MASK_DIR,
    apo_geometry,
    connected,
    fit_inner_edge,
    fragments,
    project_with_lines,
    weighted_median,
)

IMAGE_ID = "im_29_arch"


def data_url_image(arr: np.ndarray, ext: str = ".jpg") -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, 94] if ext.lower() in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, arr, params)
    if not ok:
        raise RuntimeError("failed to encode image")
    mime = "image/jpeg" if ext.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('ascii')}"


def line_points(line: tuple[float, float], width: int) -> dict:
    return {
        "x1": 0.0,
        "y1": float(line[1]),
        "x2": float(width - 1),
        "y2": float(line[0] * (width - 1) + line[1]),
        "slope": float(line[0]),
        "intercept": float(line[1]),
    }


def angle_to_line(fs: float, line_slope: float) -> float:
    angle = abs(math.degrees(math.atan(fs) - math.atan(line_slope)))
    return 180.0 - angle if angle > 90.0 else angle


def deepest_half_line(sup_x: np.ndarray, sup_y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """The user's requested line: deepest point in left half to deepest point in right half."""
    mid = (float(sup_x.min()) + float(sup_x.max())) / 2.0
    left = np.where(sup_x <= mid)[0]
    right = np.where(sup_x >= mid)[0]
    li = left[np.argmax(sup_y[left])]
    ri = right[np.argmax(sup_y[right])]
    p1 = (float(sup_x[li]), float(sup_y[li]))
    p2 = (float(sup_x[ri]), float(sup_y[ri]))
    slope = (p2[1] - p1[1]) / max(p2[0] - p1[0], 1e-9)
    return (float(slope), float(p1[1] - slope * p1[0])), p1, p2


def deepest_percentile_line(
    sup_x: np.ndarray, sup_y: np.ndarray, percentile: float = 95.0
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mid = (float(sup_x.min()) + float(sup_x.max())) / 2.0
    left = np.where(sup_x <= mid)[0]
    right = np.where(sup_x >= mid)[0]

    def point(indices: np.ndarray) -> tuple[float, float]:
        cutoff = np.percentile(sup_y[indices], percentile)
        keep = indices[sup_y[indices] >= cutoff]
        return float(np.median(sup_x[keep])), float(np.median(sup_y[keep]))

    p1 = point(left)
    p2 = point(right)
    slope = (p2[1] - p1[1]) / max(p2[0] - p1[0], 1e-9)
    return (float(slope), float(p1[1] - slope * p1[0])), p1, p2


def triangle_points(
    sup_x: np.ndarray,
    sup_y: np.ndarray,
    robust: bool = False,
) -> list[tuple[float, float]]:
    """User's triangle: low/deep left, high/shallow middle, low/deep right."""
    q25, q75 = np.percentile(sup_x, [25, 75])
    left = np.where(sup_x <= q25)[0]
    center = np.where((sup_x >= q25) & (sup_x <= q75))[0]
    right = np.where(sup_x >= q75)[0]

    if robust:
        def low(indices: np.ndarray) -> tuple[float, float]:
            cutoff = np.percentile(sup_y[indices], 95)
            keep = indices[sup_y[indices] >= cutoff]
            return float(np.median(sup_x[keep])), float(np.median(sup_y[keep]))

        def high(indices: np.ndarray) -> tuple[float, float]:
            cutoff = np.percentile(sup_y[indices], 5)
            keep = indices[sup_y[indices] <= cutoff]
            return float(np.median(sup_x[keep])), float(np.median(sup_y[keep]))

        return [low(left), high(center), low(right)]

    li = left[np.argmax(sup_y[left])]
    ci = center[np.argmin(sup_y[center])]
    ri = right[np.argmax(sup_y[right])]
    return [
        (float(sup_x[li]), float(sup_y[li])),
        (float(sup_x[ci]), float(sup_y[ci])),
        (float(sup_x[ri]), float(sup_y[ri])),
    ]


def line_from_points(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    slope = (p2[1] - p1[1]) / max(p2[0] - p1[0], 1e-9)
    return float(slope), float(p1[1] - slope * p1[0])


def segment_points(points: list[tuple[float, float]]) -> list[dict]:
    return [
        {"x1": points[i][0], "y1": points[i][1], "x2": points[i + 1][0], "y2": points[i + 1][1]}
        for i in range(len(points) - 1)
    ]


def project_with_piecewise_top(
    frags: list[dict],
    points: list[tuple[float, float]],
    deep_line: tuple[float, float],
    px_per_mm: float,
) -> dict:
    lines = [line_from_points(points[0], points[1]), line_from_points(points[1], points[2])]
    bounds = [
        (min(points[0][0], points[1][0]), max(points[0][0], points[1][0])),
        (min(points[1][0], points[2][0]), max(points[1][0], points[2][0])),
    ]
    rows = []
    angs = []
    wts = []
    for frag in frags:
        candidates = []
        for line, (lo_x, hi_x) in zip(lines, bounds):
            up = M.line_intersection((frag["fs"], frag["fb"]), line)
            if up is None:
                continue
            on_segment = lo_x - 10.0 <= up[0] <= hi_x + 10.0
            candidates.append((0 if on_segment else 1, abs(up[0] - frag["cx"]), up, line))
        if not candidates:
            continue
        _, _, up, local_line = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        lo = M.line_intersection((frag["fs"], frag["fb"]), deep_line)
        if lo is None:
            continue
        fl_px = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        signed = np.degrees(np.arctan(frag["fs"]) - np.arctan(deep_line[0]))
        while signed <= -90:
            signed += 180
        while signed > 90:
            signed -= 180
        absang = abs(signed)
        if not (10.0 <= fl_px <= 4000.0 and M.FASC_MIN_ANG <= absang <= 75.0):
            continue
        rows.append({
            "fl_px": fl_px,
            "area": frag["area"],
            "visible_frac": frag["visible_px"] / max(fl_px, 1e-9),
            "fs": frag["fs"],
            "local_top_slope": local_line[0],
            "span": {"x1": up[0], "y1": up[1], "x2": lo[0], "y2": lo[1]},
        })
        angs.append(absang)
        wts.append(frag["area"])
    vals = np.asarray([row["fl_px"] / px_per_mm for row in rows], dtype=float)
    areas = np.asarray([row["area"] for row in rows], dtype=float)
    top_angles = np.asarray([angle_to_line(row["fs"], row["local_top_slope"]) for row in rows], dtype=float)
    return {
        "rows": rows,
        "vals": vals,
        "pa_deep": weighted_median(angs, wts) if len(angs) else None,
        "pa_top": weighted_median(top_angles, areas) if len(vals) else None,
        "support": float(np.median([row["visible_frac"] for row in rows])) if rows else None,
    }


def variant_stats(
    name: str,
    label: str,
    top_line: tuple[float, float],
    deep_line: tuple[float, float],
    frags: list[dict],
    px_per_mm: float,
    truth: dict,
    anchors: list[tuple[float, float]] | None = None,
) -> dict:
    projected = project_with_lines(frags, top_line, deep_line)
    vals = np.asarray([row["fl_px"] / px_per_mm for row in projected["rows"]], dtype=float)
    areas = np.asarray([row["area"] for row in projected["rows"]], dtype=float)
    top_angles = np.asarray([angle_to_line(row["fs"], top_line[0]) for row in projected["rows"]], dtype=float)
    pa_top = weighted_median(top_angles, areas) if vals.size else None
    spans = []
    for row in projected["rows"]:
        up = M.line_intersection((row["fs"], row["fb"]), top_line)
        lo = M.line_intersection((row["fs"], row["fb"]), deep_line)
        if up is not None and lo is not None:
            spans.append({"x1": up[0], "y1": up[1], "x2": lo[0], "y2": lo[1]})
    if vals.size:
        p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
        mean = float(np.mean(vals))
    else:
        p10 = p25 = p50 = p75 = p90 = mean = None
    return {
        "name": name,
        "label": label,
        "topLine": top_line,
        "line": None,
        "anchors": [{"x": p[0], "y": p[1]} for p in anchors or []],
        "projectionSpans": spans,
        "stats": {
            "n": int(vals.size),
            "paDeepDeg": projected["pa_deg"],
            "paTopDeg": pa_top,
            "flMeanMm": mean,
            "flP10Mm": None if p10 is None else float(p10),
            "flP25Mm": None if p25 is None else float(p25),
            "flMedianMm": None if p50 is None else float(p50),
            "flP75Mm": None if p75 is None else float(p75),
            "flP90Mm": None if p90 is None else float(p90),
            "medianSupport": projected["median_support"],
            "meanDeltaMm": None if mean is None else mean - truth["flMm"],
            "medianDeltaMm": None if p50 is None else float(p50) - truth["flMm"],
            "p25DeltaMm": None if p25 is None else float(p25) - truth["flMm"],
        },
    }


def piecewise_variant_stats(
    name: str,
    label: str,
    points: list[tuple[float, float]],
    deep_line: tuple[float, float],
    frags: list[dict],
    px_per_mm: float,
    truth: dict,
) -> dict:
    projected = project_with_piecewise_top(frags, points, deep_line, px_per_mm)
    vals = projected["vals"]
    if vals.size:
        p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
        mean = float(np.mean(vals))
    else:
        p10 = p25 = p50 = p75 = p90 = mean = None
    return {
        "name": name,
        "label": label,
        "topLine": None,
        "line": None,
        "polyline": segment_points(points),
        "anchors": [{"x": p[0], "y": p[1]} for p in points],
        "projectionSpans": [row["span"] for row in projected["rows"]],
        "stats": {
            "n": int(vals.size),
            "paDeepDeg": projected["pa_deep"],
            "paTopDeg": projected["pa_top"],
            "flMeanMm": mean,
            "flP10Mm": None if p10 is None else float(p10),
            "flP25Mm": None if p25 is None else float(p25),
            "flMedianMm": None if p50 is None else float(p50),
            "flP75Mm": None if p75 is None else float(p75),
            "flP90Mm": None if p90 is None else float(p90),
            "medianSupport": projected["support"],
            "meanDeltaMm": None if mean is None else mean - truth["flMm"],
            "medianDeltaMm": None if p50 is None else float(p50) - truth["flMm"],
            "p25DeltaMm": None if p25 is None else float(p25) - truth["flMm"],
        },
    }


def build_data() -> dict:
    bench_img = next(ROOT.glob(f"data/**/{IMAGE_ID}.tif"))
    rgb = M.read_rgb(bench_img)
    if rgb.dtype != np.uint8:
        im = rgb.astype(np.float32)
        rgb = (255 * (im - im.min()) / (im.max() - im.min() + 1e-9)).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    truth, _ = BV.load_truth()
    pred = pd.read_csv(ROOT / "results" / "benchmark_pred_truescale.csv")
    pred["ImageID"] = pred["image_id"].astype(str).str.replace(".tif", "", regex=False)
    row = truth.merge(pred, on="ImageID").query("ImageID == @IMAGE_ID").iloc[0]
    px_per_mm = float(row.scale_px_per_cm) / 10.0
    truth_vals = {
        "paDeg": float(row.pa_deg_true),
        "flMm": float(row.fl_mm_true),
        "mtMm": float(row.mt_mm_true),
    }

    apo = cv2.imread(str(MASK_DIR / f"{IMAGE_ID}_apo.png"), cv2.IMREAD_UNCHANGED)
    fasc = cv2.imread(str(MASK_DIR / f"{IMAGE_ID}_fasc.png"), cv2.IMREAD_UNCHANGED)
    apo_mask = (apo[:, :, 3] > 0).astype(np.uint8) if apo is not None and apo.ndim == 3 and apo.shape[2] == 4 else (apo > 0).astype(np.uint8)
    fasc_mask = (fasc[:, :, 3] > 0).astype(np.uint8) if fasc is not None and fasc.ndim == 3 and fasc.shape[2] == 4 else (fasc > 0).astype(np.uint8)

    geom = apo_geometry(apo_mask)
    if geom is None:
        raise RuntimeError("could not recover aponeurosis geometry")
    comps = connected(apo_mask, 5)
    top2 = sorted(sorted(comps, key=lambda c: c["area"], reverse=True)[:2], key=lambda c: c["mean_y"])
    sup_x, sup_y, _, sup_meta = fit_inner_edge(top2[0], "sup")
    deepest_line, deepest_left, deepest_right = deepest_half_line(sup_x, sup_y)
    robust_line, robust_left, robust_right = deepest_percentile_line(sup_x, sup_y)
    tri_points = triangle_points(sup_x, sup_y, robust=False)
    tri_robust_points = triangle_points(sup_x, sup_y, robust=True)
    q_chord = geom["sup_chord_line"]
    q_left = (sup_meta["left_x"], sup_meta["left_y"])
    q_right = (sup_meta["right_x"], sup_meta["right_y"])
    deep_line = geom["deep_line"]
    frags = fragments(fasc_mask, geom["sup_line"], deep_line)

    variants = [
        piecewise_variant_stats("triangle", "Your triangle: low-left / high-middle / low-right", tri_points, deep_line, frags, px_per_mm, truth_vals),
        piecewise_variant_stats("triangleRobust", "Robust triangle: deepest 5% / highest 5% / deepest 5%", tri_robust_points, deep_line, frags, px_per_mm, truth_vals),
        variant_stats("deepest", "Your exact deepest-left/right line", deepest_line, deep_line, frags, px_per_mm, truth_vals, [deepest_left, deepest_right]),
        variant_stats("baseline", "Current all-edge top fit", geom["sup_line"], deep_line, frags, px_per_mm, truth_vals),
        variant_stats("qchord", "My earlier q25/q75 median chord", q_chord, deep_line, frags, px_per_mm, truth_vals, [q_left, q_right]),
        variant_stats("robustdeep", "Robust deepest 5% left/right", robust_line, deep_line, frags, px_per_mm, truth_vals, [robust_left, robust_right]),
    ]
    for variant in variants:
        if variant["topLine"] is not None:
            variant["line"] = line_points(variant["topLine"], w)
    return {
        "imageId": IMAGE_ID,
        "width": w,
        "height": h,
        "scalePxPerMm": px_per_mm,
        "imageUrl": data_url_image(bgr, ".jpg"),
        "apoMaskUrl": data_url_image((apo_mask * 180).astype(np.uint8), ".png"),
        "fascMaskUrl": data_url_image((fasc_mask * 180).astype(np.uint8), ".png"),
        "deepLine": line_points(deep_line, w),
        "truth": truth_vals,
        "variants": variants,
    }


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>im_29 Boundary Viewer</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      background: #111;
      color: #eee;
      overflow: hidden;
    }
    #top {
      height: 50px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: #1f1f1f;
      border-bottom: 1px solid #333;
      white-space: nowrap;
    }
    button, label {
      border: 1px solid #444;
      background: #2a2a2a;
      color: #f2f2f2;
      border-radius: 6px;
      padding: 6px 8px;
      font: inherit;
    }
    label { display: inline-flex; align-items: center; gap: 5px; }
    button { cursor: pointer; }
    button.active { background: #12405b; outline: 2px solid #69b8ff; }
    #wrap {
      height: calc(100vh - 50px);
      display: grid;
      grid-template-columns: 1fr 390px;
      min-width: 0;
    }
    #stage {
      overflow: auto;
      background: #050505;
      position: relative;
    }
    #canvasWrap {
      position: relative;
      width: max-content;
      height: max-content;
      margin: 18px;
    }
    canvas {
      display: block;
      transform-origin: top left;
      image-rendering: auto;
    }
    #side {
      overflow: auto;
      background: #1b1b1b;
      border-left: 1px solid #333;
      padding: 12px;
    }
    h2 { font-size: 17px; margin: 0 0 10px; }
    .muted { color: #aaa; }
    .section { border-top: 1px solid #333; padding-top: 12px; margin-top: 12px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
    .statGrid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat {
      border: 1px solid #383838;
      background: #222;
      border-radius: 7px;
      padding: 8px;
    }
    .stat b { display: block; font-size: 18px; color: #fff; }
    .stat span { font-size: 12px; color: #aaa; }
    .warn {
      border: 1px solid #755;
      background: #2b1717;
      border-radius: 7px;
      padding: 8px;
      line-height: 1.35;
    }
    .legendLine {
      display: inline-block;
      width: 28px;
      height: 4px;
      margin-right: 8px;
      vertical-align: middle;
      border-radius: 99px;
    }
  </style>
</head>
<body>
  <div id="top">
    <strong>im_29 boundary viewer</strong>
    <button id="fit">Fit</button>
    <button id="zout">-</button>
    <button id="z100">100%</button>
    <button id="zin">+</button>
    <span class="muted">No text is drawn over the image. Stats are on the side.</span>
  </div>
  <div id="wrap">
    <main id="stage"><div id="canvasWrap"><canvas id="view"></canvas></div></main>
    <aside id="side">
      <h2>Boundary Variant</h2>
      <div id="variantButtons" class="row"></div>
      <div class="section">
        <div class="row">
          <label><input type="checkbox" id="showBaseline" checked>baseline top</label>
          <label><input type="checkbox" id="showCandidate" checked>candidate top</label>
          <label><input type="checkbox" id="showDeep" checked>lower boundary</label>
          <label><input type="checkbox" id="showProj" checked>projected spans</label>
          <label><input type="checkbox" id="showAnchors" checked>anchor points</label>
          <label><input type="checkbox" id="showMasks">masks</label>
        </div>
      </div>
      <div id="stats" class="section"></div>
      <div class="section">
        <div><span class="legendLine" style="background:#bdbdbd"></span>baseline all-edge top fit</div>
        <div><span class="legendLine" style="background:#63ff6c"></span>selected candidate top line</div>
        <div><span class="legendLine" style="background:#2d8cff"></span>lower boundary</div>
        <div><span class="legendLine" style="background:#00e7ff"></span>projected spans for selected line</div>
        <div><span class="legendLine" style="background:#ff3b30"></span>candidate anchor points</div>
      </div>
    </aside>
  </div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const base = new Image();
const apo = new Image();
const fasc = new Image();
let zoom = 1;
let selected = 'triangle';
base.src = DATA.imageUrl;
apo.src = DATA.apoMaskUrl;
fasc.src = DATA.fascMaskUrl;
canvas.width = DATA.width;
canvas.height = DATA.height;

function fmt(v, d=2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : 'n/a';
}
function signed(v, d=2) {
  const n = Number(v);
  return Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(d) : 'n/a';
}
function variant() {
  return DATA.variants.find(v => v.name === selected) || DATA.variants[0];
}
function line(l, color, width=3) {
  if (!l) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width / zoom;
  ctx.beginPath();
  ctx.moveTo(l.x1, l.y1);
  ctx.lineTo(l.x2, l.y2);
  ctx.stroke();
  ctx.restore();
}
function polyline(segments, color, width=4) {
  for (const s of segments || []) segment(s, color, width);
}
function segment(s, color, width=1.2) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width / zoom;
  ctx.beginPath();
  ctx.moveTo(s.x1, s.y1);
  ctx.lineTo(s.x2, s.y2);
  ctx.stroke();
  ctx.restore();
}
function point(p, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2 / zoom;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 7 / zoom, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}
function drawMask(img, color) {
  const temp = document.createElement('canvas');
  temp.width = DATA.width;
  temp.height = DATA.height;
  const t = temp.getContext('2d');
  t.drawImage(img, 0, 0);
  const mask = t.getImageData(0, 0, temp.width, temp.height);
  const out = t.createImageData(temp.width, temp.height);
  for (let i = 0; i < mask.data.length; i += 4) {
    const a = mask.data[i];
    if (a > 0) {
      out.data[i] = color[0];
      out.data[i + 1] = color[1];
      out.data[i + 2] = color[2];
      out.data[i + 3] = 95;
    }
  }
  t.putImageData(out, 0, 0);
  ctx.drawImage(temp, 0, 0);
}
function renderStats() {
  const v = variant();
  const s = v.stats;
  const truth = DATA.truth;
  document.getElementById('stats').innerHTML = `
    <h2>${v.label}</h2>
      <div class="warn">
      Default candidate is your triangle: low/deep left, high/shallow middle, low/deep right.
      The q25/q75 chord is kept only as a comparison because it was my mistaken conservative version.
    </div>
    <div class="statGrid" style="margin-top:10px">
      <div class="stat"><b>${fmt(s.flMeanMm,2)} mm</b><span>average FL, delta ${signed(s.meanDeltaMm,2)}</span></div>
      <div class="stat"><b>${fmt(s.flMedianMm,2)} mm</b><span>median FL, delta ${signed(s.medianDeltaMm,2)}</span></div>
      <div class="stat"><b>${fmt(s.flP25Mm,2)} mm</b><span>p25 FL, delta ${signed(s.p25DeltaMm,2)}</span></div>
      <div class="stat"><b>${fmt(truth.flMm,2)} mm</b><span>expert FL</span></div>
      <div class="stat"><b>${fmt(s.paDeepDeg,2)} deg</b><span>PA vs lower/code</span></div>
      <div class="stat"><b>${fmt(s.paTopDeg,2)} deg</b><span>PA vs selected top line</span></div>
      <div class="stat"><b>${fmt(s.medianSupport,3)}</b><span>median visible/full support</span></div>
      <div class="stat"><b>${s.n}</b><span>projected fragments</span></div>
    </div>
    <div class="section muted">
      Scale: ${fmt(DATA.scalePxPerMm,3)} px/mm. Expert PA ${fmt(truth.paDeg,2)} deg, expert MT ${fmt(truth.mtMm,2)} mm.
    </div>`;
}
function renderButtons() {
  const box = document.getElementById('variantButtons');
  box.innerHTML = '';
  for (const v of DATA.variants) {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.className = v.name === selected ? 'active' : '';
    b.onclick = () => { selected = v.name; renderButtons(); draw(); };
    box.appendChild(b);
  }
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(base, 0, 0);
  if (document.getElementById('showMasks').checked) {
    drawMask(apo, [255, 70, 120]);
    drawMask(fasc, [255, 230, 70]);
  }
  const v = variant();
  const baseline = DATA.variants.find(x => x.name === 'baseline');
  if (document.getElementById('showBaseline').checked && baseline) line(baseline.line, '#bdbdbd', 2.5);
  if (document.getElementById('showDeep').checked) line(DATA.deepLine, '#2d8cff', 4);
  if (document.getElementById('showProj').checked) for (const s of v.projectionSpans) segment(s, '#00e7ff', 1.4);
  if (document.getElementById('showCandidate').checked) {
    if (v.polyline) polyline(v.polyline, '#63ff6c', 4);
    else line(v.line, '#63ff6c', 4);
  }
  if (document.getElementById('showAnchors').checked) for (const p of v.anchors) point(p, '#ff3b30');
  renderStats();
}
function applyZoom() {
  canvas.style.width = Math.round(DATA.width * zoom) + 'px';
  canvas.style.height = Math.round(DATA.height * zoom) + 'px';
  document.getElementById('canvasWrap').style.width = canvas.style.width;
  document.getElementById('canvasWrap').style.height = canvas.style.height;
  document.getElementById('z100').textContent = Math.round(zoom * 100) + '%';
  draw();
}
function fitZoom() {
  const stage = document.getElementById('stage');
  zoom = Math.max(0.25, Math.min(2.5, (stage.clientWidth - 40) / DATA.width, (stage.clientHeight - 40) / DATA.height));
  applyZoom();
}
document.getElementById('fit').onclick = fitZoom;
document.getElementById('zin').onclick = () => { zoom = Math.min(4, zoom * 1.25); applyZoom(); };
document.getElementById('zout').onclick = () => { zoom = Math.max(0.2, zoom / 1.25); applyZoom(); };
document.getElementById('z100').onclick = () => { zoom = 1; applyZoom(); };
for (const id of ['showBaseline','showCandidate','showDeep','showProj','showAnchors','showMasks']) {
  document.getElementById(id).onchange = draw;
}
base.onload = () => { renderButtons(); fitZoom(); };
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    data: dict = {}

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_response(404)
            self.end_headers()
            return
        html = HTML.replace("__DATA__", json.dumps(self.data))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()
    Handler.data = build_data()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"im_29 boundary viewer: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
