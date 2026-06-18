"""Residual scorer for the typed-correction UI. SINGLE ENGINE: recomputes PA/FL/MT from geometry with
the same helpers as production measure(), so the residual between the pipeline pre-fill and the human
correction is attributable, not an artifact of a second engine.

Corrections are applied CUMULATIVELY in pipeline order so each channel's marginal contribution to the
total error (pipeline baseline vs fully-corrected) is isolated:
    1 scale  -> 2 apo lines -> 3 fragment reject -> 4 fragment add -> 5 fragment angle

corrections record (per image, what the UI logs), all keys optional:
    {"scale_px_per_mm": float,
     "apo": {"superficial_coef":[s,b], "deep_coef":[s,b]},
     "reject": ["F1", ...],
     "add": [{"p1":[x,y], "p2":[x,y]}, ...],
     "angle": {"F2": slope, ...}}   # slope = image-space dy/dx of the corrected fascicle line

Usage:
    from correction_score import score_corrections
    out = score_corrections(prefill_dict, corrections_dict)   # per-image stages + decomposition
    python correction_score.py        # aggregate over results/correction_prefill + corrections/
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M

PREFILL_DIR = ROOT / "results" / "correction_prefill"
CORR_DIR = ROOT / "results" / "correction_labels"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
STAGES = ["scale", "apo", "reject", "add", "angle"]


def _line_from_slope_point(slope, cx, cy):
    return (float(slope), float(cy - slope * cx))


def _angle_to_deep(fs, deep_s):
    a = abs(math.degrees(math.atan(fs)) - math.degrees(math.atan(deep_s)))
    return 180 - a if a > 90 else a


def resolve(prefill, corr, upto):
    """Resolve geometry after applying correction channels up to and including `upto` stages."""
    g = prefill["geometry"]
    k = STAGES.index(upto) if upto else -1

    px = prefill["scale"].get("px_per_mm")
    if k >= 0 and corr.get("scale_px_per_mm"):
        px = float(corr["scale_px_per_mm"])

    sup_coef = list(g["apo"]["superficial_coef"])
    deep_coef = list(g["apo"]["deep_coef"])
    top_boundary = g["top_boundary"]
    if k >= 1 and corr.get("apo"):
        sup_coef = [float(v) for v in corr["apo"].get("superficial_coef", sup_coef)]
        deep_coef = [float(v) for v in corr["apo"].get("deep_coef", deep_coef)]
        if corr["apo"].get("top_boundary") and corr["apo"]["top_boundary"].get("points"):
            top_boundary = {"type": "piecewise",
                            "points": [[float(x), float(y)] for x, y in corr["apo"]["top_boundary"]["points"]]}
        else:
            top_boundary = {"type": "line", "mode": "line",
                            "points": [[0.0, sup_coef[1]], [float(g["width"]), sup_coef[0] * g["width"] + sup_coef[1]]],
                            "line": sup_coef}
    # normalize a piecewise/line top_boundary into the {"line": (s,b)} shape measure() helpers expect
    if top_boundary.get("type") != "piecewise" and "line" not in top_boundary:
        p = top_boundary["points"]
        s = (p[1][1] - p[0][1]) / max(p[1][0] - p[0][0], 1e-9)
        top_boundary = {**top_boundary, "line": (float(s), float(p[0][1] - s * p[0][0]))}
    if top_boundary.get("type") == "piecewise":
        top_boundary = {**top_boundary, "lines": [
            M.line_from_points(top_boundary["points"][0], top_boundary["points"][1]),
            M.line_from_points(top_boundary["points"][1], top_boundary["points"][2])]}

    kept = [dict(fr) for fr in g["fragments"] if fr.get("kept")]
    areas = [fr["area"] for fr in kept] or [200]
    default_area = float(np.median(areas))
    if k >= 2 and corr.get("reject"):
        rej = set(corr["reject"])
        kept = [fr for fr in kept if fr["id"] not in rej]
    frags = [{"slope": fr["slope"], "cx": fr["centroid"][0], "cy": fr["centroid"][1],
              "area": fr["area"], "visible_len": fr.get("visible_len", 0.0), "id": fr["id"]} for fr in kept]
    if k >= 3 and corr.get("add"):
        for j, a in enumerate(corr["add"]):
            p1, p2 = a["p1"], a["p2"]
            slope = (p2[1] - p1[1]) / max((p2[0] - p1[0]), 1e-9) if abs(p2[0] - p1[0]) > 1e-9 else 1e6
            frags.append({"slope": float(slope), "cx": (p1[0] + p2[0]) / 2.0, "cy": (p1[1] + p2[1]) / 2.0,
                          "area": default_area, "visible_len": math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
                          "id": f"ADD{j}"})
    if k >= 4 and corr.get("angle"):
        for fr in frags:
            if fr["id"] in corr["angle"]:
                fr["slope"] = float(corr["angle"][fr["id"]])
    return px, sup_coef, deep_coef, top_boundary, frags


def recompute(prefill, px, sup_coef, deep_coef, top_boundary, frags):
    """PA/FL/MT in mm from resolved geometry, using production helpers and conventions."""
    g = prefill["geometry"]
    deep_line = (float(deep_coef[0]), float(deep_coef[1]))
    deep_s = deep_line[0]
    xc = g["width"] / 2.0
    angs, wts, frag_rows = [], [], []
    for fr in frags:
        line = _line_from_slope_point(fr["slope"], fr["cx"], fr["cy"])
        a = _angle_to_deep(fr["slope"], deep_s)
        upper = M.top_boundary_intersection(line, top_boundary, xref=fr["cx"])
        lower = M.line_intersection(line, deep_line)
        angs.append(a)
        wts.append(max(1.0, fr["area"]))
        if upper is not None and lower is not None:
            fl = float(math.hypot(upper[0] - lower[0], upper[1] - lower[1]))
            if 10.0 <= fl <= 4000.0:
                frag_rows.append({"fl": fl, "area": fr["area"], "visible_len": fr["visible_len"],
                                  "visible_frac": float(np.clip(fr["visible_len"] / (fl + 1e-9), 0.0, 1.0))})
    pa = M.weighted_median(angs, wts) if angs else None
    fl_px, _, _ = M.aggregate_fragment_fl(frag_rows)
    mt_px = abs(M.line_y(deep_line, xc) - M.top_boundary_y(top_boundary, xc)) / math.sqrt(1 + deep_s ** 2)

    pa_out = float(np.clip(pa if pa is not None else M.PRIOR["pa_deg"], M.PA_MIN, M.PA_MAX))
    mt_mm = float(np.clip(mt_px / px, M.MT_MIN, M.MT_MAX)) if px else M.PRIOR["mt_mm"]
    if px and fl_px:
        fl_mm = float(np.clip(fl_px / px, M.FL_MIN, M.FL_MAX))
    elif pa is not None:
        fl_mm = float(np.clip(mt_mm / math.sin(math.radians(pa_out)), M.FL_MIN, M.FL_MAX))
    else:
        fl_mm = M.PRIOR["fl_mm"]
    return {"pa_deg": round(pa_out, 3), "fl_mm": round(fl_mm, 3), "mt_mm": round(mt_mm, 3)}


def _norm_dist(a, b):
    per = {t: abs(a[t] - b[t]) / TOL[t] for t in TOL}
    per["overall"] = sum(per.values()) / 3.0
    return per


def score_corrections(prefill, corr):
    """Per-image cumulative stages + per-channel error decomposition (single engine)."""
    if not prefill.get("geometry"):
        return {"error": "no geometry (measure returned None); cannot score"}
    baseline = recompute(prefill, *resolve(prefill, {}, None))   # recompute from pure pre-fill = parity check
    stage_out, prev = {}, baseline
    contrib = {t: {} for t in ["pa_deg", "fl_mm", "mt_mm", "overall"]}
    full = recompute(prefill, *resolve(prefill, corr, STAGES[-1]))
    remaining_prev = _norm_dist(baseline, full)
    for st in STAGES:
        out = recompute(prefill, *resolve(prefill, corr, st))
        stage_out[st] = out
        remaining = _norm_dist(out, full)
        for t in contrib:
            contrib[t][st] = round(remaining_prev[t] - remaining[t], 5)
        remaining_prev = remaining
        prev = out
    return {"baseline": baseline, "corrected": full,
            "baseline_vs_corrected": _norm_dist(baseline, full),
            "stages": stage_out, "contribution": contrib,
            "parity_ok": abs(baseline["pa_deg"] - prefill["derived"]["pa_deg"]) < 0.01}


def main():
    if not CORR_DIR.exists():
        print(f"no corrections at {CORR_DIR}; nothing to aggregate")
        return
    agg = {t: {st: 0.0 for st in STAGES} for t in ["pa_deg", "fl_mm", "mt_mm", "overall"]}
    n = 0
    for cf in sorted(CORR_DIR.glob("*.json")):
        corr = json.loads(cf.read_text(encoding="utf-8"))
        pf = PREFILL_DIR / cf.name
        if not pf.exists():
            continue
        res = score_corrections(json.loads(pf.read_text(encoding="utf-8")), corr)
        if "error" in res:
            continue
        n += 1
        for t in agg:
            for st in STAGES:
                agg[t][st] += res["contribution"][t][st]
    print(f"aggregated {n} corrected images")
    print("per-channel mean contribution to total error (normalized tolerances):")
    for t in agg:
        line = "  %-8s " % t + "  ".join(f"{st}={agg[t][st]/max(n,1):+.4f}" for st in STAGES)
        print(line)


if __name__ == "__main__":
    main()
