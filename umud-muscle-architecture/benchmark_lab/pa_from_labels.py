"""Measure PA from the user's HAND-DRAWN fascicles only.

For each label file in a labels dir, take the fascicles the user drew (the 'add' lines, or the single
'blind_angle_line'), measure each one's angle to the DEEP aponeurosis (the user's corrected apo if they
edited it, else the pre-fill apo), and report the median PA next to the rater ground truth. This is the
user's own ground-truth check, independent of the pipeline.

    python umud-muscle-architecture/benchmark_lab/pa_from_labels.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LABELS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "correction_labels_gm"
PREFILL = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "correction_prefill_gm"
GM = ROOT / "data/gm_dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0"


def line_ang(p1, p2):
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], float)
    if v[0] < 0:
        v = -v                                  # orient rightward so the sign is consistent
    return np.degrees(np.arctan2(v[1], v[0]))


def pa_vs_deep(fasc_ang, deep_ang):
    return abs(((fasc_ang - deep_ang + 90) % 180) - 90)


def deep_angle(corr, prefill):
    apo = corr.get("apo")
    if apo and apo.get("deep_coef"):
        return np.degrees(np.arctan(apo["deep_coef"][0]))          # user-corrected deep apo
    d = prefill["geometry"]["apo"]["deep"]                          # else pre-fill deep apo (2 endpoints)
    return line_ang(d[0], d[1])


def fascicles(corr):
    lines = [(a["p1"], a["p2"]) for a in corr.get("add", [])]
    if corr.get("blind_angle_line"):
        b = corr["blind_angle_line"]; lines.append((b["p1"], b["p2"]))
    return lines


def main():
    truth = pd.read_excel(list(GM.glob("*.xlsx"))[0], sheet_name="PA")
    truth = truth[np.isfinite(truth["frame"])].copy(); truth["frame"] = truth["frame"].astype(int)
    tmean = dict(zip(truth["frame"], truth["Mean"]))
    files = sorted(LABELS.glob("frame_*.json"))
    if not files:
        print("no label files in", LABELS); return
    print(f"hand-measured PA vs rater truth  (labels: {LABELS.name})")
    print(f"  {'frame':<14}{'n_fasc':>7}{'hand PA (median)':>18}{'per-fascicle':>34}{'rater':>8}")
    for lf in files:
        rec = json.loads(lf.read_text(encoding="utf-8")); corr = rec.get("corrections", {})
        pf = PREFILL / lf.name
        prefill = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {"geometry": {"apo": {"deep": [[0, 0], [1, 0]]}}}
        da = deep_angle(corr, prefill)
        lines = fascicles(corr)
        if not lines:
            print(f"  {lf.stem:<14}{'0':>7}   (no fascicles drawn)"); continue
        pas = [pa_vs_deep(line_ang(p1, p2), da) for p1, p2 in lines]
        fr = int(lf.stem.split("_")[1]); rt = tmean.get(fr, float("nan"))
        per = " ".join(f"{p:.0f}" for p in pas)
        print(f"  {lf.stem:<14}{len(pas):>7}{np.median(pas):>18.1f}{per:>34}{rt:>8.1f}")


if __name__ == "__main__":
    main()
