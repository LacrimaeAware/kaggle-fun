"""Honest end-to-end validator on the 35-image expert benchmark.

Unlike experiments/score_weights.py, this does NOT recenter predicted FL to the truth mean. It runs
the FULL production pipeline (segment -> measure -> configured FL path) and converts px->mm with the
benchmark's TRUE scale.

Why true scale here: the benchmark .tif images use a different on-image tick convention (~1 cm,
77-126 px/cm) than the Kaggle test set (134-150 px/cm), so the test-set scale router constants do
NOT transfer. True scale is the only fair px->mm conversion on these images.

What this CATCHES that score_weights.py / the old gate could not:
  - the raw FL geometry bias. score_weights.py:54 multiplied predicted FL by (truth_mean/pred_mean),
    which divides the global FL error out. That global error is the exact degree of freedom the
    leaderboard FL x1.05+ win exploited. With the recenter removed, that bias is visible here.

What this STILL cannot catch:
  - test-set scale-recovery error (benchmark tick convention differs from the Kaggle test families),
  - test-distribution shift (RF muscle, Philips Lumify device, cerebral-palsy subjects are test-only).
So this is the per-image gate for MEASUREMENT/GEOMETRY correctness, not a leaderboard predictor for
scale. It is the instrument that would have flagged the FL undershoot before we spent LB slots on it.

Usage:
    python benchmark_lab/honest_validate.py                 # default weights
    python benchmark_lab/honest_validate.py apo.pt fasc.pt  # specific weights
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402  (honest scorer: no recenter)
import segment_then_measure as M  # noqa: E402

TOL = BV.TOL
apo_path = sys.argv[1] if len(sys.argv) > 1 else str(M.weights_path("apo"))
fasc_path = sys.argv[2] if len(sys.argv) > 2 else str(M.weights_path("fasc"))


def load(path):
    m = M.build_model(encoder_weights=None)
    m.load_state_dict(M.checkpoint_state(torch.load(path, map_location="cpu")))
    return m.eval().to(M.DEVICE)


def main():
    truth, floor = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench is None:
        raise SystemExit("benchmark images not found under data/**/im_01_arch.tif")
    apo, fasc = load(apo_path), load(fasc_path)
    print(f"apo  = {apo_path}\nfasc = {fasc_path}")
    print(f"benchmark: {len(truth)} images | recenter=OFF | scale=TRUE (from xlsx)\n")

    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = M.predict_mask(apo, img, "apo")
        fm = M.predict_mask(fasc, img, "fasc")
        nfrag = int(cv2.connectedComponentsWithStats((fm > 0).astype(np.uint8))[0] - 1)
        g = M.measure(am, fm)
        ppm = float(r.scale_px_per_cm) / 10.0  # px per mm (true)
        if g and g["pa_deg"]:
            pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX))
            mt = float(np.clip(g["mt_px"] / ppm, M.MT_MIN, M.MT_MAX))
            if M.USE_FRAGMENT_FL and g.get("fl_px"):
                fl = float(np.clip(g["fl_px"] / ppm, M.FL_MIN, M.FL_MAX))
            else:
                fl = float(np.clip(mt / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
            measured = True
        else:
            pa, mt, fl = M.PRIOR["pa_deg"], M.PRIOR["mt_mm"], M.PRIOR["fl_mm"]
            measured = False
        rows.append({"image_id": r.ImageID, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt,
                     "n_frag": nfrag, "measured": measured})
    pred = pd.DataFrame(rows)

    # honest per-image scoring (BV.score does NOT recenter)
    res = BV.score(pred, truth)
    print("RAW pipeline (no recenter, true scale) vs expert consensus:")
    print(f"  overall {res['overall']:.4f}   pa {res['pa_deg']:.4f}   fl {res['fl_mm']:.4f}   "
          f"mt {res['mt_mm']:.4f}   (n={res['n']})")

    # references
    hf = float(np.mean([floor[c] / TOL[c] for c in BV.TGT]))
    print(f"\nreferences (tol-normalized MAE, lower better):")
    print(f"  human floor (expert vs rest): {hf:.4f}   <- irreducible noise; do not chase below this")
    print(f"  DL-Track (true scale):        {BV._tool_score(truth, '_dlt'):.4f}")
    print(f"  SMA:                          {BV._tool_score(truth, '_sma'):.4f}")

    # the bias the old gate hid: what recenter factor would score_weights.py have applied?
    m = truth.merge(pred.assign(ImageID=pred.image_id), on="ImageID")
    for c in BV.TGT:
        pm, tm = m[c].mean(), m[c + "_true"].mean()
        signed = (m[c] - m[c + "_true"]).mean()
        print(f"\n{c}: pred mean {pm:7.2f}  truth mean {tm:7.2f}  signed bias {signed:+7.2f}  "
              f"recenter factor truth/pred = {tm/pm:5.3f}")
    print("  (recenter factor != 1.0 is the global bias score_weights.py:54 silently removed.)")

    # per-image worst offenders, to point at structure (variance), not bias
    m["err_pa"] = (m["pa_deg"] - m["pa_deg_true"]).abs() / TOL["pa_deg"]
    m["err_fl"] = (m["fl_mm"] - m["fl_mm_true"]).abs() / TOL["fl_mm"]
    m["err_mt"] = (m["mt_mm"] - m["mt_mm_true"]).abs() / TOL["mt_mm"]
    m["err_tot"] = m[["err_pa", "err_fl", "err_mt"]].mean(axis=1)
    worst = m.sort_values("err_tot", ascending=False).head(10)
    print("\nworst 10 images (where the structural error lives):")
    print(f"  {'image':<14}{'tot':>6}{'pa':>6}{'fl':>6}{'mt':>6}  {'frag':>4}  detail")
    for _, w in worst.iterrows():
        nf = int(pred.loc[pred.image_id == w.ImageID, "n_frag"].iloc[0])
        print(f"  {w.ImageID:<14}{w.err_tot:6.2f}{w.err_pa:6.2f}{w.err_fl:6.2f}{w.err_mt:6.2f}  "
              f"{nf:>4}  pa {w.pa_deg:5.1f}/{w.pa_deg_true:5.1f}  fl {w.fl_mm:5.1f}/{w.fl_mm_true:5.1f}  "
              f"mt {w.mt_mm:5.1f}/{w.mt_mm_true:5.1f}")

    out = ROOT / "results" / "honest_benchmark_pred.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"\nwrote per-image table -> {out}")


if __name__ == "__main__":
    main()
