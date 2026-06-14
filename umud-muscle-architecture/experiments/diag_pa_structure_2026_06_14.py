"""Structure of the PA bias - 2026-06-14.

The flat +2deg PA shift improved the public LB 0.58910 -> 0.55075, confirming a real net
under-prediction. But a flat shift cannot distinguish a GENUINELY low-pennation muscle (model
correct) from a HEDGED prediction (model regressing to the prior because it is uncertain). This
script asks which structure the bias actually has, so the next correction can be targeted:

  (A) UNIFORM      : err independent of PA and of confidence -> a flat shift is already right.
  (B) COMPRESSION  : err grows with predicted PA (model shrinks the range toward the mean)
                     -> the fix is de-shrinkage pa -> c + k*(pa-c), not a flat add.
  (C) CONFIDENCE   : err concentrated where the model is uncertain (few fragments / parked near the
                     prior 15.1) -> correct hedged rows more, confident rows less. THIS is the
                     "detect genuine-vs-systematic" signal the user wants.

Confidence proxy = model fascicle fragment count (fl_fragment_n) and distance from the prior.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
PRIOR_PA = 15.105
pd.set_option("display.width", 140)


def stripext(s): return s.astype(str).str.replace(r"\.(tif|tiff|png|jpg|jpeg|bmp)$", "", regex=True)


def main():
    hv = pd.read_csv(RES / "human_benchmark" / "target_human_vs_submission.csv")
    calib = pd.read_csv(RES / "calibration_measurement_debug.csv")
    hv["id"] = stripext(hv["image_id"]); calib["id"] = stripext(calib["image_id"])
    # attach the MODEL fragment count (calib) to the 19 labels
    m = hv.merge(calib[["id", "fl_fragment_n", "pa_deg"]], on="id", how="left", suffixes=("", "_calib"))
    m["err"] = m["human_pa_deg"] - m["submission_pa_deg"]   # +ve = model under-predicts
    n = m["err"].notna().sum()
    print(f"=== 19 hand labels (n usable {n}) : what predicts the PA under-prediction? ===")
    print(f"mean err {m.err.mean():+.2f}  median {m.err.median():+.2f}")
    for f, lbl in [("submission_pa_deg", "predicted PA  (compression test)"),
                   ("fl_fragment_n", "model fragment count (confidence test)")]:
        sub = m[["err", f]].dropna()
        r = sub["err"].corr(sub[f])
        b, a = np.polyfit(sub[f], sub["err"], 1)
        print(f"  corr(err, {lbl:38s}) = {r:+.3f}   slope err ~ {a:+.2f} {b:+.3f}*x")
    # near-prior (hedged) vs not, within the 19
    m["hedged"] = (m["submission_pa_deg"] - PRIOR_PA).abs() < 2.5
    print("\n  err by hedged(|pred-15.1|<2.5) vs confident:")
    print(m.groupby("hedged")["err"].agg(["mean", "median", "count"]).to_string())
    # err by fragment bucket
    m["fragbucket"] = pd.cut(m["fl_fragment_n"], [0, 8, 16, 999], labels=["<=8", "9-16", "17+"])
    print("\n  err by model-fragment-count bucket:")
    print(m.groupby("fragbucket")["err"].agg(["mean", "median", "count"]).to_string())

    print("\n=== 309 test set: is the low-PA cluster hedged (uncertain) or confident? ===")
    c = calib.copy()
    print(f"PA: mean {c.pa_deg.mean():.2f} std {c.pa_deg.std():.2f} | fragment_n: median {c.fl_fragment_n.median():.0f}")
    print(f"corr(pa_deg, fragment_n) = {c.pa_deg.corr(c.fl_fragment_n):+.3f}  "
          "(if +ve: low-PA preds have FEWER fragments => low PA tracks uncertainty, supports correcting them up)")
    c["near_prior"] = (c.pa_deg - PRIOR_PA).abs() < 2.0
    print(f"near-prior rows (|pa-15.1|<2): {int(c.near_prior.sum())}/309  "
          f"mean fragment_n {c[c.near_prior].fl_fragment_n.mean():.1f} vs others {c[~c.near_prior].fl_fragment_n.mean():.1f}")
    for lo, hi in [(0, 11), (11, 15), (15, 20), (20, 99)]:
        seg = c[(c.pa_deg >= lo) & (c.pa_deg < hi)]
        print(f"  PA [{lo:2d},{hi:2d}): n={len(seg):3d}  mean frag {seg.fl_fragment_n.mean():5.1f}  "
              f"by family: " + ", ".join(f"{k}:{v}" for k, v in seg.calibration_method.value_counts().head(3).items()))

    # What the compression model implies as a per-row correction vs the flat +2 already shipped
    print("\n=== implied corrections (informational; LB is the judge) ===")
    b, a = np.polyfit(m.dropna(subset=["err"]).submission_pa_deg, m.dropna(subset=["err"]).err, 1)
    print(f"compression fit on 19: err ~ {a:+.2f} {b:+.3f}*pred  => correction grows with PA")
    c["corr_flat2"] = 2.0
    c["corr_compress"] = (a + b * c.pa_deg).clip(-3, 8)
    print("per-row correction summary over 309:")
    print(f"  flat+2:     constant +2.00")
    print(f"  compression: min {c.corr_compress.min():+.2f}  mean {c.corr_compress.mean():+.2f}  max {c.corr_compress.max():+.2f}")
    print(f"  rows where compression correction < +1 (model trusted): {int((c.corr_compress < 1).sum())}")
    print(f"  rows where compression correction > +3 (model lifted hard): {int((c.corr_compress > 3).sum())}")


if __name__ == "__main__":
    main()
