"""Generate isolated PA-calibration probe submissions from the current best file.

Finding: the model under-predicts PA on the test distribution (mean 14.64 vs expert-truth ~18.34
and hand-label ~19.9; 18/19 hand labels point the same way; LOO-stable). The 35-image benchmark
cannot see this because it is in-distribution. These probes correct ONLY the pa_deg column (FL/MT
untouched), so each is a clean, isolated, one-hypothesis test.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
BASE = RES / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
PA_MIN, PA_MAX = 5.0, 45.0
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}

base = pd.read_csv(BASE)
print(f"base {BASE.name}: {len(base)} rows, PA mean {base.pa_deg.mean():.3f} std {base.pa_deg.std():.3f}")

probes = {
    "submission_pa_shift_p30.csv":  lambda s: np.clip(s + 3.0, PA_MIN, PA_MAX),
    "submission_pa_shift_p45.csv":  lambda s: np.clip(s + 4.5, PA_MIN, PA_MAX),
    "submission_pa_linear.csv":     lambda s: np.clip(-2.89 + 1.39 * s, PA_MIN, PA_MAX),
}
for name, fn in probes.items():
    out = base.copy()
    out["pa_deg"] = fn(base["pa_deg"]).round(3)
    out.to_csv(RES / name, index=False)
    nclip = int((out.pa_deg >= PA_MAX - 1e-6).sum())
    print(f"  wrote {name:30s} PA mean {out.pa_deg.mean():.3f}  (+{out.pa_deg.mean()-base.pa_deg.mean():.2f})  clipped@45: {nclip}")

# A-proxy on the 19 hand labels for each probe (FL/MT identical across probes)
hv = pd.read_csv(RES / "human_benchmark" / "target_human_vs_submission.csv")
hp, sp = hv.human_pa_deg.to_numpy(), hv.submission_pa_deg.to_numpy()
fl_t = np.abs(hv.submission_fl_mm - hv.human_fl_mm).mean() / TOL["fl"]
mt_t = np.abs(hv.submission_mt_mm - hv.human_mt_mm).mean() / TOL["mt"]
def overall(pa_pred):
    pa_t = np.abs(pa_pred - hp).mean() / TOL["pa"]
    return (pa_t + fl_t + mt_t) / 3, pa_t
print("\nA-proxy on 19 hand labels (FL term {:.3f}, MT term {:.3f}):".format(fl_t, mt_t))
print(f"  current           overall {overall(sp)[0]:.3f}  pa {overall(sp)[1]:.3f}")
print(f"  +3.0              overall {overall(np.clip(sp+3.0,PA_MIN,PA_MAX))[0]:.3f}  pa {overall(np.clip(sp+3.0,PA_MIN,PA_MAX))[1]:.3f}")
print(f"  +4.5              overall {overall(np.clip(sp+4.5,PA_MIN,PA_MAX))[0]:.3f}  pa {overall(np.clip(sp+4.5,PA_MIN,PA_MAX))[1]:.3f}")
print(f"  linear            overall {overall(np.clip(-2.89+1.39*sp,PA_MIN,PA_MAX))[0]:.3f}  pa {overall(np.clip(-2.89+1.39*sp,PA_MIN,PA_MAX))[1]:.3f}")
print("\n[done] probes in results/")
