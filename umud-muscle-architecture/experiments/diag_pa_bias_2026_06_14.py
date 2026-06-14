"""PA-bias decisive diagnostic - 2026-06-14.

The 19 hand labels show the live submission under-predicts PA by ~3.5deg (a systematic bias the
35-image benchmark hides). Before betting a submission on a PA correction, disambiguate:
  - Is the model also PA-biased vs the EXPERT benchmark (real model bias / range compression)?
  - Or is the bias only vs the rough hand labels (label artifact -> correcting it is the overfit trap)?
Also quantify what a constant shift and a linear correction would do to the A-proxy (19 labels),
and re-run the depth-based scale sanity with real image heights.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"; DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
TOL = {"pa": 6.0, "fl": 12.0, "mt": 3.0}
np.set_printoptions(suppress=True)


def banner(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def aproxy(pa_t, pa_p, fl_t, fl_p, mt_t, mt_p):
    pa = np.abs(pa_p - pa_t).mean() / TOL["pa"]
    fl = np.abs(fl_p - fl_t).mean() / TOL["fl"]
    mt = np.abs(mt_p - mt_t).mean() / TOL["mt"]
    return (pa + fl + mt) / 3, pa, fl, mt


def expert_bias():
    banner("EXPERT BENCHMARK PA BIAS (model vs the real raters, true scale)")
    try:
        import benchmark_validate as BV
        truth, _ = BV.load_truth()
        truth = truth.rename(columns={"ImageID": "image_id"})
    except Exception as e:
        print("load_truth failed:", e); return
    pred = pd.read_csv(RES / "benchmark_pred_truescale.csv")
    m = truth.merge(pred, on="image_id", suffixes=("_true", "_pred"))
    print(f"matched {len(m)} benchmark images")
    for t in ["pa", "fl", "mt"]:
        tc, pc = f"{t}_deg_true" if t == "pa" else f"{t}_mm_true", f"{t}_deg" if t == "pa" else f"{t}_mm"
        if tc not in m or pc not in m:
            # column naming may differ; try generic
            tc = [c for c in m.columns if c.startswith(t) and c.endswith("true")]
            pc = [c for c in m.columns if c.startswith(t) and not c.endswith("true")]
            tc, pc = (tc[0] if tc else None), (pc[0] if pc else None)
        if tc and pc:
            err = m[pc] - m[tc]
            print(f"{t.upper():3s} bias(pred-true)={err.mean():+7.3f}  MAE={err.abs().mean():7.3f}  norm_MAE={err.abs().mean()/TOL[t]:.3f}")
    # PA range compression on benchmark
    if "pa_deg_true" in m and "pa_deg" in m:
        b, a = np.polyfit(m["pa_deg"], m["pa_deg_true"], 1)
        print(f"\nPA range: truth ~ {a:.2f} + {b:.2f}*pred   (b<1 => model OVER-spreads; b>1 => model COMPRESSES toward mean)")
        print(f"pred PA mean {m['pa_deg'].mean():.2f} std {m['pa_deg'].std():.2f} | truth PA mean {m['pa_deg_true'].mean():.2f} std {m['pa_deg_true'].std():.2f}")


def handlabel_pa_fix():
    banner("TEST HAND LABELS (n=19): PA bias structure + correction A-proxy")
    d = pd.read_csv(RES / "human_benchmark" / "target_human_vs_submission.csv")
    hp, sp = d["human_pa_deg"].to_numpy(), d["submission_pa_deg"].to_numpy()
    hf, sf = d["human_fl_mm"].to_numpy(), d["submission_fl_mm"].to_numpy()
    hm, sm = d["human_mt_mm"].to_numpy(), d["submission_mt_mm"].to_numpy()
    diff = hp - sp
    print(f"PA (human-submission): mean {diff.mean():+.3f}  median {np.median(diff):+.3f}  "
          f"sign: {int((diff>0).sum())} human>sub / {int((diff<0).sum())} human<sub  (of {len(diff)})")
    print(f"submission PA mean {sp.mean():.2f} std {sp.std():.2f} | human PA mean {hp.mean():.2f} std {hp.std():.2f}")
    b, a = np.polyfit(sp, hp, 1)
    print(f"linear: human_pa ~ {a:.2f} + {b:.2f}*sub_pa  (b>1 => submission compresses PA range)")

    base = aproxy(hp, sp, hf, sf, hm, sm)
    print(f"\nA-proxy current:                 overall {base[0]:.3f}  (pa {base[1]:.3f} fl {base[2]:.3f} mt {base[3]:.3f})")
    dconst = np.median(diff)
    c = aproxy(hp, sp + dconst, hf, sf, hm, sm)
    print(f"A-proxy + const shift {dconst:+.2f}:      overall {c[0]:.3f}  (pa {c[1]:.3f} fl {c[2]:.3f} mt {c[3]:.3f})")
    lin = aproxy(hp, a + b * sp, hf, sf, hm, sm)
    print(f"A-proxy + linear correction:     overall {lin[0]:.3f}  (pa {lin[1]:.3f} fl {lin[2]:.3f} mt {lin[3]:.3f})")

    # leave-one-out stability of the constant-shift gain (is n=19 enough to trust the direction?)
    gains = []
    for i in range(len(diff)):
        keep = np.ones(len(diff), bool); keep[i] = False
        dlo = np.median(diff[keep])
        g0 = np.abs(sp[keep] - hp[keep]).mean() / TOL["pa"]
        g1 = np.abs(sp[keep] + dlo - hp[keep]).mean() / TOL["pa"]
        gains.append(g0 - g1)
    gains = np.array(gains)
    print(f"\nLOO PA-term gain from const shift: mean {gains.mean():+.3f}  min {gains.min():+.3f}  "
          f"(all positive => robust direction)  shift range {np.median(diff):+.2f}")


def scale_sanity_heights():
    banner("D1b SCALE SANITY VIA DEPTH (with real image heights)")
    import cv2
    calib = pd.read_csv(RES / "calibration_measurement_debug.csv")
    notes = json.load(open(RES / "scale_oracle_review" / "oracle_notes.json"))
    folder = next((p for p in DATA.glob("test_images*") if p.is_dir()), None)
    print("test image folder:", folder)
    def H(i):
        f = folder / i
        if not f.exists():
            f = folder / (Path(i).stem + ".tif")
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        return float(im.shape[0]) if im is not None else np.nan
    def depth(i):
        try: return float(str(notes.get(i, {}).get("oracle_depth_mm", "")).strip())
        except Exception: return np.nan
    calib["H"] = calib["image_id"].map(H)
    calib["depth_mm"] = calib["image_id"].map(depth)
    calib["field_frac"] = (calib["px_per_mm"] * calib["depth_mm"]) / calib["H"]
    calib["family"] = calib["calibration_method"].fillna("none")
    print("rows with height:", int(calib["H"].notna().sum()), " sample H:", calib["H"].dropna().head(3).tolist())
    g = calib.groupby("family").agg(n=("image_id", "size"),
        ffrac_med=("field_frac", "median"), ffrac_min=("field_frac", "min"), ffrac_max=("field_frac", "max"))
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    imp = calib[(calib["field_frac"] > 1.02) | (calib["field_frac"] < 0.20)].dropna(subset=["field_frac"])
    print(f"\nimpossible (>1.02) or tiny (<0.20) field_frac rows: {len(imp)}")
    if len(imp): print(imp[["image_id","family","px_per_mm","depth_mm","H","field_frac","mt_mm","fl_mm"]].sort_values("field_frac").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    # within-family robust outliers
    rows=[]
    for fam, sub in calib.dropna(subset=["field_frac"]).groupby("family"):
        med=sub["field_frac"].median(); mad=(sub["field_frac"]-med).abs().median() or 1e-9
        z=0.6745*(sub["field_frac"]-med)/mad; out=sub[z.abs()>3.5]
        for _,r in out.iterrows(): rows.append((r["image_id"],fam,r["px_per_mm"],r["depth_mm"],r["field_frac"],med))
    print(f"\nwithin-family field_frac outliers (|z|>3.5): {len(rows)}")
    for r in rows: print(f"  {r[0]:16s} {r[1]:22s} ppm={r[2]:.2f} depth={r[3]:.0f} ffrac={r[4]:.3f} fam_med={r[5]:.3f}")


if __name__ == "__main__":
    expert_bias()
    handlabel_pa_fix()
    scale_sanity_heights()
    print("\n[done]")
