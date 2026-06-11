"""Pretend we do NOT know the true scale. Recover it the way we must on the test set (OCR + tick
detection), compute FL/MT in mm from the RECOVERED scale, and score vs experts. Compare to the same
geometry scored with the TRUE scale. The gap = what scale-recovery error actually costs in the metric.

This is the honest test of 'is scale our problem': same images, same segmentation, same FL/MT geometry,
only the scale differs (true vs what-we'd-recover). Prints progress and a clear breakdown.

    python experiments/bench_scale_cost.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV       # noqa: E402
import segment_then_measure as M      # noqa: E402
import scale_ocr as SO                # noqa: E402
import scale_ticks as ST              # noqa: E402

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def recover_tick(gray):
    best = None
    for fn in (ST.recover_scale, ST.recover_scale_left_ruler, ST.recover_scale_right_ruler):
        try:
            d = fn(gray)
        except Exception:
            d = None
        if d and (best is None or d["conf"] > best["conf"]):
            best = d
    return best


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    reader = SO.get_reader()
    print(f"scoring {len(truth)} benchmark images: TRUE scale vs RECOVERED scale\n", flush=True)
    rows = []
    for i, r in enumerate(truth.itertuples()):
        p = bench / f"{r.ImageID}.tif"
        if not p.exists():
            continue
        im = M.read_rgb(p)                                   # original orientation for segmentation
        g = M.measure(M.predict_mask(apo, im), M.predict_mask(fasc, im))
        if not g or not g.get("fl_px") or not g.get("mt_px"):
            print(f"  {r.ImageID}: no geometry, skipped", flush=True)
            continue
        flip = cv2.flip(cv2.imread(str(p)), 1)               # un-mirror for the UI text
        grayf = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)
        s = SO.read_scale(flip, reader)
        rec, how = None, "PRIOR"
        if s["px_per_mm"]:
            rec, how = s["px_per_mm"] * 10.0, "ocr-ruler"
        else:
            tk = recover_tick(grayf)
            if tk and 40 <= tk["scale_px_per_cm"] <= 220:
                rec, how = tk["scale_px_per_cm"], "tick"
        rows.append(dict(id=r.ImageID, fl_px=g["fl_px"], mt_px=g["mt_px"], pa=g["pa_deg"],
                         true=float(r.scale_px_per_cm), rec=rec, how=how,
                         pa_t=r.pa_deg_true, fl_t=r.fl_mm_true, mt_t=r.mt_mm_true))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(truth)} measured", flush=True)

    # ---- scale-recovery accuracy ----
    got = [x for x in rows if x["rec"]]
    errs = [abs(x["rec"] - x["true"]) / x["true"] for x in got]
    print(f"\nrecovered a scale on {len(got)}/{len(rows)} images "
          f"({sum(1 for x in rows if x['how']=='ocr-ruler')} ocr, {sum(1 for x in rows if x['how']=='tick')} tick, "
          f"{sum(1 for x in rows if x['rec'] is None)} fell to PRIOR)")
    if errs:
        print(f"scale error |rec-true|/true: median {100*np.median(errs):.0f}%  mean {100*np.mean(errs):.0f}%  "
              f"max {100*np.max(errs):.0f}%")

    # ---- score FL/MT with true vs recovered scale (per-image, NO mean recenter) ----
    pa_t = np.mean([abs(x["pa"] - x["pa_t"]) for x in rows]) / TOL["pa_deg"]

    def scored(use_rec):
        fl_e, mt_e = [], []
        for x in rows:
            scm = x["rec"] if (use_rec and x["rec"]) else x["true"]
            ppm = scm / 10.0
            if use_rec and not x["rec"]:
                fl_mm, mt_mm = M.PRIOR["fl_mm"], M.PRIOR["mt_mm"]   # test-set behaviour: no scale -> prior
            else:
                fl_mm, mt_mm = x["fl_px"] / ppm, x["mt_px"] / ppm
            fl_e.append(abs(fl_mm - x["fl_t"]))
            mt_e.append(abs(mt_mm - x["mt_t"]))
        return np.mean(fl_e) / TOL["fl_mm"], np.mean(mt_e) / TOL["mt_mm"]

    fl_true, mt_true = scored(False)
    fl_rec, mt_rec = scored(True)
    print(f"\n=== per-image FL/MT (no mean recenter), PA={pa_t:.3f} both ===")
    print(f"  TRUE scale:       FL {fl_true:.3f}  MT {mt_true:.3f}   overall {(pa_t+fl_true+mt_true)/3:.3f}")
    print(f"  RECOVERED scale:  FL {fl_rec:.3f}  MT {mt_rec:.3f}   overall {(pa_t+fl_rec+mt_rec)/3:.3f}")
    print(f"  COST of scale recovery error:  FL +{fl_rec-fl_true:.3f}  MT +{mt_rec-mt_true:.3f}  "
          f"overall +{((pa_t+fl_rec+mt_rec)-(pa_t+fl_true+mt_true))/3:.3f}")
    print("\n(if the cost is near 0, our scale recovery is good enough and scale is NOT the bottleneck;")
    print(" if the cost is large, scale recovery error is directly hurting FL/MT.)")


if __name__ == "__main__":
    main()
