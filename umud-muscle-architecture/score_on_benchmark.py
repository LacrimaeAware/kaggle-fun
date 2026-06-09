"""Run our trained U-Nets on the 35 expert-benchmark images and score PA/FL/MT vs the experts.

Uses the TRUE per-image scale (from the xlsx) to convert pixels->mm. That isolates our
segmentation + geometry quality from calibration: if we are far from DL-Track's 0.331 even with
perfect scale, the bottleneck is masks/geometry; if we are close, calibration is the remaining gap.
CPU only. Needs results/seg_apo.pt and results/seg_fasc.pt.

Caveat: these images are different devices than our training data, so a poor score may reflect
cross-device domain shift rather than the approach itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
import segment_then_measure as M  # geometry + predict_mask + read_rgb + PRIOR (resolves local data on import)
import benchmark_validate as BV


def load(target):
    w = HERE / "results" / f"seg_{target}.pt"
    if not w.exists():
        raise SystemExit(f"missing {w} - move the Kaggle weights into results/")
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(w, map_location="cpu"))
    return m.eval().to(M.DEVICE)


def main():
    truth, floor = BV.load_truth()
    bench_dir = next((p.parent for p in HERE.glob("data/**/im_01_arch.tif")), None)
    if bench_dir is None:
        raise SystemExit("benchmark images not found under data/")
    apo, fasc = load("apo"), load("fasc")
    print(f"scoring our pipeline on {len(truth)} benchmark images from {bench_dir.name} ...")

    rows, geom_ok = [], 0
    for _, r in truth.iterrows():
        img = M.read_rgb(bench_dir / f"{r.ImageID}.tif")
        try:
            geom = M.measure(M.predict_mask(apo, img), M.predict_mask(fasc, img))
        except Exception as e:
            print(f"  measure failed {r.ImageID}: {e}")
            geom = None
        ppm = float(r.scale_px_per_cm) / 10.0  # TRUE pixels per mm
        pa = geom["pa_deg"] if geom and geom["pa_deg"] is not None else M.PRIOR["pa_deg"]
        fl = geom["fl_px"] / ppm if geom and geom["fl_px"] is not None else M.PRIOR["fl_mm"]
        mt = geom["mt_px"] / ppm if geom and geom["mt_px"] is not None else M.PRIOR["mt_mm"]
        if geom and geom["pa_deg"] is not None:
            geom_ok += 1
        rows.append({"image_id": r.ImageID, "pa_deg": float(np.clip(pa, 5, 45)),
                     "fl_mm": float(fl), "mt_mm": float(mt)})
    pred = pd.DataFrame(rows)
    pred.to_csv(HERE / "results" / "benchmark_pred_truescale.csv", index=False)
    rs = BV.score(pred, truth)

    print(f"\n=== OUR pipeline on the 35 benchmark images (TRUE scale, apo geometry ok on {geom_ok}/35) ===")
    print(f"  overall {rs['overall']:.3f}  (pa {rs['pa_deg']:.3f}, fl {rs['fl_mm']:.3f}, mt {rs['mt_mm']:.3f})")
    print("\n  references (same 35, same metric):")
    print(f"    human floor 0.307 | DL-Track 0.331 | SMA 0.409 | our constant-prior 0.923")
    print("\n  read: PA/FL/MT terms above show which stage is the bottleneck even with perfect scale.")


if __name__ == "__main__":
    main()
