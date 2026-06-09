"""Experiment 11: how much of the test set can we actually calibrate?

Finding that reframed the scale problem: the 251 "unscaled" TIFFs are mostly the SAME 800x1200 full-UI
format as the 58 PNGs we already calibrate - choose_candidate just gates the good png_left_ruler reader
to filenames ending .png, so ~181 identical TIFFs fall through to the weak generic detector.

(A) Validate detection accuracy vs the 35 benchmark images' TRUE scale.
(B) Coverage on all 309 test images, by shape family, gated (current) vs ungated (try png_left_ruler
    on everything first). CPU, read-only.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import tick_calibration as TC  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"


def detect(gray, name, ungated):
    if ungated:
        c = TC.png_left_ruler_candidate(gray, 5.0)
        if c is not None:
            return c
        cands = (TC.side_candidates(gray, "left", 5.0) + TC.side_candidates(gray, "right", 5.0)
                 + TC.bottom_candidates(gray, 10.0))
        return max(cands, key=lambda c: c.score) if cands else None
    return TC.choose_candidate(gray, 5.0, 10.0, image_name=name)


def main():
    # (A) benchmark accuracy vs true scale
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    print("=== (A) benchmark accuracy (true scale known) ===")
    ratios = []
    for _, r in truth.iterrows():
        g = cv2.imread(str(bench / f"{r.ImageID}.tif"), cv2.IMREAD_GRAYSCALE)
        c = detect(g, r.ImageID + ".tif", ungated=True)
        if c is not None and c.confidence >= 0.4:
            implied = c.px_per_mm * 10.0
            ratios.append((float(r.scale_px_per_cm), implied, c.method))
    if ratios:
        arr = np.array([(t, d) for t, d, _ in ratios])
        rr = arr[:, 0] / arr[:, 1]
        print(f"detected {len(ratios)}/35 (conf>=0.4). implied px/cm vs true:")
        print(f"  ratio true/implied: median {np.median(rr):.2f} mean {np.mean(rr):.2f} std {np.std(rr):.2f}")
        for mult in (1.0, 2.0, 0.5):
            mae = np.abs(arr[:, 0] - arr[:, 1] * mult).mean()
            print(f"  if implied x{mult}: MAE vs true {mae:.1f} px/cm")

    # (B) coverage on all 309 test images, by shape
    print("\n=== (B) test coverage by shape: gated (current) vs ungated ===")
    files = sorted(TEST.iterdir())
    byshape = defaultdict(lambda: {"n": 0, "gated": [], "ungated": []})
    for p in files:
        g = TC.read_gray(p)
        s = g.shape
        rec = byshape[s]
        rec["n"] += 1
        cg = detect(g, p.name, ungated=False)
        cu = detect(g, p.name, ungated=True)
        if cg is not None and cg.confidence >= 0.5:
            rec["gated"].append(cg.px_per_mm * 10)
        if cu is not None and cu.confidence >= 0.5:
            rec["ungated"].append(cu.px_per_mm * 10)
    print(f"{'shape':>14} {'n':>4} {'gatedOK':>8} {'ungatedOK':>10} {'ungated px/cm median':>22}")
    tot_g = tot_u = 0
    for s, rec in sorted(byshape.items(), key=lambda kv: -kv[1]["n"]):
        med = f"{np.median(rec['ungated']):.1f}" if rec["ungated"] else "--"
        print(f"{str(s):>14} {rec['n']:>4} {len(rec['gated']):>8} {len(rec['ungated']):>10} {med:>22}")
        tot_g += len(rec["gated"]); tot_u += len(rec["ungated"])
    print(f"{'TOTAL':>14} {len(files):>4} {tot_g:>8} {tot_u:>10}   (conf>=0.5)")


if __name__ == "__main__":
    main()
