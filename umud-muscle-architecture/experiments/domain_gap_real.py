"""Run the domain_probe diagnostics on REAL train vs test frames (not synthetic).

Answers, from global image statistics only (no GPU, no training):
  1. GAP       - is there an intensity/contrast gap between training frames and the 309 test frames?
  2. DISCRETE? - do test frames form their own cluster(s) (-> route per-class) or mix into train
                 (-> normalize/augment one model)?
  3. FREE FIX? - does any per-image normalization close the gap, or is it structural (-> augment)?
  4. BY FAMILY - break the test set down by canvas size (the scale families) and show which families
                 are in-distribution vs out-of-distribution relative to training.

    python experiments/domain_gap_real.py
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
BRIEF = Path.home() / "Desktop" / "scale-brief"
sys.path.insert(0, str(BRIEF))
import domain_probe as DP  # noqa: E402

TRAIN_DIR = ROOT / "data" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
TEST_DIR = ROOT / "data" / "test_images_v2" / "test_set_v2"
SIZE = (384, 256)  # (w, h) common size so train/test features are comparable
N_TRAIN = 400
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_gray(p, native=False):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if native:
        return g
    return cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)


def main():
    rng = np.random.default_rng(0)
    train_files = sorted(p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in EXTS)
    test_files = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTS)
    sel = rng.choice(len(train_files), min(N_TRAIN, len(train_files)), replace=False)
    train_files = [train_files[i] for i in sel]
    print(f"train frames: {len(train_files)} (sampled)   test frames: {len(test_files)}")

    train_imgs = [g for g in (load_gray(p) for p in train_files) if g is not None]
    # capture native dims for the test set to tag families
    test_imgs, test_dims = [], []
    for p in test_files:
        nat = load_gray(p, native=True)
        if nat is None:
            continue
        test_dims.append(nat.shape)  # (h, w)
        test_imgs.append(cv2.resize(nat, SIZE, interpolation=cv2.INTER_AREA))

    ftr = [DP.image_features(g) for g in train_imgs]
    fte = [DP.image_features(g) for g in test_imgs]

    print("\n=== 1) RAW train->test gap (standardized mean diff; |smd|>1 = real domain axis) ===")
    cg = DP.compare_groups(ftr, fte)
    for k, v in cg.items():
        flag = "  <== GAP" if abs(v["smd"]) > 1 else ("  <- moderate" if abs(v["smd"]) > 0.5 else "")
        print(f"   {k:8} train {v['train']:8.2f}  test {v['test']:8.2f}  smd {v['smd']:+5.2f}{flag}")
    print(f"   mean|smd| over all features: {DP.mean_abs_smd(ftr, fte):.2f}")

    print("\n=== 2) Family clustering (k=5): do test frames isolate (DISCRETE) or mix (continuum)? ===")
    Xtr, Xte = DP._matrix(ftr), DP._matrix(fte)
    X = np.vstack([Xtr, Xte])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lab = DP._kmeans(Xs, 5)
    ntr = len(Xtr)
    lab_tr, lab_te = lab[:ntr], lab[ntr:]
    for c in range(5):
        a, b = int((lab_tr == c).sum()), int((lab_te == c).sum())
        tag = ""
        if b and not a:
            tag = "  <== TEST-ONLY (out-of-distribution)"
        elif a and not b:
            tag = "  (train-only)"
        elif b and a:
            frac = b / (a + b)
            tag = f"  (mixed; {frac:.0%} test)"
        print(f"   cluster {c}: train {a:4d}  test {b:4d}{tag}")

    print("\n=== 3) Normalization sweep (mean|smd|, lower = gap closed; >~0.8 = structural) ===")
    sweep = DP.normalization_sweep(train_imgs, test_imgs)
    for m, v in sorted(sweep.items(), key=lambda kv: kv[1]):
        print(f"   {m:13} {v:.2f}")
    best = min(sweep, key=sweep.get)
    print(f"   best: {best} ({sweep[best]:.2f}) | raw {sweep['raw']:.2f}")

    print("\n=== 4) Test set by canvas size (scale family), each vs TRAIN baseline ===")
    dim_counts = Counter(test_dims)
    by_dim = defaultdict(list)
    for f, d in zip(fte, test_dims):
        by_dim[d].append(f)
    tr_mean = {k: np.mean([f[k] for f in ftr]) for k in DP.FEATS}
    tr_sd = {k: (np.std([f[k] for f in ftr]) + 1e-9) for k in DP.FEATS}
    print(f"   train baseline:  mean {tr_mean['mean']:.1f}  std {tr_mean['std']:.1f}  "
          f"p95 {tr_mean['p95']:.1f}  grad_x {tr_mean['grad_x']:.2f}")
    for d, cnt in dim_counts.most_common():
        fs = by_dim[d]
        mu = {k: np.mean([f[k] for f in fs]) for k in DP.FEATS}
        smd_mean = (mu["mean"] - tr_mean["mean"]) / tr_sd["mean"]
        smd_p95 = (mu["p95"] - tr_mean["p95"]) / tr_sd["p95"]
        smd_gx = (mu["grad_x"] - tr_mean["grad_x"]) / tr_sd["grad_x"]
        ood = "  <== far from train" if (abs(smd_mean) > 1 or abs(smd_p95) > 1) else ""
        print(f"   {d[1]}x{d[0]:<5} n={cnt:3d}  mean {mu['mean']:6.1f} (smd {smd_mean:+.2f})  "
              f"p95 {mu['p95']:6.1f} (smd {smd_p95:+.2f})  grad_x smd {smd_gx:+.2f}{ood}")

    print("\n   read: a family whose mean/p95 smd is large is a brightness/gain-shifted device;")
    print("   a family whose grad_x smd is very negative is lower-detail (blur/lower freq) content.")


if __name__ == "__main__":
    main()
