"""Does our segmentation actually generalize to the test set? Measure mask quality directly
(not from image stats) by running the real models on stratified groups and comparing.

Groups:
  TRAIN-fasc : in-distribution training frames (control, should be best)
  BENCHMARK  : the 35 expert images (clean, known-good control)
  TEST-<dim> : the 309 test frames, split by canvas size (the scale families)

Per image we record mask-quality proxies that don't need ground truth:
  apo:  n_bands (components > 60px; a good prediction finds ~2), apo_area_frac
  fasc: n_frag (fascicle fragments), fasc_area_frac
  geom: did measure() return a valid pennation angle?

If TEST mask quality matches TRAIN/BENCHMARK, segmentation generalizes and the gap is elsewhere
(FL method / scale). If it drops -- and on which families -- that localizes the real failure.

    python experiments/seg_quality_test.py
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TRAIN_FASC = ROOT / "data" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
TEST_DIR = ROOT / "data" / "test_images_v2" / "test_set_v2"


def load(target):
    w = ROOT / "results" / f"seg_{target}.pt"
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(w, map_location="cpu"))
    return m.eval().to(M.DEVICE)


def mask_stats(mask, min_area=60):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    big = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(big), float(m.mean())


def analyze(apo, fasc, paths, label, rows):
    for p in paths:
        img = M.read_rgb(p)
        am = M.predict_mask(apo, img)
        fm = M.predict_mask(fasc, img)
        n_bands, apo_frac = mask_stats(am)
        n_frag, fasc_frac = mask_stats(fasc_min_area_mask(fm))
        try:
            g = M.measure(am, fm)
            ok = bool(g and g.get("pa_deg") is not None)
            pa = float(g["pa_deg"]) if ok else np.nan
        except Exception:
            ok, pa = False, np.nan
        rows.append(dict(group=label, n_bands=n_bands, apo_frac=apo_frac,
                         n_frag=n_frag, fasc_frac=fasc_frac, geom_ok=ok, pa=pa))


def fasc_min_area_mask(fm):
    return fm  # mask_stats already applies min_area


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    print(f"\n{'group':16} {'n':>4} {'apo_bands':>9} {'apo%':>6} {'frags':>6} {'fasc%':>6} "
          f"{'geom_ok':>8} {'pa_mean':>7}")
    for label, rs in groups.items():
        n = len(rs)
        ab = np.mean([r["n_bands"] for r in rs])
        af = np.mean([r["apo_frac"] for r in rs]) * 100
        fr = np.mean([r["n_frag"] for r in rs])
        ff = np.mean([r["fasc_frac"] for r in rs]) * 100
        ok = np.mean([r["geom_ok"] for r in rs]) * 100
        pa = np.nanmean([r["pa"] for r in rs])
        # fraction with the "healthy" 2 apo bands and at least one fascicle fragment
        print(f"{label:16} {n:>4} {ab:>9.2f} {af:>6.2f} {fr:>6.2f} {ff:>6.2f} {ok:>7.0f}% {pa:>7.2f}")


def main():
    rng = np.random.default_rng(0)
    apo, fasc = load("apo"), load("fasc")
    rows = []

    # control 1: in-distribution training frames
    tr = sorted(p for p in TRAIN_FASC.iterdir() if p.suffix.lower() in EXTS)
    tr = [tr[i] for i in rng.choice(len(tr), 40, replace=False)]
    analyze(apo, fasc, tr, "TRAIN-fasc", rows)

    # control 2: the 35 benchmark images
    bench_dir = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench_dir:
        bench = sorted(p for p in bench_dir.iterdir() if p.suffix.lower() in EXTS)
        analyze(apo, fasc, bench, "BENCHMARK", rows)

    # test set, split by canvas size (family), stratified sample per family
    test = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTS)
    by_dim = defaultdict(list)
    for p in test:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            by_dim[g.shape].append(p)
    for dim, ps in sorted(by_dim.items(), key=lambda kv: -len(kv[1])):
        take = ps if len(ps) <= 40 else [ps[i] for i in rng.choice(len(ps), 40, replace=False)]
        analyze(apo, fasc, take, f"TEST-{dim[1]}x{dim[0]}", rows)

    summarize(rows)
    print("\ncontrols (TRAIN-fasc, BENCHMARK) set the ceiling; a TEST family with fewer apo bands,")
    print("fewer fragments, or lower geom_ok is where segmentation is actually failing.")


if __name__ == "__main__":
    main()
