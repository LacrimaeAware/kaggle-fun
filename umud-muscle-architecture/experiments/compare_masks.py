"""Compare OUR predicted masks vs the HOST ground-truth masks on training images. Everything is
resized to a common square (the host's stated alignment), so the overlays actually line up. Prints
per-image: how many apo bands the host labels vs how many we predict, and mask overlap (IoU). Saves
apo overlays (host=yellow, ours=cyan) to results/mask_compare/.

    python umud-muscle-architecture/experiments/compare_masks.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

AI = ROOT / "data/apo_imgs_v1/apo_images_new_model_v1"
AM = ROOT / "data/apo_masks_v1/apo_masks_new_model_v1"
FM = ROOT / "data/fasc_masks_v1/fasc_masks_new_model_v1"
OUT = ROOT / "results" / "mask_compare"
S = 512


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def nbands(m, minarea=150):
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return sum(1 for i in range(1, n) if st[i, 4] >= minarea)


def main():
    apo, fasc = load("apo"), load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in AI.iterdir() if p.suffix.lower() in (".tif", ".png", ".jpg"))
    sample = files[:: max(1, len(files) // 16)][:16]
    print(f"{'image':16} {'host_apo_bands':>14} {'our_apo_bands':>13} {'apo_IoU':>8} {'fasc_IoU':>9}")
    rows = []
    for p in sample:
        img = M.read_rgb(p)
        gt_a = cv2.imread(str(AM / p.name), 0)
        gt_f = cv2.imread(str(FM / p.name), 0)
        if gt_a is None or gt_f is None:
            continue
        gt_a = cv2.resize(gt_a, (S, S)) > 127
        gt_f = cv2.resize(gt_f, (S, S)) > 127
        our_a = cv2.resize(np.asarray(M.predict_mask(apo, img), np.uint8), (S, S)) > 0
        our_f = cv2.resize(np.asarray(M.predict_mask(fasc, img), np.uint8), (S, S)) > 0
        gb, ob = nbands(gt_a), nbands(our_a)
        ai, fi = iou(gt_a, our_a), iou(gt_f, our_f)
        rows.append((gb, ob, ai, fi))
        print(f"{p.stem:16} {gb:>14} {ob:>13} {ai:>8.2f} {fi:>9.2f}")
        # overlay: host apo = yellow, our apo = cyan, on the squared grayscale image
        im = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (S, S))
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        ov = vis.copy(); ov[gt_a] = (0, 220, 255); ov[our_a] = (255, 220, 0)
        vis = cv2.addWeighted(ov, 0.5, vis, 0.5, 0)
        cv2.putText(vis, f"{p.stem}  host={gb} bands (yellow)  ours={ob} (cyan)  IoU={ai:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, f"{p.stem}  host={gb} bands (yellow)  ours={ob} (cyan)  IoU={ai:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
        cv2.imwrite(str(OUT / f"{p.stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    a = np.array(rows, float)
    print(f"\nSUMMARY over {len(a)} images:")
    print(f"  host apo bands: almost always 2?  ->  values seen: {sorted(set(int(x) for x in a[:,0]))}")
    print(f"  OUR apo bands: mean {a[:,1].mean():.1f}  | we predict >2 bands on {int((a[:,1]>2).sum())}/{len(a)} images")
    print(f"  apo IoU (our vs host): mean {a[:,2].mean():.2f}  | fascicle IoU: mean {a[:,3].mean():.2f}")
    print(f"  overlays -> {OUT}")


if __name__ == "__main__":
    main()
