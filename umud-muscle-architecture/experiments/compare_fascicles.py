"""Compare OUR fascicle masks vs HOST fascicle masks, using the host's stated alignment (resize both
to a square) and a metric that's fair to thin sparse dashes: dilate both, then measure COVERAGE
(what fraction of the host's fascicles we hit, and vice versa) instead of exact-pixel IoU.
Overlays: host fascicle = green, ours = red, on the aligned image. -> results/fasc_compare/

    python umud-muscle-architecture/experiments/compare_fascicles.py
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
FM = ROOT / "data/fasc_masks_v1/fasc_masks_new_model_v1"
OUT = ROOT / "results" / "fasc_compare"
S = 640


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def main():
    fasc = load("fasc")
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in AI.iterdir() if p.suffix.lower() == ".tif")
    sample = files[:: max(1, len(files) // 12)][:12]
    if (AI / "image_0910.tif").exists() and AI / "image_0910.tif" not in sample:
        sample.append(AI / "image_0910.tif")
    k = max(1, int(S * 0.012))
    ker = np.ones((k, k), np.uint8)
    print(f"dash-tolerance dilation = {k}px")
    print(f"{'image':16} {'host_dashes':>11} {'our_dashes':>10} {'we_cover_host%':>14} {'host_covers_our%':>16}")
    rows = []
    for p in sample:
        gt = cv2.imread(str(FM / p.name), 0)
        if gt is None:
            continue
        img = M.read_rgb(p)
        gt = (cv2.resize(gt, (S, S)) > 127).astype(np.uint8)
        our = (cv2.resize(np.asarray(M.predict_mask(fasc, img), np.uint8), (S, S)) > 0).astype(np.uint8)
        gd, od = cv2.dilate(gt, ker), cv2.dilate(our, ker)
        inter = ((gd > 0) & (od > 0)).sum()
        cover_host = inter / max(1, (gd > 0).sum())      # of host fascicle area, how much we hit
        cover_our = inter / max(1, (od > 0).sum())       # of our fascicle area, how much they label
        ng = cv2.connectedComponentsWithStats(gt, 8)[0] - 1
        no = cv2.connectedComponentsWithStats(our, 8)[0] - 1
        rows.append((cover_host, cover_our))
        print(f"{p.stem:16} {ng:>11} {no:>10} {100*cover_host:>13.0f}% {100*cover_our:>15.0f}%")
        im = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (S, S))
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vis[gt > 0] = (0, 255, 0)        # host = green
        vis[our > 0] = (0, 0, 255)       # ours = red
        cv2.putText(vis, f"{p.stem}  host={ng} green  ours={no} red  we-cover-host={100*cover_host:.0f}%", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, f"{p.stem}  host={ng} green  ours={no} red  we-cover-host={100*cover_host:.0f}%", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
        cv2.imwrite(str(OUT / f"{p.stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    a = np.array(rows, float)
    print(f"\nSUMMARY ({len(a)} imgs): we cover host's fascicles {100*a[:,0].mean():.0f}% on avg "
          f"(high = we find their dashes) | overlays -> {OUT}")
    print("note: 'we_cover_host' is the one to trust - do our predictions land on the host's labeled fascicles.")


if __name__ == "__main__":
    main()
