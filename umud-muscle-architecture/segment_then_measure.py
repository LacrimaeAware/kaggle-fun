"""UMUD segment-then-measure: train aponeurosis/fascicle U-Nets, predict on the test
images, measure geometry, write a submission.

Designed to run unchanged on a Kaggle GPU notebook (the clean path, free CUDA) or
locally. It auto-detects the data path (/kaggle/input or ./data) and the device.
On Kaggle, attach the competition and enable GPU and internet, then run this file.

Pipeline:
    test image -> apo U-Net + fascicle U-Net -> binary masks -> fit lines -> PA, FL_px, MT_px
Submission for this first version: pennation angle from the segmentation geometry,
fascicle length and thickness from the prior (calibration is the next step).
"""

import os
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# segmentation_models_pytorch and albumentations (install on Kaggle if missing)
try:
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    os.system("pip install -q segmentation-models-pytorch albumentations")
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

# ---- environment ----
KAGGLE = Path("/kaggle/input/umud-challenge-muscle-architecture-in-ultrasound-data")
LOCAL = Path(__file__).resolve().parent / "data" if "__file__" in globals() else Path("data")
DATA = KAGGLE if KAGGLE.exists() else LOCAL
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else (Path(__file__).resolve().parent / "results")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
SEED = 42
PRIOR = {"fl_mm": 74.424, "mt_mm": 18.628, "pa_deg": 15.105}

PAIRS = {
    "apo": ("apo_imgs_v1/apo_images_new_model_v1", "apo_masks_v1/apo_masks_new_model_v1"),
    "fasc": ("fasc_imgs_v1/fasc_images_new_model_v1", "fasc_masks_v1/fasc_masks_new_model_v1"),
}


def read_rgb(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"read fail {path}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] >= 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2RGB)
    return a


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.float32)


class SegDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = read_rgb(self.items[i][0])
        msk = read_mask(self.items[i][1])
        if msk.shape[:2] != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        t = self.tf(image=img, mask=msk)
        return t["image"], t["mask"].unsqueeze(0)


def tf(train):
    aug = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    aug += [A.Normalize(), ToTensorV2()]
    return A.Compose(aug)


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)


def train_segmenter(target, epochs=12, bs=8):
    weights = OUT / f"seg_{target}.pt"
    img_dir, msk_dir = DATA / PAIRS[target][0], DATA / PAIRS[target][1]
    names = sorted({p.name for p in img_dir.glob("*.tif")} & {p.name for p in msk_dir.glob("*.tif")})
    items = [(img_dir / n, msk_dir / n) for n in names]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(items))
    n_val = max(1, int(0.15 * len(items)))
    val = [items[i] for i in idx[:n_val]]
    tr = [items[i] for i in idx[n_val:]]
    print(f"[{target}] {len(items)} pairs, device {DEVICE}", flush=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()
    tr_dl = DataLoader(SegDS(tr, tf(True)), batch_size=bs, shuffle=True, num_workers=2)
    va_dl = DataLoader(SegDS(val, tf(False)), batch_size=bs, num_workers=2)

    best = 0.0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for img, msk in tr_dl:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            opt.zero_grad()
            out = model(img)
            (0.5 * dice_loss(out, msk) + 0.5 * bce(out, msk)).backward()
            opt.step()
        sched.step()
        model.eval(); ds = 0.0; n = 0
        with torch.no_grad():
            for img, msk in va_dl:
                img, msk = img.to(DEVICE), msk.to(DEVICE)
                p = (torch.sigmoid(model(img)) > 0.5).float()
                ds += float(2 * (p * msk).sum() / (p.sum() + msk.sum() + 1e-6)); n += 1
        vdice = ds / max(n, 1)
        print(f"[{target}] epoch {ep}: val_dice {vdice:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vdice > best:
            best = vdice
            torch.save(model.state_dict(), weights)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    print(f"[{target}] best dice {best:.4f}", flush=True)
    return model


@torch.no_grad()
def predict_mask(model, image_rgb):
    h, w = image_rgb.shape[:2]
    t = tf(False)(image=image_rgb, mask=np.zeros((h, w), np.float32))
    logit = model(t["image"].unsqueeze(0).to(DEVICE))
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return cv2.resize((prob > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def fit_line(ys, xs):
    s, b = np.polyfit(xs, ys, 1)
    return float(s), float(b)


def measure(apo_mask, fasc_mask):
    """Return pennation angle in degrees, or None if geometry fails."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    lines = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        s, b = fit_line(ys, xs)
        lines.append((np.mean(ys), s, b))
    lines.sort()
    deep_s = lines[-1][1]  # lower band = deep aponeurosis
    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs = []
    for i in range(1, nf):
        if statsf[i, 4] < 20:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, _ = fit_line(ys, xs)
        a = abs(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        if 2 <= a <= 75:
            angs.append(a)
    return float(np.median(angs)) if angs else None


def main():
    apo = train_segmenter("apo")
    fasc = train_segmenter("fasc")
    test_files = sorted(glob.glob(str(DATA / "test_images_v2" / "**" / "*.*"), recursive=True))
    test_files = [f for f in test_files if f.lower().endswith((".tif", ".tiff", ".png"))]
    print(f"test images: {len(test_files)}", flush=True)
    rows, ok = [], 0
    for f in test_files:
        img = read_rgb(Path(f))
        pa = measure(predict_mask(apo, img), predict_mask(fasc, img))
        if pa is None:
            pa = PRIOR["pa_deg"]
        else:
            ok += 1
        rows.append({"image_id": Path(f).name, "pa_deg": round(pa, 3),
                     "fl_mm": PRIOR["fl_mm"], "mt_mm": PRIOR["mt_mm"]})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission_segmentation.csv", index=False)
    print(f"geometry succeeded on {ok}/{len(test_files)} images", flush=True)
    print(f"wrote {OUT / 'submission_segmentation.csv'}", flush=True)
    print(sub["pa_deg"].describe().to_string())


if __name__ == "__main__":
    main()
