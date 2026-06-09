"""Train a U-Net segmenter (aponeurosis or fascicle) for UMUD.

Runs on an AMD GPU via DirectML, on CUDA, or on CPU, picking whatever is available.
This is the real segment-then-measure path: a model that draws aponeurosis/fascicle
masks on new images, so geometry can be measured on the 309 test images.

Usage (from the repository root, in the project .venv):
    python umud-muscle-architecture/train_segmentation.py --target apo --epochs 12
    python umud-muscle-architecture/train_segmentation.py --target fasc --epochs 12

Saves weights to umud-muscle-architecture/results/seg_<target>.pt (gitignored).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
IMG_SIZE = 384
SEED = 42

PAIRS = {
    "apo": ("apo_imgs_v1/apo_images_new_model_v1", "apo_masks_v1/apo_masks_new_model_v1"),
    "fasc": ("fasc_imgs_v1/fasc_images_new_model_v1", "fasc_masks_v1/fasc_masks_new_model_v1"),
}


def get_device():
    try:
        import torch_directml as dml
        if dml.device_count() > 0:
            return dml.device(), "directml:" + dml.device_name(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def read_rgb(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"read fail {path}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)
    return a


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"read fail {path}")
    return (m > 0).astype(np.float32)


class SegDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, msk_p = self.items[i]
        img = read_rgb(img_p)
        msk = read_mask(msk_p)
        if msk.shape[:2] != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        t = self.tf(image=img, mask=msk)
        return t["image"], t["mask"].unsqueeze(0)


def make_tf(train):
    aug = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3)]
    aug += [A.Normalize(), ToTensorV2()]
    return A.Compose(aug)


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum() + eps
    den = p.sum() + target.sum() + eps
    return 1 - num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["apo", "fasc"], required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    img_dir = DATA / PAIRS[args.target][0]
    msk_dir = DATA / PAIRS[args.target][1]
    names = sorted({p.name for p in img_dir.glob("*.tif")} & {p.name for p in msk_dir.glob("*.tif")})
    items = [(img_dir / n, msk_dir / n) for n in names]
    if not items:
        raise SystemExit(f"no image/mask pairs for {args.target} under {img_dir}")
    print(f"{args.target}: {len(items)} image/mask pairs")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(items))
    n_val = max(1, int(0.15 * len(items)))
    val_items = [items[i] for i in idx[:n_val]]
    tr_items = [items[i] for i in idx[n_val:]]

    dev, devname = get_device()
    print("device:", devname)

    tr_dl = DataLoader(SegDS(tr_items, make_tf(True)), batch_size=args.bs, shuffle=True, num_workers=0)
    va_dl = DataLoader(SegDS(val_items, make_tf(False)), batch_size=args.bs, shuffle=False, num_workers=0)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tl = 0.0
        for img, msk in tr_dl:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad()
            out = model(img)
            loss = 0.5 * dice_loss(out, msk) + 0.5 * bce(out, msk)
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()

        model.eval()
        ds, n = 0.0, 0
        with torch.no_grad():
            for img, msk in va_dl:
                img, msk = img.to(dev), msk.to(dev)
                p = (torch.sigmoid(model(img)) > 0.5).float()
                inter = (p * msk).sum()
                ds += (2 * inter / (p.sum() + msk.sum() + 1e-6)).item()
                n += 1
        vdice = ds / max(n, 1)
        print(f"epoch {ep}: train_loss {tl/len(tr_dl):.4f} val_dice {vdice:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if vdice > best:
            best = vdice
            RESULTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "img_size": IMG_SIZE,
                        "encoder": "resnet34", "target": args.target}, RESULTS / f"seg_{args.target}.pt")

    print(f"best val dice {best:.4f}; saved {RESULTS / ('seg_' + args.target + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
