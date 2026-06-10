"""Experiment 28: train a weak-label scale-cue segmenter.

This trains a small multi-label segmentation model against the code-generated
scale-cue masks from exp26. It is not a submission generator and it does not
prove hidden-test accuracy; the validation metric is agreement with the weak
teacher labels. Its purpose is to create a learned cue detector that can be
compared against the deterministic router and used to flag disagreements.

Run after:
    .\\.venv\\Scripts\\python.exe experiments\\exp26_scale_cue_pseudolabels.py

Smoke test:
    .\\.venv\\Scripts\\python.exe experiments\\exp28_train_scale_cue_segmenter.py --smoke

Outputs:
    results/scale_cue_segmenter.pt
    results/scale_cue_segmenter_metrics.csv
    results/scale_cue_segmenter_smoke.pt (smoke mode)
    results/scale_cue_segmenter_smoke_metrics.csv (smoke mode)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "test_images_v2" / "test_set_v2"
RESULTS = ROOT / "results"
LABEL_DIR = RESULTS / "scale_cue_pseudolabels"
MANIFEST = LABEL_DIR / "manifest.csv"
MODEL_OUT = RESULTS / "scale_cue_segmenter.pt"
METRICS_OUT = RESULTS / "scale_cue_segmenter_metrics.csv"

CUE_CLASSES = [
    "bottom_tick_axis",
    "left_ruler_ticks",
    "right_ruler_ticks",
    "ui_signature_marks",
    "bottom_scale_bar",
]


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


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {path}")
    return (mask > 0).astype(np.float32)


def load_items(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; run exp26 first")
    df = pd.read_csv(manifest_path)
    items = []
    for image_id, sub in df.groupby("image_id"):
        channels = []
        for cue_class in CUE_CLASSES:
            rows = sub[sub["cue_class"] == cue_class]
            if len(rows) == 0:
                channels.append(None)
            else:
                channels.append(ROOT / str(rows.iloc[0]["mask_path"]))
        if any(p is not None for p in channels):
            items.append({"image_id": image_id, "image_path": DATA / image_id, "mask_paths": channels})
    return sorted(items, key=lambda x: x["image_id"])


class CueDS(Dataset):
    def __init__(self, items: list[dict], tf, mask_dilate: int = 0):
        self.items = items
        self.tf = tf
        self.mask_dilate = int(mask_dilate)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = read_rgb(item["image_path"])
        masks = []
        for path in item["mask_paths"]:
            if path is None:
                masks.append(np.zeros(img.shape[:2], dtype=np.float32))
            else:
                m = read_mask(path)
                if m.shape[:2] != img.shape[:2]:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(m)
        mask = np.stack(masks, axis=-1)
        t = self.tf(image=img, mask=mask)
        mask_t = t["mask"].permute(2, 0, 1).float()
        if self.mask_dilate > 0:
            k = 2 * self.mask_dilate + 1
            mask_t = F.max_pool2d(mask_t.unsqueeze(0), kernel_size=k, stride=1, padding=self.mask_dilate).squeeze(0)
        return t["image"], mask_t


def make_tf(img_size: int, train: bool):
    aug = [A.Resize(img_size, img_size)]
    if train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            A.Affine(translate_percent=0.02, scale=(0.95, 1.05), rotate=(-2, 2), p=0.35),
        ]
    aug += [A.Normalize(), ToTensorV2()]
    return A.Compose(aug)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (prob * target).sum(dims)
    den = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (den + eps)
    return 1 - dice.mean()


def dice_by_class(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    rows = []
    for i, name in enumerate(CUE_CLASSES):
        p = pred[:, i]
        t = target[:, i]
        inter = (p * t).sum()
        den = p.sum() + t.sum()
        dice = float((2 * inter + eps) / (den + eps))
        rows.append({"cue_class": name, "dice": dice, "target_px": int(t.sum().item()), "pred_px": int(p.sum().item())})
    return rows


def split_items(items: list[dict], val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = max(1, int(round(len(items) * val_frac)))
    val_idx = set(idx[:n_val])
    train = [item for i, item in enumerate(items) if i not in val_idx]
    val = [item for i, item in enumerate(items) if i in val_idx]
    return train, val


def estimate_pos_weight(items: list[dict], img_size: int, mask_dilate: int) -> torch.Tensor:
    ds = CueDS(items, make_tf(img_size, False), mask_dilate=mask_dilate)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    pos = torch.zeros(len(CUE_CLASSES), dtype=torch.float64)
    total = 0
    for _img, mask in dl:
        pos += mask.sum(dim=(0, 2, 3)).double()
        total += mask.shape[0] * mask.shape[2] * mask.shape[3]
    neg = torch.clamp(torch.tensor(float(total), dtype=torch.float64) - pos, min=1.0)
    weights = neg / torch.clamp(pos, min=1.0)
    return torch.clamp(weights.float(), min=1.0, max=100.0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--encoder-weights", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-dilate", type=int, default=2)
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.bs = min(args.bs, 2)
        args.img_size = min(args.img_size, 128)
    model_out = RESULTS / ("scale_cue_segmenter_smoke.pt" if args.smoke else "scale_cue_segmenter.pt")
    metrics_out = RESULTS / (
        "scale_cue_segmenter_smoke_metrics.csv" if args.smoke else "scale_cue_segmenter_metrics.csv"
    )

    items = load_items(MANIFEST)
    if args.smoke:
        items = items[:24]
    if len(items) < 4:
        raise SystemExit("not enough cue-label items")

    train_items, val_items = split_items(items, args.val_frac, args.seed)
    dev, devname = get_device()
    print(f"items train/val: {len(train_items)}/{len(val_items)}")
    print(f"device: {devname}")

    train_dl = DataLoader(
        CueDS(train_items, make_tf(args.img_size, True), mask_dilate=args.mask_dilate),
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,
    )
    val_dl = DataLoader(
        CueDS(val_items, make_tf(args.img_size, False), mask_dilate=args.mask_dilate),
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
    )

    model = smp.Unet(
        args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=len(CUE_CLASSES),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if args.no_pos_weight:
        pos_weight = None
        bce = nn.BCEWithLogitsLoss()
    else:
        pos_weight = estimate_pos_weight(train_items, args.img_size, args.mask_dilate).to(dev).view(1, -1, 1, 1)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"mask_dilate: {args.mask_dilate}")
    if pos_weight is not None:
        print("pos_weight:", ", ".join(f"{x:.1f}" for x in pos_weight.flatten().detach().cpu().tolist()))
    best = -1.0
    metrics_rows = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for img, mask in train_dl:
            img, mask = img.to(dev), mask.to(dev)
            opt.zero_grad()
            logits = model(img)
            loss = 0.5 * dice_loss(logits, mask) + 0.5 * bce(logits, mask)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        model.eval()
        class_rows = []
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(dev), mask.to(dev)
                class_rows.extend(dice_by_class(model(img), mask))
        val_mean = float(np.mean([r["dice"] for r in class_rows]))
        metrics_rows.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_dl)),
            "val_mean_weak_dice": val_mean,
            "seconds": time.time() - t0,
            "smoke": bool(args.smoke),
        })
        print(
            f"epoch {epoch}: loss {metrics_rows[-1]['train_loss']:.4f} "
            f"val_weak_dice {val_mean:.4f} ({metrics_rows[-1]['seconds']:.0f}s)",
            flush=True,
        )

        if val_mean > best:
            best = val_mean
            RESULTS.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "cue_classes": CUE_CLASSES,
                "img_size": args.img_size,
                "encoder": args.encoder,
                "encoder_weights": args.encoder_weights,
                "weak_label_manifest": str(MANIFEST.relative_to(ROOT)),
                "mask_dilate": args.mask_dilate,
                "pos_weight": None if pos_weight is None else pos_weight.flatten().detach().cpu().tolist(),
                "smoke": bool(args.smoke),
            }, model_out)

    metrics = pd.DataFrame(metrics_rows)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_out, index=False)
    print(f"best weak-label val dice: {best:.4f}")
    print(f"saved {model_out}")
    print(f"wrote {metrics_out}")
    print("read: this validates the learned cue path against weak labels only, not true target labels.")


if __name__ == "__main__":
    main()
