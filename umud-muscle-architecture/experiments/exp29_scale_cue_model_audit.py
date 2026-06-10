"""Experiment 29: audit the learned scale-cue segmenter.

This compares the exp28 learned cue model against exp26's code-generated weak
labels. It does not use human target labels and does not generate a submission.

Outputs:
    results/scale_cue_segmenter_audit.csv
    results/scale_cue_segmenter_audit_summary.csv
    results/scale_cue_segmenter_threshold_sweep.csv
    results/scale_cue_segmenter_threshold_best.csv
    results/scale_cue_segmenter_audit_overlays/*.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp28_train_scale_cue_segmenter import CUE_CLASSES, read_rgb  # noqa: E402

DATA = ROOT / "data" / "test_images_v2" / "test_set_v2"
RESULTS = ROOT / "results"
LABEL_DIR = RESULTS / "scale_cue_pseudolabels"
MANIFEST = LABEL_DIR / "manifest.csv"
MODEL_PATH = RESULTS / "scale_cue_segmenter.pt"
OUT_ROWS = RESULTS / "scale_cue_segmenter_audit.csv"
OUT_SUMMARY = RESULTS / "scale_cue_segmenter_audit_summary.csv"
OUT_SWEEP = RESULTS / "scale_cue_segmenter_threshold_sweep.csv"
OUT_BEST = RESULTS / "scale_cue_segmenter_threshold_best.csv"
OUT_OVERLAYS = RESULTS / "scale_cue_segmenter_audit_overlays"
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def read_mask(path: Path, size: int, dilate: int = 0) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    if dilate > 0:
        k = 2 * int(dilate) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask.astype(np.float32)


def display_rgb(path: Path) -> Image.Image:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"failed to read {path}")
    lo, hi = np.percentile(gray, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    disp = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB))


def load_teacher_masks(size: int):
    df = pd.read_csv(MANIFEST)
    teacher: dict[str, dict[str, Path]] = {}
    for _, row in df.iterrows():
        teacher.setdefault(row["image_id"], {})[row["cue_class"]] = ROOT / str(row["mask_path"])
    return teacher


def make_tf(size: int):
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing {MODEL_PATH}; run exp28 real training first")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt.get("cue_classes", CUE_CLASSES)
    if list(classes) != CUE_CLASSES:
        raise SystemExit(f"unexpected cue classes in checkpoint: {classes}")
    model = smp.Unet(
        ckpt.get("encoder", "resnet18"),
        encoder_weights=None,
        in_channels=3,
        classes=len(CUE_CLASSES),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, int(ckpt.get("img_size", 256)), ckpt


def dice(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
    inter = float((pred * target).sum())
    den = float(pred.sum() + target.sum())
    return (2 * inter + eps) / (den + eps)


def threshold_sweep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cls, sub in df.groupby("cue_class"):
        y = sub["target_present"].astype(bool).to_numpy()
        score = sub["prob_max"].to_numpy()
        for thr in np.linspace(0.50, 0.99, 50):
            pred = score >= thr
            tp = int((pred & y).sum())
            fp = int((pred & ~y).sum())
            fn = int((~pred & y).sum())
            tn = int((~pred & ~y).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({
                "cue_class": cls,
                "threshold": float(thr),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / len(sub),
            })
    sweep = pd.DataFrame(rows)
    best = (
        sweep.sort_values(["cue_class", "f1", "accuracy"], ascending=[True, False, False])
        .groupby("cue_class", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return sweep, best


def clear_overlays():
    OUT_OVERLAYS.mkdir(parents=True, exist_ok=True)
    for item in OUT_OVERLAYS.iterdir():
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            item.unlink()


def draw_overlay(image_path: Path, probs: np.ndarray, targets: dict[str, np.ndarray], size: int, threshold: float, out_path: Path):
    img = display_rgb(image_path).resize((size, size))
    draw = ImageDraw.Draw(img, "RGBA")
    colors = {
        "bottom_tick_axis": (0, 255, 0, 90),
        "left_ruler_ticks": (0, 200, 255, 90),
        "right_ruler_ticks": (255, 80, 80, 90),
        "ui_signature_marks": (255, 180, 0, 90),
        "bottom_scale_bar": (255, 0, 255, 90),
    }
    for i, cls in enumerate(CUE_CLASSES):
        pred = probs[i] >= threshold
        target = targets.get(cls)
        color = colors[cls]
        ys, xs = np.where(pred)
        for x, y in zip(xs[::10], ys[::10]):
            draw.point((int(x), int(y)), fill=color)
        if target is not None:
            ys, xs = np.where(target > 0)
            for x, y in zip(xs[::4], ys[::4]):
                draw.point((int(x), int(y)), fill=(255, 255, 255, 150))
    img.save(out_path, quality=92)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--presence-threshold", type=int, default=12)
    ap.add_argument("--target-dilate", type=int, default=0)
    ap.add_argument("--max-overlays", type=int, default=40)
    args = ap.parse_args()

    model, size, ckpt = load_model()
    dev, devname = get_device()
    model = model.to(dev).eval()
    tf = make_tf(size)
    teacher = load_teacher_masks(size)
    clear_overlays()

    rows = []
    images = sorted(p for p in DATA.iterdir() if p.suffix.lower() in IMG_EXTS)
    with torch.no_grad():
        for path in images:
            img = read_rgb(path)
            t = tf(image=img)["image"].unsqueeze(0).to(dev)
            probs = torch.sigmoid(model(t))[0].cpu().numpy()
            targets = {
                cls: read_mask(mask_path, size, args.target_dilate)
                for cls, mask_path in teacher.get(path.name, {}).items()
            }
            for i, cls in enumerate(CUE_CLASSES):
                pred = (probs[i] >= args.threshold).astype(np.float32)
                target = targets.get(cls, np.zeros((size, size), dtype=np.float32))
                target_present = bool(target.sum() > 0)
                pred_px = int(pred.sum())
                pred_present = pred_px >= args.presence_threshold
                rows.append({
                    "image_id": path.name,
                    "cue_class": cls,
                    "target_present": target_present,
                    "pred_present": pred_present,
                    "presence_match": target_present == pred_present,
                    "dice": dice(pred, target),
                    "target_px": int(target.sum()),
                    "pred_px": pred_px,
                    "prob_max": float(probs[i].max()),
                    "prob_mean": float(probs[i].mean()),
                    "threshold": args.threshold,
                    "presence_threshold": args.presence_threshold,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROWS, index=False)
    sweep, best = threshold_sweep(df)
    sweep.to_csv(OUT_SWEEP, index=False)
    best.to_csv(OUT_BEST, index=False)

    summary_rows = []
    for cls, sub in df.groupby("cue_class"):
        pos = sub[sub["target_present"]]
        neg = sub[~sub["target_present"]]
        summary_rows.append({
            "cue_class": cls,
            "n_images": int(len(sub)),
            "target_positive": int(sub["target_present"].sum()),
            "pred_positive": int(sub["pred_present"].sum()),
            "presence_accuracy": float(sub["presence_match"].mean()),
            "positive_presence_recall": float((pos["pred_present"]).mean()) if len(pos) else np.nan,
            "negative_false_presence_rate": float((neg["pred_present"]).mean()) if len(neg) else np.nan,
            "mean_dice_on_positive": float(pos["dice"].mean()) if len(pos) else np.nan,
            "median_prob_max_on_positive": float(pos["prob_max"].median()) if len(pos) else np.nan,
            "median_prob_max_on_negative": float(neg["prob_max"].median()) if len(neg) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    # Save overlays for the worst target-positive misses first, then strongest negative false positives.
    misses = df[df["target_present"] & ~df["pred_present"]].sort_values("prob_max").head(args.max_overlays // 2)
    false_pos = df[(~df["target_present"]) & df["pred_present"]].sort_values("pred_px", ascending=False).head(args.max_overlays - len(misses))
    overlay_cases = pd.concat([misses, false_pos], ignore_index=True)
    for _, r in overlay_cases.iterrows():
        path = DATA / r["image_id"]
        img = read_rgb(path)
        t = tf(image=img)["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = torch.sigmoid(model(t))[0].cpu().numpy()
        targets = {
            cls: read_mask(mask_path, size)
            for cls, mask_path in teacher.get(path.name, {}).items()
        }
        if args.target_dilate > 0:
            targets = {
                cls: read_mask(mask_path, size, args.target_dilate)
                for cls, mask_path in teacher.get(path.name, {}).items()
            }
        reason = "miss" if r["target_present"] else "falsepos"
        out_path = OUT_OVERLAYS / f"{reason}__{Path(r['image_id']).stem}__{r['cue_class']}.jpg"
        draw_overlay(path, probs, targets, size, args.threshold, out_path)

    print(f"device: {devname}")
    print(f"model: {MODEL_PATH}")
    print(f"checkpoint smoke: {ckpt.get('smoke', False)}")
    print(f"threshold: {args.threshold}; presence_threshold: {args.presence_threshold}")
    print(f"target_dilate: {args.target_dilate}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nbest prob_max thresholds against weak labels:")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {OUT_ROWS}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"wrote {OUT_SWEEP}")
    print(f"wrote {OUT_BEST}")
    print(f"wrote overlays -> {OUT_OVERLAYS}")
    print("\nread: this audits agreement with weak labels only; disagreements are code-review targets, not human labels.")


if __name__ == "__main__":
    main()
