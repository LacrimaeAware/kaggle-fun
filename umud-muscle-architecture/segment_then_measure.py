"""UMUD segment-then-measure: train aponeurosis/fascicle U-Nets, predict on the test
images, measure geometry, write a submission.

Designed to run unchanged on a Kaggle GPU notebook (the clean path, free CUDA) or
locally. It auto-discovers the data folders wherever they are mounted (Kaggle slug
paths and extra wrapper folders vary, so nothing is hard-coded) and picks the device.
On Kaggle: Add Input -> attach the competition, enable GPU and Internet, then run.

Pipeline:
    test image -> apo U-Net + fascicle U-Net -> binary masks -> fit lines -> PA, FL_px, MT_px
Submission for this first version: pennation angle from the segmentation geometry,
fascicle length and thickness from the prior (calibration is the next step).

Env knobs (optional):
    UMUD_EPOCHS=2      fast smoke run (default 12)
    UMUD_MAX_PAIRS=24  cap training pairs for a quick local test (default 0 = all)
"""

import os
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
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOCAL = HERE / "data"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else (HERE / "results")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = 4 if os.name != "nt" else 0  # Kaggle Linux: 4 workers; Windows spawns/re-imports, keep 0
PIN = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"
IMG_SIZE = 384
SEED = 42
EPOCHS = int(os.environ.get("UMUD_EPOCHS", "12"))      # UMUD_EPOCHS=2 for a fast smoke run
MAX_PAIRS = int(os.environ.get("UMUD_MAX_PAIRS", "0"))  # >0 caps train pairs (local smoke tests)
PRIOR = {"fl_mm": 74.424, "mt_mm": 18.628, "pa_deg": 15.105}
PA_MIN, PA_MAX = 5.0, 45.0  # competition physiological range for pennation angle
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Known leaf folder names, located wherever they are mounted. Do not hard-code the
# full path: the Kaggle competition mount and the local download nest these differently.
LEAVES = {
    "apo_img": ["apo_images_new_model_v1"],
    "apo_msk": ["apo_masks_new_model_v1"],
    "fasc_img": ["fasc_images_new_model_v1"],
    "fasc_msk": ["fasc_masks_new_model_v1"],
    "test": ["test_set_v2", "test_images_v2"],  # inner leaf preferred, wrapper as fallback
}
TARGET_DIRS = {"apo": ("apo_img", "apo_msk"), "fasc": ("fasc_img", "fasc_msk")}
_TERMINAL_LEAVES = {"apo_images_new_model_v1", "apo_masks_new_model_v1",
                    "fasc_images_new_model_v1", "fasc_masks_new_model_v1", "test_set_v2"}


def _list_dir_images(d):
    try:
        return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    except OSError:
        return []


def _search_roots():
    roots = []
    ki = Path("/kaggle/input")
    if ki.exists():
        roots += sorted(p for p in ki.iterdir() if p.is_dir())
    if LOCAL.exists():
        roots.append(LOCAL)
    return roots


def _index_root(root):
    """One bounded walk per root: map directory leaf name -> [full paths]. Does not
    descend into the big image folders (records their path, skips their thousands of
    files), but does descend into the test wrapper so the inner test_set_v2 is found."""
    index = {}
    for dirpath, dirnames, _files in os.walk(root):
        for dn in dirnames:
            index.setdefault(dn, []).append(Path(dirpath) / dn)
        dirnames[:] = [d for d in dirnames if d not in _TERMINAL_LEAVES]
    return index


def _resolve_in_index(index, key):
    for leaf in LEAVES[key]:
        cands = index.get(leaf, [])
        if key == "test":  # pick the dir that DIRECTLY holds the most images
            cands = sorted(cands, key=lambda d: len(_list_dir_images(d)), reverse=True)
            cands = [d for d in cands if _list_dir_images(d)]
        if cands:
            return cands[0]
    return None


def _print_tree(root, max_depth=3, depth=0, budget=None):
    budget = [500] if budget is None else budget
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for p in entries:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        print("  " * depth + ("[D] " if p.is_dir() else "    ") + p.name, flush=True)
        if p.is_dir() and depth < max_depth:
            _print_tree(p, max_depth, depth + 1, budget)


def resolve_data():
    roots = _search_roots()
    indexes = {root: _index_root(root) for root in roots}
    # Prefer the single root that resolves the most keys, so an extra attached dataset
    # (e.g. FALLMUD) with a same-named folder cannot bind a key away from the competition.
    best = None
    for root in roots:
        dirs = {k: _resolve_in_index(indexes[root], k) for k in LEAVES}
        n = sum(v is not None for v in dirs.values())
        if best is None or n > best[0]:
            best = (n, dirs)
    dirs = best[1] if best else {k: None for k in LEAVES}
    for k in LEAVES:  # fill any gaps from other roots
        if dirs.get(k) is None:
            for root in roots:
                d = _resolve_in_index(indexes[root], k)
                if d is not None:
                    dirs[k] = d
                    break
    print("=== data resolution ===", flush=True)
    for k in LEAVES:
        v = dirs.get(k)
        cnt = f" ({len(_list_dir_images(v))} images)" if v is not None else ""
        print(f"  {k:8s} -> {v}{cnt}", flush=True)
    missing = [k for k in LEAVES if dirs.get(k) is None]
    if missing:
        print(f"\nMISSING data folders: {missing}", flush=True)
        print("Currently mounted (confirm the competition is attached here):", flush=True)
        for root in roots or [Path("/kaggle/input")]:
            print(f"[root] {root}", flush=True)
            _print_tree(root)
        raise SystemExit(
            "Could not find the UMUD data folders. On Kaggle: open the right panel, "
            "click Add Input, attach the competition 'UMUD Challenge: Muscle "
            "Architecture in Ultrasound Data', then Run All again.")
    return dirs


DIRS = resolve_data()


def read_rgb(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"read fail {path}")
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] >= 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2RGB)
    else:  # 1- or 2-channel 3D array: take the first channel
        a = cv2.cvtColor(np.ascontiguousarray(a[:, :, 0]), cv2.COLOR_GRAY2RGB)
    if a.dtype != np.uint8:  # guard against 16-bit inputs
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return a


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"read fail {path}")
    return (m > 0).astype(np.float32)


def list_images(d):
    return {p.stem: p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS}


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
    img_key, msk_key = TARGET_DIRS[target]
    img_dir, msk_dir = DIRS[img_key], DIRS[msk_key]
    imgs, msks = list_images(img_dir), list_images(msk_dir)
    common = sorted(set(imgs) & set(msks))
    items = [(imgs[s], msks[s]) for s in common]
    if MAX_PAIRS > 0:
        items = items[:MAX_PAIRS]
    if not items:
        raise SystemExit(
            f"[{target}] 0 image/mask pairs.\n  images: {img_dir} (e.g. {list(imgs)[:3]})\n"
            f"  masks:  {msk_dir} (e.g. {list(msks)[:3]})")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(items))
    n_val = max(1, int(0.15 * len(items)))
    val = [items[i] for i in idx[:n_val]]
    tr = [items[i] for i in idx[n_val:]] or val
    print(f"[{target}] {len(items)} pairs ({len(tr)} train / {len(val)} val), device {DEVICE}", flush=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    tr_dl = DataLoader(SegDS(tr, tf(True)), batch_size=bs, shuffle=True, num_workers=NW,
                       pin_memory=PIN, persistent_workers=(NW > 0))
    va_dl = DataLoader(SegDS(val, tf(False)), batch_size=bs, num_workers=NW,
                       pin_memory=PIN, persistent_workers=(NW > 0))

    best = -1.0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for img, msk in tr_dl:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(img)
                loss = 0.5 * dice_loss(out, msk) + 0.5 * bce(out, msk)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval(); inter = 0.0; union = 0.0
        with torch.no_grad():
            for img, msk in va_dl:
                img, msk = img.to(DEVICE), msk.to(DEVICE)
                p = (torch.sigmoid(model(img)) > 0.5).float()
                inter += float((p * msk).sum())
                union += float(p.sum() + msk.sum())
        vdice = (2 * inter + 1e-6) / (union + 1e-6)  # global Dice over the val set
        print(f"[{target}] epoch {ep}: val_dice {vdice:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vdice > best:
            best = vdice
            torch.save(model.state_dict(), weights)
    if not weights.exists():  # epochs==0 or never improved: still save a checkpoint
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
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if xs.max() - xs.min() < 1e-6:  # vertical column: polyfit SVD would fail
        return 1e6, float(ys.mean())  # arctan(1e6) ~= +90 deg, the correct vertical slope
    s, b = np.polyfit(xs, ys, 1)
    return float(s), float(b)


def measure(apo_mask, fasc_mask):
    """Return pennation angle in degrees, or None if geometry fails."""
    apo_mask = np.ascontiguousarray(apo_mask, np.uint8)
    fasc_mask = np.ascontiguousarray(fasc_mask, np.uint8)
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
    test_dir = DIRS["test"]
    test_files = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print(f"test images: {len(test_files)} (from {test_dir})", flush=True)
    if len(test_files) != 309:
        print(f"WARNING: expected 309 test images, found {len(test_files)} -- "
              f"check the resolved test folder above.", flush=True)

    apo = train_segmenter("apo", epochs=EPOCHS)
    fasc = train_segmenter("fasc", epochs=EPOCHS)

    rows, ok = [], 0
    for p in test_files:
        img = read_rgb(p)
        try:
            pa = measure(predict_mask(apo, img), predict_mask(fasc, img))
        except Exception as e:  # no single image can abort the 309-row submission
            print(f"  measure failed on {p.name}: {e}", flush=True)
            pa = None
        if pa is None:
            pa = PRIOR["pa_deg"]
        else:
            ok += 1
        pa = float(np.clip(pa, PA_MIN, PA_MAX))  # keep inside the scored 5-45 deg band
        rows.append({"image_id": p.name, "pa_deg": round(pa, 3),
                     "fl_mm": PRIOR["fl_mm"], "mt_mm": PRIOR["mt_mm"]})
    sub = pd.DataFrame(rows)
    out_csv = OUT / "submission_segmentation.csv"
    sub.to_csv(out_csv, index=False)
    print(f"geometry succeeded on {ok}/{len(test_files)} images", flush=True)
    print(f"wrote {out_csv} ({len(sub)} rows)", flush=True)
    print(sub["pa_deg"].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
