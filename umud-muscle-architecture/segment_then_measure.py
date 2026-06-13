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
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # quiet libtiff ImageJ-tag warns (set before cv2 import)
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tick_calibration import choose_candidate as choose_calibration_candidate
    from tick_calibration import read_gray as read_calibration_gray
    import scale_ticks  # validated per-family scale router (competition_reference.md 3b)
    CALIBRATION_AVAILABLE = True
except Exception as exc:
    choose_calibration_candidate = None
    read_calibration_gray = None
    scale_ticks = None
    CALIBRATION_AVAILABLE = False
    CALIBRATION_IMPORT_ERROR = exc

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

# Quiet OpenCV's per-image libtiff warnings: these images carry ImageJ private tags
# (50838/50839) that libtiff does not recognize. The pixels decode fine; only the log
# is affected. Suppressing keeps the training output readable.
try:
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 2))
except Exception:
    pass

# ---- environment ----
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOCAL = HERE / "data"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else (HERE / "results")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW = 4 if os.name != "nt" else 0  # Kaggle Linux: 4 workers; Windows spawns/re-imports, keep 0
PIN = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"
IMG_SIZE = int(os.environ.get("UMUD_IMG_SIZE", "384"))
TTA_EXTRA_SIZE = int(os.environ.get("UMUD_TTA_EXTRA_SIZE", "448"))
MODEL_ARCH = os.environ.get("UMUD_MODEL_ARCH", "unet").lower()  # unet | unetplusplus | fpn | deeplabv3plus
MODEL_ENCODER = os.environ.get("UMUD_MODEL_ENCODER", "resnet34")
MODEL_ENCODER_WEIGHTS = os.environ.get("UMUD_MODEL_ENCODER_WEIGHTS", "imagenet")
LOSS_MODE = os.environ.get("UMUD_LOSS_MODE", "dice_bce").lower()  # dice_bce | dice_focal | dice_tversky
AUG_LEVEL = os.environ.get("UMUD_AUG_LEVEL", "light").lower()  # light | strong
MASK_THRESHOLD = float(os.environ.get("UMUD_MASK_THRESHOLD", "0.5"))
APO_MASK_THRESHOLD = float(os.environ.get("UMUD_APO_MASK_THRESHOLD", str(MASK_THRESHOLD)))
FASC_MASK_THRESHOLD = float(os.environ.get("UMUD_FASC_MASK_THRESHOLD", str(MASK_THRESHOLD)))
BS = int(os.environ.get("UMUD_BATCH_SIZE", "8"))
WEIGHTS_TAG = os.environ.get("UMUD_WEIGHTS_TAG", "").strip()
SEED = 42
EPOCHS = int(os.environ.get("UMUD_EPOCHS", "12"))      # UMUD_EPOCHS=2 for a fast smoke run
MAX_PAIRS = int(os.environ.get("UMUD_MAX_PAIRS", "0"))  # >0 caps train pairs (local smoke tests)
PRIOR = {"fl_mm": 74.424, "mt_mm": 18.628, "pa_deg": 15.105}
PA_MIN, PA_MAX = 5.0, 45.0  # competition physiological range for pennation angle
FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0
USE_CALIBRATED_MT = os.environ.get("UMUD_USE_CALIBRATED_MT", "1") != "0"
USE_CALIBRATED_FL = os.environ.get("UMUD_USE_CALIBRATED_FL", "0") == "1"
USE_SCALE_ROUTER = os.environ.get("UMUD_SCALE_ROUTER", "1") != "0"  # validated per-family scale (54% coverage)
USE_IDENTITY_FL = os.environ.get("UMUD_USE_IDENTITY_FL", "1") != "0"  # FL = MT/sin(PA), fallback when no fragment
USE_FRAGMENT_FL = os.environ.get("UMUD_FRAGMENT_FL", "1") != "0"  # FL from fascicle fragment extrapolated to apo lines; beats identity once apo inner-edge is fixed (0.481->0.353 on the 35 experts)
FL_IDENTITY_BLEND = float(os.environ.get("UMUD_FL_IDENTITY_BLEND", "0"))  # keep fragment-only FL by default; blend=.5 looked better locally but regressed public LB 0.61918->~0.64
FL_FRAGMENT_MODE = os.environ.get("UMUD_FL_FRAGMENT_MODE", "median").lower()  # median | min_extrap_top3 | visibility_weighted
FL_FRAGMENT_TOPK = int(os.environ.get("UMUD_FL_FRAGMENT_TOPK", "3"))  # host protocol uses 3 manually selected low-extrapolation structures
USE_FL_FACING = os.environ.get("UMUD_FL_FACING", "0") == "1"  # opt-in rejected probe: consensus angle + facing-parabola apo + minimize-extrapolation. It improved the 35-expert FL proxy but regressed public LB 0.61918->0.66459, so keep the safe fragment-FL baseline as default.
USE_FL_RECENTER = os.environ.get("UMUD_FL_RECENTER", "1") != "0"  # ON (default) = pin submission FL mean to PRIOR (the 0.61918 baseline). OFF = honest per-image FL (fl_px/scale), prior only where no scale; this is the principled "no mean" version but it exposes the FL geometry's ~+6mm overshoot the recenter masks, so it is a leaderboard bet, not a free win.
USE_TTA = os.environ.get("UMUD_TTA", "1") != "0"  # mirror+scale test-time aug; denoises masks (exp08: 0.383->0.370)
FASC_MIN_AREA = int(os.environ.get("UMUD_FASC_MIN_AREA", "40"))   # drop tiniest fragments (exp09)
FASC_MIN_ANG = float(os.environ.get("UMUD_FASC_MIN_ANG", "6"))    # reject apo-parallel fragments (exp07/09: PA 0.171->0.164)
USE_APO_INNER = os.environ.get("UMUD_APO_INNER", "1") != "0"      # measure MT between bands' muscle-facing INNER edges, not centroids (exp14: MT-term 0.49->0.18)
TOP_BOUNDARY_MODE = os.environ.get("UMUD_TOP_BOUNDARY_MODE", "line").lower()  # line | triangle | robust_triangle; opt-in upper-boundary shape candidate
MT_MODE = os.environ.get("UMUD_MT_MODE", "perp_center").lower()    # perp_center | vertical_3 (host straight-line left/mid/right approximation)
FASC_POS_WEIGHT = float(os.environ.get("UMUD_FASC_POS_WEIGHT", "0"))  # >0 biases fascicle BCE toward recall (Kaggle retrain only)
USE_CLAHE = os.environ.get("UMUD_CLAHE", "0") == "1"  # CLAHE contrast-normalize input; surfaces more fragments but MUST retrain both models with it on (exp10)
USE_TEMPORAL_SMOOTH = os.environ.get("UMUD_TEMPORAL_SMOOTH", "0") == "1"  # median-smooth within sequence clips (exp02); off by default
PIPELINE_VERSION = "2026-06-13.01"  # bump on every pipeline change; printed at run start so the version is verifiable
CALIBRATION_MIN_CONF = float(os.environ.get("UMUD_CALIBRATION_MIN_CONF", "0.3"))  # router gates per-method internally (png/644/right>=0.5, bottom>=0.9, faint-left>=0.30)
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
    if USE_CLAHE:  # contrast-normalize; only meaningful if BOTH models are (re)trained with it on (exp10)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))
        a = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
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
        if AUG_LEVEL == "strong":
            aug += [
                A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.12, rotate_limit=12,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.RandomBrightnessContrast(p=0.35),
                A.GaussNoise(p=0.15),
                A.MotionBlur(blur_limit=3, p=0.10),
            ]
    aug += [A.Normalize(), ToTensorV2()]
    return A.Compose(aug)


def weights_path(target):
    tag = f"_{WEIGHTS_TAG}" if WEIGHTS_TAG else ""
    return OUT / f"seg_{target}{tag}.pt"


def build_model(encoder_weights=None):
    kwargs = dict(
        encoder_name=MODEL_ENCODER,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )
    if MODEL_ARCH in {"unet", "u-net"}:
        return smp.Unet(**kwargs)
    if MODEL_ARCH in {"unetplusplus", "unet++", "unet_plus_plus"}:
        return smp.UnetPlusPlus(**kwargs)
    if MODEL_ARCH == "fpn":
        return smp.FPN(**kwargs)
    if MODEL_ARCH in {"deeplabv3plus", "deeplabv3+"}:
        return smp.DeepLabV3Plus(**kwargs)
    raise ValueError(f"unknown UMUD_MODEL_ARCH={MODEL_ARCH!r}")


def checkpoint_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)


def focal_bce_loss(logits, target, alpha=0.25, gamma=2.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * ((1 - pt) ** gamma) * bce).mean()


def tversky_loss(logits, target, alpha=0.35, beta=0.65, eps=1e-6):
    p = torch.sigmoid(logits)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    return 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def combined_loss(logits, target, bce):
    if LOSS_MODE == "dice_bce":
        return 0.5 * dice_loss(logits, target) + 0.5 * bce(logits, target)
    if LOSS_MODE == "dice_focal":
        return 0.5 * dice_loss(logits, target) + 0.5 * focal_bce_loss(logits, target)
    if LOSS_MODE == "dice_tversky":
        return 0.5 * dice_loss(logits, target) + 0.5 * tversky_loss(logits, target)
    raise ValueError(f"unknown UMUD_LOSS_MODE={LOSS_MODE!r}")


def train_segmenter(target, epochs=12, bs=None):
    bs = BS if bs is None else bs
    weights = weights_path(target)
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
    print(f"[{target}] model={MODEL_ARCH}/{MODEL_ENCODER} img_size={IMG_SIZE} loss={LOSS_MODE} "
          f"aug={AUG_LEVEL} batch={bs} weights={weights.name}", flush=True)

    model = build_model(encoder_weights=MODEL_ENCODER_WEIGHTS if MODEL_ENCODER_WEIGHTS != "none" else None).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    if target == "fasc" and FASC_POS_WEIGHT > 0:  # idea-2: counter dash sparsity, push fascicle recall
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(FASC_POS_WEIGHT, device=DEVICE))
        print(f"[{target}] using pos_weight={FASC_POS_WEIGHT} on BCE (recall bias)", flush=True)
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
                loss = combined_loss(out, msk, bce)
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
            torch.save({
                "state_dict": model.state_dict(),
                "target": target,
                "img_size": IMG_SIZE,
                "model_arch": MODEL_ARCH,
                "model_encoder": MODEL_ENCODER,
                "loss_mode": LOSS_MODE,
                "aug_level": AUG_LEVEL,
                "pipeline_version": PIPELINE_VERSION,
            }, weights)
    if not weights.exists():  # epochs==0 or never improved: still save a checkpoint
        torch.save({"state_dict": model.state_dict()}, weights)
    model.load_state_dict(checkpoint_state(torch.load(weights, map_location=DEVICE)))
    print(f"[{target}] best dice {best:.4f}", flush=True)
    return model


@torch.no_grad()
def _prob_at(model, image_rgb, size):
    """Sigmoid probability map at native resolution, run through the model at `size`x`size`."""
    h, w = image_rgb.shape[:2]
    im = cv2.resize(image_rgb, (size, size))
    t = tf(False)(image=im, mask=np.zeros((size, size), np.float32))
    with torch.no_grad():
        prob = torch.sigmoid(model(t["image"].unsqueeze(0).to(DEVICE)))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h))


def predict_mask(model, image_rgb, target=""):
    if USE_TTA:  # average original + horizontal mirror + second scale, then threshold (exp08: denoises)
        flipped = np.ascontiguousarray(image_rgb[:, ::-1])
        prob = (_prob_at(model, image_rgb, IMG_SIZE)
                + _prob_at(model, flipped, IMG_SIZE)[:, ::-1]
                + _prob_at(model, image_rgb, TTA_EXTRA_SIZE)) / 3.0
    else:
        prob = _prob_at(model, image_rgb, IMG_SIZE)
    thr = APO_MASK_THRESHOLD if target == "apo" else FASC_MASK_THRESHOLD if target == "fasc" else MASK_THRESHOLD
    return (prob > thr).astype(np.uint8)


def fit_line(ys, xs):
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if xs.max() - xs.min() < 1e-6:  # vertical column: polyfit SVD would fail
        return 1e6, float(ys.mean())  # arctan(1e6) ~= +90 deg, the correct vertical slope
    s, b = np.polyfit(xs, ys, 1)
    return float(s), float(b)


def pca_line(ys, xs):
    """Total-least-squares line (slope, intercept); unbiased for steep fascicles, unlike polyfit (exp05)."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    xc, yc = xs.mean(), ys.mean()
    d = np.stack([xs - xc, ys - yc])
    _, v = np.linalg.eigh(d @ d.T / max(len(xs), 1))
    vx, vy = v[:, 1]
    s = (vy / vx) if abs(vx) > 1e-6 else 1e6
    return float(s), float(yc - s * xc)


def weighted_median(vals, wts):
    order = np.argsort(vals)
    v = np.asarray(vals, float)[order]
    c = np.cumsum(np.asarray(wts, float)[order])
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def mad_gated_weighted_mean(vals, wts, k=2.5):
    """Robust aggregate for fragment PA: median seed, MAD inlier gate, weighted mean on inliers."""
    vals = np.asarray(vals, float)
    wts = np.asarray(wts, float)
    if len(vals) == 0:
        return None
    center = weighted_median(vals, wts)
    mad = np.median(np.abs(vals - center)) + 1e-9
    inlier = np.abs(vals - center) <= k * 1.4826 * mad
    if int(inlier.sum()) < 3:
        return float(center)
    return float(np.sum(vals[inlier] * wts[inlier]) / (np.sum(wts[inlier]) + 1e-9))


def line_y(line, x):
    s, b = line
    return s * x + b


def line_intersection(a, b):
    s1, b1 = a
    s2, b2 = b
    denom = s1 - s2
    if abs(denom) < 1e-6:
        return None
    x = (b2 - b1) / denom
    return float(x), float(line_y(a, x))


def line_from_points(p1, p2):
    slope = (p2[1] - p1[1]) / max(p2[0] - p1[0], 1e-9)
    return float(slope), float(p1[1] - slope * p1[0])


def make_top_boundary(line, edge_x=None, edge_y=None, mode="line"):
    """Return a line or a two-segment upper-boundary approximation."""
    if mode not in {"triangle", "robust_triangle"} or edge_x is None or edge_y is None or len(edge_x) < 12:
        return {"type": "line", "line": line, "mode": "line"}

    edge_x = np.asarray(edge_x, dtype=float)
    edge_y = np.asarray(edge_y, dtype=float)
    q25, q75 = np.percentile(edge_x, [25, 75])
    left = np.where(edge_x <= q25)[0]
    center = np.where((edge_x >= q25) & (edge_x <= q75))[0]
    right = np.where(edge_x >= q75)[0]
    if len(left) == 0 or len(center) == 0 or len(right) == 0:
        return {"type": "line", "line": line, "mode": "line"}

    if mode == "robust_triangle":
        def low(indices):
            cutoff = np.percentile(edge_y[indices], 95)
            keep = indices[edge_y[indices] >= cutoff]
            return float(np.median(edge_x[keep])), float(np.median(edge_y[keep]))

        def high(indices):
            cutoff = np.percentile(edge_y[indices], 5)
            keep = indices[edge_y[indices] <= cutoff]
            return float(np.median(edge_x[keep])), float(np.median(edge_y[keep]))

        pts = [low(left), high(center), low(right)]
    else:
        li = left[np.argmax(edge_y[left])]
        ci = center[np.argmin(edge_y[center])]
        ri = right[np.argmax(edge_y[right])]
        pts = [
            (float(edge_x[li]), float(edge_y[li])),
            (float(edge_x[ci]), float(edge_y[ci])),
            (float(edge_x[ri]), float(edge_y[ri])),
        ]
    return {"type": "piecewise", "points": pts, "lines": [line_from_points(pts[0], pts[1]), line_from_points(pts[1], pts[2])], "mode": mode}


def top_boundary_y(boundary, x):
    if boundary.get("type") != "piecewise":
        return line_y(boundary["line"], x)
    pts = boundary["points"]
    line = boundary["lines"][0] if x <= pts[1][0] else boundary["lines"][1]
    return line_y(line, x)


def top_boundary_intersection(fasc_line, boundary, xref=None):
    if boundary.get("type") != "piecewise":
        return line_intersection(fasc_line, boundary["line"])
    pts = boundary["points"]
    candidates = []
    for line, p1, p2 in ((boundary["lines"][0], pts[0], pts[1]), (boundary["lines"][1], pts[1], pts[2])):
        hit = line_intersection(fasc_line, line)
        if hit is None:
            continue
        lo, hi = sorted((p1[0], p2[0]))
        on_segment = lo - 10.0 <= hit[0] <= hi + 10.0
        ref = hit[0] if xref is None else xref
        candidates.append((0 if on_segment else 1, abs(hit[0] - ref), hit))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def fragment_visible_length(xs, ys, slope):
    """Length of the observed component projected onto its fitted axis."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ux = 1.0 / np.sqrt(1.0 + slope * slope)
    uy = slope * ux
    proj = xs * ux + ys * uy
    return float(np.ptp(proj)) if len(proj) else 0.0


def isect_par(cs, b, par, xref):
    """Intersect line y=cs*x+b with parabola y=par(x); return the (x,y) root nearest xref, or None."""
    A, B, C = float(par[0]), float(par[1]), float(par[2])
    a2, b2, c2 = A, B - cs, C - b
    if abs(a2) < 1e-9:
        if abs(b2) < 1e-12:
            return None
        x = -c2 / b2
    else:
        disc = b2 * b2 - 4 * a2 * c2
        if disc < 0:
            return None
        sq = disc ** 0.5
        r1, r2 = (-b2 + sq) / (2 * a2), (-b2 - sq) / (2 * a2)
        x = r1 if abs(r1 - xref) <= abs(r2 - xref) else r2
    return (x, cs * x + b)


def compute_facing_fl(used, sup, deep, sup_edges, deep_edges):
    """FL = median facing-parabola span over the minimize-extrapolation fragments, all cast at the
    single consensus angle (parallel, non-crossing). The parabolic apo is kept only where it bows
    toward the muscle (gap cannot diverge off-frame); otherwise the straight apo line is used. Visual-
    review validated on the 35 experts (raw FL 6.4mm->2.5mm, bias +6.0->0) and generalization-checked
    on 220 training muscles (identical behaviour, 0 haywire). Returns FL in px, or None."""
    if not used:
        return None
    sup_par = deep_par = None
    if sup_edges is not None and len(sup_edges[0]) >= 6:
        p = np.polyfit(sup_edges[0], sup_edges[1], 2)
        if float(p[0]) >= 0:                       # superficial bows down in the middle -> gap narrows at edges
            sup_par = p
    if deep_edges is not None and len(deep_edges[0]) >= 6:
        p = np.polyfit(deep_edges[0], deep_edges[1], 2)
        if float(p[0]) <= 0:                       # deep bows up in the middle
            deep_par = p
    ang = np.array([np.arctan(r["fs"]) for r in used], float)
    wt = np.array([max(1.0, r["area"]) for r in used], float)
    cs = float(np.tan(float(weighted_median(ang, wt))))    # consensus angle (length/area-weighted, robust)
    sel, allf = [], []
    for r in used:
        b = r["cy"] - cs * r["cx"]
        up = isect_par(cs, b, sup_par, r["cx"]) if sup_par is not None else line_intersection((cs, b), sup)
        lo = isect_par(cs, b, deep_par, r["cx"]) if deep_par is not None else line_intersection((cs, b), deep)
        if up is None or lo is None:
            continue
        fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        if not (10.0 <= fl <= 4000.0):
            continue
        allf.append(fl)
        if r["visible_len"] / (fl + 1e-9) >= 0.25:         # minimize-extrapolation (host's fascicle selection)
            sel.append(fl)
    if sel:
        return float(np.median(sel))
    if allf:
        return float(np.median(allf))
    return None


def aggregate_fragment_fl(fragment_rows):
    """Aggregate extrapolated spans from accepted fragments."""
    valid = [r for r in fragment_rows if r.get("fl") is not None and 10.0 <= r["fl"] <= 4000.0]
    if not valid:
        return None, None, 0

    median_fl = float(np.median([r["fl"] for r in valid]))
    mode = FL_FRAGMENT_MODE
    if mode == "median":
        return median_fl, median_fl, len(valid)

    if mode in {"min_extrap_top3", "host_top3", "top3_min_extrap"}:
        topk = max(1, FL_FRAGMENT_TOPK)
        ranked = sorted(
            valid,
            key=lambda r: (r["visible_frac"], r["visible_len"], r["area"]),
            reverse=True,
        )
        chosen = ranked[:min(topk, len(ranked))]
        return float(np.median([r["fl"] for r in chosen])), median_fl, len(chosen)

    if mode == "visibility_weighted":
        vals = np.asarray([r["fl"] for r in valid], dtype=float)
        wts = np.asarray(
            [max(1.0, r["area"]) * max(0.05, r["visible_frac"]) ** 2 for r in valid],
            dtype=float,
        )
        return float(weighted_median(vals, wts)), median_fl, len(valid)

    return median_fl, median_fl, len(valid)


def measure(apo_mask, fasc_mask):
    """Return PA plus pixel-space FL/MT geometry, or None if aponeurosis geometry fails."""
    apo_mask = np.ascontiguousarray(apo_mask, np.uint8)
    fasc_mask = np.ascontiguousarray(fasc_mask, np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    band_info = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        band_info.append((float(np.mean(ys)), xs, ys))
    band_info.sort()  # by mean y: [0] = superficial (top band), [1] = deep (bottom band)
    fit = []
    apo_edges = [None, None]  # (ux, inner-edge y) per band, for the facing-parabola FL
    for k, (role, (_, xs, ys)) in enumerate(zip(("sup", "deep"), band_info)):
        if USE_APO_INNER:  # fit the muscle-facing INNER edge: bottom of superficial, top of deep
            ux, inv = np.unique(xs, return_inverse=True)
            if role == "sup":
                ey = np.full(len(ux), -1.0); np.maximum.at(ey, inv, ys.astype(float))
            else:
                ey = np.full(len(ux), 1e18); np.minimum.at(ey, inv, ys.astype(float))
            fit.append(fit_line(ey, ux.astype(float)))
            apo_edges[k] = (ux.astype(float), ey.astype(float))
        else:
            fit.append(fit_line(ys, xs))
    superficial = fit[0]
    deep = fit[1]  # bands were ordered superficial-then-deep above
    top_boundary = make_top_boundary(
        superficial,
        apo_edges[0][0] if apo_edges[0] is not None else None,
        apo_edges[0][1] if apo_edges[0] is not None else None,
        TOP_BOUNDARY_MODE,
    )
    deep_s = deep[0]
    x_center = apo_mask.shape[1] / 2.0
    if MT_MODE in {"vertical_3", "host_vertical_3"}:
        sup_xs = band_info[0][1]
        deep_xs = band_info[1][1]
        x_min = max(float(np.min(sup_xs)), float(np.min(deep_xs)))
        x_max = min(float(np.max(sup_xs)), float(np.max(deep_xs)))
        if x_max - x_min >= 12:
            xs_mt = np.linspace(x_min, x_max, 5)[1:4]
        else:
            xs_mt = np.asarray([x_center])
        mt_px = float(np.mean([abs(line_y(deep, x) - top_boundary_y(top_boundary, x)) for x in xs_mt]))
    else:
        mt_px = abs(line_y(deep, x_center) - top_boundary_y(top_boundary, x_center)) / np.sqrt(1 + deep_s**2)

    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs, wts, fragment_rows, used_frags = [], [], [], []
    for i in range(1, nf):
        area = int(statsf[i, 4])
        if area < FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, fb = pca_line(ys, xs)  # unbiased fascicle orientation (exp05)
        a = abs(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        fasc = (fs, fb)
        upper = top_boundary_intersection(fasc, top_boundary, xref=float(np.mean(xs)))
        lower = line_intersection(fasc, deep)
        fl = None
        if upper is not None and lower is not None:
            fl = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
        if FASC_MIN_ANG <= a <= 75:
            angs.append(a)
            wts.append(area)  # weight by fragment size (exp05: sharpens PA)
            visible_len = fragment_visible_length(xs, ys, fs)
            used_frags.append({"fs": float(fs), "cx": float(np.mean(xs)), "cy": float(np.mean(ys)),
                               "visible_len": visible_len, "area": area})  # for the facing-parabola FL
            if fl is not None and 10.0 <= fl <= 4000.0:
                fragment_rows.append({
                    "fl": fl,
                    "area": area,
                    "visible_len": visible_len,
                    "visible_frac": float(np.clip(visible_len / (fl + 1e-9), 0.0, 1.0)),
                })
    pa_deg = weighted_median(angs, wts) if angs else None
    fl_fragment_px, fl_fragment_median_px, fl_fragment_n = aggregate_fragment_fl(fragment_rows)
    fl_identity_px = None
    if angs:
        pa_gated = mad_gated_weighted_mean(angs, wts)
        if pa_gated is not None and pa_gated > 0:
            fl_identity_px = float(mt_px / np.sin(np.radians(pa_gated)))
    fl_px = fl_fragment_px
    blend = float(np.clip(FL_IDENTITY_BLEND, 0.0, 1.0))
    if fl_fragment_px is not None and fl_identity_px is not None and blend > 0:
        fl_px = (1.0 - blend) * fl_fragment_px + blend * fl_identity_px
    elif fl_px is None and fl_identity_px is not None:
        fl_px = fl_identity_px
    if USE_FL_FACING:  # consensus + facing-parabola + minimize-extrapolation overrides the per-fragment span
        facing_px = compute_facing_fl(used_frags, superficial, deep, apo_edges[0], apo_edges[1])
        if facing_px is not None:
            fl_px = facing_px
    return {
        "pa_deg": pa_deg,
        "fl_px": fl_px,
        "fl_fragment_px": fl_fragment_px,
        "fl_fragment_median_px": fl_fragment_median_px,
        "fl_fragment_n": fl_fragment_n,
        "fl_identity_px": fl_identity_px,
        "mt_px": float(mt_px),
        "n_fascicles": len(angs),
        "top_boundary_mode": top_boundary.get("mode", "line"),
    }


class _Cal:  # lightweight calibration result, matches the attrs main() reads off a Candidate
    def __init__(self, px_per_mm, confidence, method, edge="", **extra):
        self.px_per_mm = px_per_mm
        self.confidence = confidence
        self.method = method
        self.edge = edge
        for k, v in extra.items():
            setattr(self, k, v)


def calibrate_image(path):
    if not CALIBRATION_AVAILABLE:
        return None
    gray = read_calibration_gray(path)
    if USE_SCALE_ROUTER and scale_ticks is not None:  # validated per-family router (PNG/644/Telemed/cropped)
        if hasattr(scale_ticks, "recover_for_image_detail"):
            det = scale_ticks.recover_for_image_detail(gray, path.name)
            scale, method, conf = det["scale_px_per_cm"], det["method"], det["conf"]
            extra = {k: v for k, v in det.items() if k not in ("scale_px_per_cm", "method", "conf")}
        else:
            scale, method, conf = scale_ticks.recover_for_image(gray, path.name)
            extra = {}
        if scale is None:
            return None
        return _Cal(px_per_mm=scale / 10.0, confidence=float(conf), method=method, **extra)
    return choose_calibration_candidate(gray, side_tick_mm=5.0, bottom_tick_mm=10.0, image_name=path.name)


def fingerprint(image_rgb):
    """Tiny normalized descriptor for detecting consecutive same-muscle frames (sequence clips)."""
    g = cv2.resize(image_rgb, (32, 32)).astype(np.float32).reshape(-1)
    return g / (np.linalg.norm(g) + 1e-9)


def temporal_smooth(sub, fps, thresh=0.92, max_len=12):
    """Median-smooth PA/FL/MT within runs of consecutive highly-similar frames (the test set's clip
    structure, not labels - a legitimate variance reduction; exp02 found ~112/308 consecutive pairs
    >0.9 similar). Clips longer than max_len are left alone (likely a mis-grouping)."""
    fps = np.asarray(fps, np.float32)
    sim = (fps[:-1] * fps[1:]).sum(axis=1)        # consecutive cosine similarity (fps are unit norm)
    clip = np.zeros(len(fps), int)
    for i in range(1, len(fps)):
        clip[i] = clip[i - 1] + (1 if sim[i - 1] < thresh else 0)
    out = sub.copy()
    smoothed = 0
    for cid in np.unique(clip):
        idx = np.where(clip == cid)[0]
        if 2 <= len(idx) <= max_len:
            smoothed += 1
            for col in ("pa_deg", "fl_mm", "mt_mm"):
                out.iloc[idx, out.columns.get_loc(col)] = round(float(np.median(sub[col].values[idx])), 3)
    print(f"temporal smoothing: {smoothed} clips (2-{max_len} frames) median-smoothed", flush=True)
    return out


def main():
    print(f"\n##### UMUD pipeline VERSION {PIPELINE_VERSION} #####", flush=True)
    print(f"      scale_router={USE_SCALE_ROUTER}  TTA={USE_TTA}  fasc_pos_weight={FASC_POS_WEIGHT}  "
          f"clahe={USE_CLAHE}  temporal={USE_TEMPORAL_SMOOTH}  epochs={EPOCHS}  min_conf={CALIBRATION_MIN_CONF}  "
          f"fl_identity_blend={FL_IDENTITY_BLEND}",
          flush=True)
    print(f"      model={MODEL_ARCH}/{MODEL_ENCODER} img_size={IMG_SIZE} tta_extra={TTA_EXTRA_SIZE} "
          f"loss={LOSS_MODE} aug={AUG_LEVEL} thresholds apo={APO_MASK_THRESHOLD} fasc={FASC_MASK_THRESHOLD} "
          f"weights_tag={WEIGHTS_TAG or '(default)'}",
          flush=True)
    print("      (old code would NOT print this line - if you don't see VERSION, re-run the wget cell)\n", flush=True)
    test_dir = DIRS["test"]
    test_files = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print(f"test images: {len(test_files)} (from {test_dir})", flush=True)
    if len(test_files) != 309:
        print(f"WARNING: expected 309 test images, found {len(test_files)} -- "
              f"check the resolved test folder above.", flush=True)

    apo = train_segmenter("apo", epochs=EPOCHS)
    fasc = train_segmenter("fasc", epochs=EPOCHS)
    if (USE_CALIBRATED_MT or USE_CALIBRATED_FL) and not CALIBRATION_AVAILABLE:
        print(f"calibration unavailable: {CALIBRATION_IMPORT_ERROR}", flush=True)
    print(f"calibrated MT: {USE_CALIBRATED_MT}, calibrated FL: {USE_CALIBRATED_FL}, "
          f"min calibration confidence: {CALIBRATION_MIN_CONF}, "
          f"fl_mode: {FL_FRAGMENT_MODE}, top_boundary: {TOP_BOUNDARY_MODE}, mt_mode: {MT_MODE}", flush=True)

    rows, calib_rows, ok, mt_ok, fl_ok = [], [], 0, 0, 0
    fps = []
    for p in test_files:
        img = read_rgb(p)
        fps.append(fingerprint(img))
        try:
            geom = measure(predict_mask(apo, img, "apo"), predict_mask(fasc, img, "fasc"))
        except Exception as e:  # no single image can abort the 309-row submission
            print(f"  measure failed on {p.name}: {e}", flush=True)
            geom = None
        pa = geom["pa_deg"] if geom else None
        if pa is None:
            pa = PRIOR["pa_deg"]
        else:
            ok += 1
        pa = float(np.clip(pa, PA_MIN, PA_MAX))  # keep inside the scored 5-45 deg band

        fl_mm = PRIOR["fl_mm"]
        mt_mm = PRIOR["mt_mm"]
        cand = calibrate_image(p) if (USE_CALIBRATED_MT or USE_CALIBRATED_FL) else None
        px_per_mm = None
        calib_conf = 0.0
        calib_method = "none"
        if cand is not None:
            px_per_mm = cand.px_per_mm
            calib_conf = cand.confidence
            calib_method = f"{cand.method}/{cand.edge}"
        if cand is not None and calib_conf >= CALIBRATION_MIN_CONF and geom is not None:
            if USE_CALIBRATED_MT and geom["mt_px"] is not None:
                mt_mm = float(np.clip(geom["mt_px"] / px_per_mm, MT_MIN, MT_MAX))
                mt_ok += 1
            if USE_CALIBRATED_FL and geom["fl_px"] is not None:
                fl_mm = float(np.clip(geom["fl_px"] / px_per_mm, FL_MIN, FL_MAX))
                fl_ok += 1
        # FL: prefer fascicle-fragment extrapolation to the apo lines where we have scale + a fragment
        # (best on the 35 experts once the apo inner-edge fix is in: 0.481->0.353, vs DL-Track 0.312).
        # Else the MT/sin(PA) identity (needs a measured PA), else the prior constant.
        if USE_FRAGMENT_FL and px_per_mm and geom is not None and geom.get("fl_px"):
            fl_mm = float(np.clip(geom["fl_px"] / px_per_mm, FL_MIN, FL_MAX))
            fl_ok += 1
        elif USE_IDENTITY_FL and geom is not None and geom["pa_deg"] is not None:
            fl_mm = float(np.clip(mt_mm / np.sin(np.radians(pa)), FL_MIN, FL_MAX))
            fl_ok += 1
        calib_rows.append({
            "image_id": p.name,
            "px_per_mm": px_per_mm,
            "calibration_confidence": calib_conf,
            "calibration_method": calib_method,
            "scale_spacing_px": getattr(cand, "spacing_px", None) if cand is not None else None,
            "scale_spacing_raw_px": getattr(cand, "spacing_raw_px", None) if cand is not None else None,
            "scale_subpx_resid_rms_px": getattr(cand, "subpx_resid_rms_px", None) if cand is not None else None,
            "scale_subpx_spacing_se": getattr(cand, "subpx_spacing_se", None) if cand is not None else None,
            "scale_subpx_n_ticks": getattr(cand, "subpx_n_ticks", None) if cand is not None else None,
            "scale_subpx_score": getattr(cand, "subpx_score", None) if cand is not None else None,
            "pa_deg": pa,
            "fl_px": geom["fl_px"] if geom else None,
            "mt_px": geom["mt_px"] if geom else None,
            "fl_fragment_mode": FL_FRAGMENT_MODE,
            "top_boundary_mode": geom.get("top_boundary_mode") if geom else TOP_BOUNDARY_MODE,
            "fl_fragment_median_px": geom.get("fl_fragment_median_px") if geom else None,
            "fl_fragment_n": geom.get("fl_fragment_n") if geom else None,
            "fl_mm": fl_mm,
            "mt_mm": mt_mm,
        })
        rows.append({"image_id": p.name, "pa_deg": round(pa, 3),
                     "fl_mm": round(fl_mm, 3), "mt_mm": round(mt_mm, 3)})
    sub = pd.DataFrame(rows)
    if USE_FL_RECENTER and (USE_FRAGMENT_FL or USE_IDENTITY_FL) and sub["fl_mm"].mean() > 0:  # pin FL mean to the prior (UMUD_FL_RECENTER=0 to keep honest per-image FL)
        sub["fl_mm"] = (sub["fl_mm"] * (PRIOR["fl_mm"] / sub["fl_mm"].mean())).clip(FL_MIN, FL_MAX).round(3)
    if USE_TEMPORAL_SMOOTH:  # variance reduction within sequence clips (off by default)
        sub = temporal_smooth(sub, fps)
    out_csv = OUT / "submission_segmentation.csv"
    sub.to_csv(out_csv, index=False)
    pd.DataFrame(calib_rows).to_csv(OUT / "calibration_measurement_debug.csv", index=False)
    print(f"geometry succeeded on {ok}/{len(test_files)} images", flush=True)
    print(f"calibrated MT on {mt_ok}/{len(test_files)}; per-image FL on {fl_ok}/{len(test_files)}", flush=True)
    print(f"wrote {out_csv} ({len(sub)} rows)", flush=True)
    print(sub["pa_deg"].describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
