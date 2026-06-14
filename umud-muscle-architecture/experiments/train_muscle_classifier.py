"""Muscle classifier from ultrasound pixels (VL / RF / GM) - Kaggle GPU notebook script.

WHY: 140 of the 309 test images print the muscle (read by OCR -> results/muscle_labels.csv).
The other 169 have no text and sit at the lowest predicted PA (mean 13.2deg). We do not know if
those are genuinely shallow muscles (model correct) or high-pennation muscles being under-read
(model wrong). A classifier that learns the muscle from the IMAGE (not the burned-in text) labels
those 169 so we can tell. It is also the clean, leak-free version of the muscle idea.

DATA: trains on the 140 OCR-labeled test images (longitudinal architecture scans, labels VL/RF/GM),
stratified train/val. Predicts muscle for the 169 'UNK' images. (Optionally fold in the OSF
architecture benchmark images if you attach them and they are muscle-segregated.)

RUN ON KAGGLE: attach the UMUD competition input + this repo's results/muscle_labels.csv, GPU on.
Set TEST_DIR / LABELS_CSV below to the attached paths, then run. Outputs muscle_predictions.csv.

NOTE ON SCORE: the per-muscle correction toward literature means did NOT beat the flat PA shift,
so treat this as a DIAGNOSTIC + learning asset, not a guaranteed score gain. Its value is telling
us what the no-text cluster actually is.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision as tv

# ---- paths (edit for Kaggle: point TEST_DIR at the attached competition test images) ----
TEST_DIR = Path(os.environ.get("UMUD_TEST_DIR", "data/test_images_v2/test_set_v2"))
PNG_GLOB_DIRS = [TEST_DIR, TEST_DIR.parent]              # png family may live alongside
OUT_CSV = Path("results/muscle_predictions.csv") if Path("results").exists() else Path("muscle_predictions.csv")

# muscle labels read by OCR (embedded so no CSV upload is needed on Kaggle)
_LUMIFY = {253:'RF',256:'RF',261:'RF',263:'RF',268:'RF',272:'RF',277:'RF',280:'RF',283:'RF',286:'RF',291:'RF',298:'RF',301:'RF',303:'RF',304:'RF',307:'RF',
254:'GM',257:'GM',258:'GM',264:'GM',265:'GM',266:'GM',270:'GM',273:'GM',275:'GM',278:'GM',281:'GM',284:'GM',287:'GM',292:'GM',305:'GM',308:'GM',309:'GM',
252:'VL',255:'VL',260:'VL',262:'VL',267:'VL',271:'VL',274:'VL',276:'VL',279:'VL',282:'VL',285:'VL',288:'VL',289:'VL',290:'VL',302:'VL',306:'VL'}
def muscle_of(image_id: str) -> str:
    n = int("".join(ch for ch in image_id if ch.isdigit()) or 0)
    if n in _LUMIFY: return _LUMIFY[n]
    if n == 235: return 'GM'                       # Tri Surae
    if 1 <= n <= 35 or 196 <= n <= 251: return 'VL'  # 12L3 Quadriceps
    return 'UNK'
CLASSES = ["VL", "RF", "GM"]
IMG = 224
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def find_file(image_id: str):
    stem = image_id.replace(".tif", "").replace(".png", "")
    for d in PNG_GLOB_DIRS:
        for ext in (".tif", ".tiff", ".png", ".jpg"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_gray_rgb(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (IMG, IMG))
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return im


class MuscleDS(Dataset):
    def __init__(self, rows, train: bool):
        self.rows = rows; self.train = train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, y = self.rows[i]
        im = load_gray_rgb(path).astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5: im = im[:, ::-1].copy()              # horizontal flip
            im = np.clip(im * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.06, 0.06), 0, 1)
        im = (im - 0.5) / 0.5
        return torch.from_numpy(im.transpose(2, 0, 1)), y


def build_rows(df):
    rows = []
    for _, r in df.iterrows():
        p = find_file(r.image_id)
        if p is not None:
            rows.append((p, CLASSES.index(r.muscle)))
    return rows


def main():
    # build labels by scanning the attached test images and applying the embedded OCR map
    import glob
    files = [Path(p).name for d in PNG_GLOB_DIRS for p in glob.glob(str(d / "IMG_*"))]
    files = sorted(set(files))
    labels = pd.DataFrame({"image_id": files})
    labels["muscle"] = labels.image_id.map(muscle_of)
    known = labels[labels.muscle.isin(CLASSES)].reset_index(drop=True)
    unk = labels[~labels.muscle.isin(CLASSES)].reset_index(drop=True)
    print(f"found {len(labels)} test images | labeled {len(known)} ({dict(known.muscle.value_counts())}), to-predict {len(unk)}")

    # stratified split
    tr, va = [], []
    rng = np.random.default_rng(SEED)
    for c in CLASSES:
        idx = known.index[known.muscle == c].to_numpy(); rng.shuffle(idx)
        k = max(1, int(0.2 * len(idx)))
        va += list(idx[:k]); tr += list(idx[k:])
    train_rows = build_rows(known.loc[tr]); val_rows = build_rows(known.loc[va])
    print(f"train {len(train_rows)}  val {len(val_rows)}")

    # class-weighted loss (RF/GM are minorities)
    counts = known.muscle.value_counts().reindex(CLASSES).fillna(0).to_numpy() + 1
    w = torch.tensor((counts.sum() / counts), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)

    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tl = DataLoader(MuscleDS(train_rows, True), batch_size=16, shuffle=True)
    vl = DataLoader(MuscleDS(val_rows, False), batch_size=32)

    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        # val acc
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(DEVICE)).argmax(1).cpu()
                correct += (p == y).sum().item(); tot += len(y)
        acc = correct / max(tot, 1)
        if acc >= best: best = acc; torch.save(model.state_dict(), "results/muscle_clf.pt")
        print(f"ep {ep:02d}  val_acc {acc:.3f}  (best {best:.3f})")

    # predict the no-text images
    model.load_state_dict(torch.load("results/muscle_clf.pt")); model.eval()
    preds = []
    with torch.no_grad():
        for _, r in unk.iterrows():
            p = find_file(r.image_id)
            if p is None: preds.append((r.image_id, "UNK", 0.0)); continue
            im = load_gray_rgb(p).astype(np.float32) / 255.0
            im = (im - 0.5) / 0.5
            t = torch.from_numpy(im.transpose(2, 0, 1))[None].to(DEVICE)
            prob = model(t).softmax(1)[0].cpu().numpy()
            preds.append((r.image_id, CLASSES[int(prob.argmax())], float(prob.max())))
    out = pd.DataFrame(preds, columns=["image_id", "muscle_pred", "confidence"])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nval best {best:.3f}.  predicted no-text muscles:")
    print(out.muscle_pred.value_counts().to_string())
    print(f"wrote {OUT_CSV}. Next: join muscle_pred to the 169 and check if the predicted-GM/VL ones "
          f"(high-pennation) sit at low PA -> they are the under-read cluster.")


if __name__ == "__main__":
    main()
