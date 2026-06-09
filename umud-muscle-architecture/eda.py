"""EDA for the UMUD data: image shapes, mask encoding, calibration metadata, test IDs.

Pennation angle is scale-free, but fascicle length and thickness need a pixel-to-mm
scale, so this checks the TIFF resolution tags (a likely calibration source) and the
mask structure. It also reconciles the true test-image count and IDs against the
sample submission, since the constant baseline assumed 309 rows.

Run from the repository root in the project .venv:
    python umud-muscle-architecture/eda.py
"""

from pathlib import Path

import cv2
import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FOLDERS = ["apo_imgs_v1", "apo_masks_v1", "fasc_imgs_v1", "fasc_masks_v1", "test_images_v2"]


def list_tifs(folder):
    p = DATA / folder
    return sorted(p.rglob("*.tif")) if p.exists() else []


def list_test_images():
    p = DATA / "test_images_v2"
    if not p.exists():
        return []
    return sorted(
        f for f in p.rglob("*") if f.suffix.lower() in {".tif", ".tiff", ".png"}
    )


def imread(path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    return arr


def tiff_tags(path):
    tags = {}
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            for key in ("XResolution", "YResolution", "ResolutionUnit", "ImageDescription"):
                if key in page.tags:
                    v = page.tags[key].value
                    tags[key] = str(v)[:160]
    except Exception as e:
        tags["_err"] = str(e)[:80]
    return tags


def main():
    print("=== folder counts ===")
    for f in FOLDERS:
        print(f"{f}: {len(list_tifs(f))} tif files")

    for folder in FOLDERS:
        files = list_tifs(folder)
        if not files:
            continue
        arr = imread(files[0])
        print(f"\n=== {folder}: {files[0].name} ===")
        print(f"shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
        print("tiff tags:", tiff_tags(files[0]))
        if "mask" in folder:
            vals, counts = np.unique(arr, return_counts=True)
            top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:6]
            print("mask top values (count,val):", top)

    # name pairing
    for kind in ("apo", "fasc"):
        imgs = {f.name for f in list_tifs(f"{kind}_imgs_v1")}
        masks = {f.name for f in list_tifs(f"{kind}_masks_v1")}
        print(f"\n{kind}: imgs {len(imgs)}, masks {len(masks)}, shared names {len(imgs & masks)}")

    # test reconciliation
    test = list_test_images()
    names = sorted(f.name for f in test)
    print(f"\n=== test images: {len(names)} on disk ===")
    print("first:", names[:2], "last:", names[-2:])
    man = DATA / "file_manifest.csv"
    if man.exists():
        txt = man.read_text(errors="ignore")
        n_manifest = sum(
            1
            for ln in txt.splitlines()
            if "test_images_v2" in ln and (".tif" in ln.lower() or ".png" in ln.lower())
        )
        print(f"manifest lists {n_manifest} test_images_v2 image entries")
    samp = DATA / "sample_submission.csv"
    if samp.exists():
        print("sample_submission:", samp.read_text(errors='ignore')[:120].replace(chr(10), ' | '))

    # test image size consistency (global scale only plausible if uniform)
    shapes = {}
    for f in test[:80]:
        s = imread(f).shape
        shapes[s] = shapes.get(s, 0) + 1
    print("\n=== test image shapes (first 80) ===")
    print(shapes)


if __name__ == "__main__":
    main()
