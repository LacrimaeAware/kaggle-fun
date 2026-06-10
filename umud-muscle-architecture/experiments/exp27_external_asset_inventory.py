"""Experiment 27: inventory local public/external assets.

This is a bookkeeping experiment. It records what training data, labels,
documentation, licenses, and pretrained weights are already present locally so
the next modeling branch can use them deliberately instead of treating the
external-data path as hypothetical.

Outputs:
    results/external_asset_inventory.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "external_asset_inventory.csv"
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
WEIGHT_EXTS = {".h5", ".pt", ".pth", ".onnx", ".keras"}


def files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file()]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def summarize_dir(path: Path, asset_group: str, role: str, usable_now: bool, notes: str) -> dict:
    files = files_under(path)
    image_files = [p for p in files if p.suffix.lower() in IMG_EXTS]
    mask_like = [p for p in files if "mask" in p.name.lower() or "label" in p.name.lower()]
    return {
        "asset_group": asset_group,
        "path": rel(path),
        "role": role,
        "usable_now": usable_now,
        "files": len(files),
        "image_files": len(image_files),
        "mask_or_label_files": len(mask_like),
        "bytes": sum(p.stat().st_size for p in files),
        "notes": notes,
    }


def paired_count(img_dir: Path, mask_dir: Path) -> int:
    if not img_dir.exists() or not mask_dir.exists():
        return 0
    imgs = {p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    masks = {p.name for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    return len(imgs & masks)


def main():
    rows = [
        summarize_dir(
            DATA / "apo_imgs_v1",
            "public_segmentation_images_a",
            "supervised_training_images",
            True,
            "Pairs with public binary masks for one existing segmentation target.",
        ),
        summarize_dir(
            DATA / "apo_masks_v1",
            "public_segmentation_masks_a",
            "supervised_training_masks",
            True,
            "Ground-truth binary masks paired by filename.",
        ),
        summarize_dir(
            DATA / "fasc_imgs_v1",
            "public_segmentation_images_b",
            "supervised_training_images",
            True,
            "Pairs with public binary masks for the other existing segmentation target.",
        ),
        summarize_dir(
            DATA / "fasc_masks_v1",
            "public_segmentation_masks_b",
            "supervised_training_masks",
            True,
            "Ground-truth binary masks paired by filename.",
        ),
        summarize_dir(
            DATA / "osf_arch_benchmark",
            "public_reference_benchmark",
            "offline_validation",
            True,
            "Reference images and expert measurements used for local scoring.",
        ),
        summarize_dir(
            DATA / "osfstorage-archive",
            "public_osf_archive",
            "external_docs_examples_weights",
            True,
            "Local OSF archive with docs, usage policy, example annotations, and at least one weight file.",
        ),
        summarize_dir(
            DATA / "test_images_v2",
            "competition_target_images",
            "inference_only",
            True,
            "Competition target records. Do not hand-label these for submissions.",
        ),
    ]

    weights = [p for p in files_under(DATA / "osfstorage-archive") if p.suffix.lower() in WEIGHT_EXTS]
    for p in weights:
        rows.append({
            "asset_group": "public_pretrained_weight",
            "path": rel(p),
            "role": "candidate_external_model",
            "usable_now": True,
            "files": 1,
            "image_files": 0,
            "mask_or_label_files": 0,
            "bytes": p.stat().st_size,
            "notes": "Public archive weight file; document/declare before use in a submission path.",
        })

    rows.append({
        "asset_group": "pair_count_a",
        "path": "data/apo_imgs_v1 + data/apo_masks_v1",
        "role": "supervised_training_pair_count",
        "usable_now": True,
        "files": paired_count(
            DATA / "apo_imgs_v1" / "apo_images_new_model_v1",
            DATA / "apo_masks_v1" / "apo_masks_new_model_v1",
        ),
        "image_files": 0,
        "mask_or_label_files": 0,
        "bytes": 0,
        "notes": "Filename-matched public image/mask pairs.",
    })
    rows.append({
        "asset_group": "pair_count_b",
        "path": "data/fasc_imgs_v1 + data/fasc_masks_v1",
        "role": "supervised_training_pair_count",
        "usable_now": True,
        "files": paired_count(
            DATA / "fasc_imgs_v1" / "fasc_images_new_model_v1",
            DATA / "fasc_masks_v1" / "fasc_masks_new_model_v1",
        ),
        "image_files": 0,
        "mask_or_label_files": 0,
        "bytes": 0,
        "notes": "Filename-matched public image/mask pairs.",
    })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df[["asset_group", "files", "image_files", "mask_or_label_files", "usable_now", "role"]].to_string(index=False))
    print(f"\nwrote {OUT}")
    print("\nread: public supervised assets are already local; target records remain inference/pseudo-label only.")


if __name__ == "__main__":
    main()
