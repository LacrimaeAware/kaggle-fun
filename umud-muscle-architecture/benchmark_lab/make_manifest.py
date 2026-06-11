"""Create a manifest for the local human benchmark labeling app.

The manifest is deliberately plain CSV so it can be reviewed, edited, committed, or thrown away.
Labels themselves should usually live under results/human_benchmark/, which is gitignored.

Examples:
    python benchmark_lab/make_manifest.py --fallmud 24 --target 0
    python benchmark_lab/make_manifest.py --target 24 --fallmud 0 --include IMG_00275
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def find_target_images() -> list[Path]:
    folder = first_existing([
        ROOT / "data" / "test_images_v2" / "test_set_v2",
        ROOT / "data" / "test_images_v2",
    ])
    return image_files(folder) if folder else []


def find_fallmud_rows() -> list[dict[str, str]]:
    base = ROOT / "data" / "dropoff" / "FALLMUD"
    rows: list[dict[str, str]] = []
    if not base.exists():
        return rows
    for provider in sorted(p for p in base.iterdir() if p.is_dir()):
        img_dir = provider / "images"
        apo_dir = provider / "aponeurosis_masks"
        fasc_dir = provider / "fascicle_masks"
        for img in image_files(img_dir):
            apo = next(iter(sorted(apo_dir.glob(img.stem + ".*"))), None)
            fasc = next(iter(sorted(fasc_dir.glob(img.stem + ".*"))), None)
            rows.append({
                "label_id": safe_id(f"fallmud_{provider.name}_{img.stem}"),
                "source": f"fallmud/{provider.name}",
                "image_id": img.stem,
                "image_path": str(img.resolve()),
                "reference_apo_mask_path": str(apo.resolve()) if apo else "",
                "reference_fasc_mask_path": str(fasc.resolve()) if fasc else "",
                "scale_px_per_mm": "",
                "label_mode": "public_external",
                "priority": "normal",
                "notes": "Existing public masks are available for convention/scorer sanity checks.",
            })
    return rows


def find_osf_benchmark_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    base = ROOT / "data" / "osf_arch_benchmark"
    if not base.exists():
        return rows
    for img in image_files(base):
        # Avoid xlsx-adjacent thumbnails or generated files if the archive contains any.
        if "mask" in str(img).lower():
            continue
        rows.append({
            "label_id": safe_id(f"osf_benchmark_{img.stem}"),
            "source": "osf_benchmark",
            "image_id": img.stem,
            "image_path": str(img.resolve()),
            "reference_apo_mask_path": "",
            "reference_fasc_mask_path": "",
            "scale_px_per_mm": "",
            "label_mode": "public_external",
            "priority": "normal",
            "notes": "Public 35-expert benchmark image; measurements exist in xlsx, masks may not.",
        })
    return rows


def target_rows(include_ids: set[str]) -> list[dict[str, str]]:
    rows = []
    for img in find_target_images():
        priority = "requested" if img.stem in include_ids else "normal"
        rows.append({
            "label_id": safe_id(f"target_{img.stem}"),
            "source": "competition_target",
            "image_id": img.stem,
            "image_path": str(img.resolve()),
            "reference_apo_mask_path": "",
            "reference_fasc_mask_path": "",
            "scale_px_per_mm": "",
            "label_mode": "declared_human_in_loop",
            "priority": priority,
            "notes": "Target-image human label. Keep separate and declare if used.",
        })
    return rows


def sample_rows(rows: list[dict[str, str]], n: int, seed: int, include_ids: set[str]) -> list[dict[str, str]]:
    if n <= 0:
        return []
    requested = [r for r in rows if r["image_id"] in include_ids or r["label_id"] in include_ids]
    rest = [r for r in rows if r not in requested]
    rng = random.Random(seed)
    rng.shuffle(rest)
    return (requested + rest)[: min(n, len(requested) + len(rest))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "human_benchmark" / "manifest.csv"))
    ap.add_argument("--target", type=int, default=0, help="number of competition target images to include")
    ap.add_argument("--fallmud", type=int, default=24, help="number of FALLMUD/public images to include")
    ap.add_argument("--osf-benchmark", type=int, default=0, help="number of OSF benchmark images to include")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include", action="append", default=[], help="image_id or label_id to force-include")
    args = ap.parse_args()

    include_ids = set(args.include)
    rows: list[dict[str, str]] = []
    rows.extend(sample_rows(find_fallmud_rows(), args.fallmud, args.seed, include_ids))
    rows.extend(sample_rows(find_osf_benchmark_rows(), args.osf_benchmark, args.seed + 1, include_ids))
    rows.extend(sample_rows(target_rows(include_ids), args.target, args.seed + 2, include_ids))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label_id",
        "source",
        "image_id",
        "image_path",
        "reference_apo_mask_path",
        "reference_fasc_mask_path",
        "scale_px_per_mm",
        "label_mode",
        "priority",
        "notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(f"wrote {len(rows)} rows -> {out}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
