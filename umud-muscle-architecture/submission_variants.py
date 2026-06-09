"""Write ID-format variants for a UMUD prediction CSV without changing predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def no_ext_upper(image_id: str) -> str:
    return Path(image_id).stem.upper()


def no_ext_lower(image_id: str) -> str:
    return Path(image_id).stem.lower()


def all_tif(image_id: str) -> str:
    return f"{Path(image_id).stem}.tif"


def page_example_lower(image_id: str) -> str:
    stem = Path(image_id).stem
    if stem.upper().startswith("IMG_"):
        return "image_" + stem.split("_", 1)[1]
    return stem.lower()


VARIANTS = {
    "no_ext_upper": no_ext_upper,
    "no_ext_lower": no_ext_lower,
    "page_example_lower": page_example_lower,
    "all_tif": all_tif,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=HERE / "results" / "submission_pseudo_pa_constant_flmt_comma_309.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "variants")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    if "image_id" not in df.columns:
        raise ValueError(f"{source} has no image_id column")

    for name, transform in VARIANTS.items():
        out = df.copy()
        out["image_id"] = out["image_id"].map(transform)
        out_path = args.output_dir / f"{source.stem}_{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
