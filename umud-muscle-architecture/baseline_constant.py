"""Create a valid constant baseline submission for the UMUD challenge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TARGET_COLUMNS = ("pa_deg", "fl_mm", "mt_mm")
TEST_PREFIX = "test_images_v2/test_set_v2/"
RANGE_MIDPOINTS = {
    "pa_deg": 25.0,
    "fl_mm": 115.0,
    "mt_mm": 30.0,
}


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "submission_constant_comma_309.csv"


def test_image_ids(data_dir: Path) -> list[str]:
    """Return sorted test image file names from a Kaggle manifest or local files."""
    manifest_path = data_dir / "file_manifest.csv"
    if manifest_path.exists():
        image_ids: list[str] = []
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["name"]
                lower = name.lower()
                if name.startswith(TEST_PREFIX) and lower.endswith((".tif", ".tiff", ".png")):
                    image_ids.append(Path(name).name)
        return sorted(image_ids)

    test_dir = data_dir / "test_images_v2" / "test_set_v2"
    if test_dir.exists():
        return sorted(
            path.name
            for path in test_dir.iterdir()
            if path.suffix.lower() in {".tif", ".tiff", ".png"}
        )

    raise FileNotFoundError(
        "No test images found. Expected data/file_manifest.csv or "
        "data/test_images_v2/test_set_v2/."
    )


def sample_submission_values(data_dir: Path) -> dict[str, float]:
    """Use the sample submission target means as a cheap format baseline."""
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample submission: {sample_path}")

    rows: list[dict[str, str]] = []
    with sample_path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f, delimiter=";"))

    if not rows:
        raise ValueError(f"No rows found in {sample_path}")

    return {
        col: sum(float(row[col]) for row in rows) / len(rows)
        for col in TARGET_COLUMNS
    }


def write_constant_submission(
    image_ids: list[str],
    values: dict[str, float],
    output_path: Path,
    delimiter: str = ",",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("image_id", *TARGET_COLUMNS), delimiter=delimiter)
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow(
                {
                    "image_id": image_id,
                    "pa_deg": f"{values['pa_deg']:.3f}",
                    "fl_mm": f"{values['fl_mm']:.3f}",
                    "mt_mm": f"{values['mt_mm']:.3f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument(
        "--delimiter",
        choices=("comma", "semicolon"),
        default="comma",
        help="Kaggle's CSV parser expects comma. The provided sample file uses semicolons.",
    )
    parser.add_argument(
        "--strategy",
        choices=("sample-mean", "range-midpoint"),
        default="sample-mean",
        help="Constant values to use for every test image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()

    image_ids = test_image_ids(data_dir)
    if args.strategy == "sample-mean":
        values = sample_submission_values(data_dir)
    else:
        values = RANGE_MIDPOINTS

    delimiter = "," if args.delimiter == "comma" else ";"
    write_constant_submission(image_ids, values, output_path, delimiter=delimiter)

    print(f"wrote {output_path}")
    print(f"rows: {len(image_ids)}")
    print(f"delimiter: {args.delimiter}")
    print(
        "values: "
        + ", ".join(f"{col}={values[col]:.3f}" for col in TARGET_COLUMNS)
    )


if __name__ == "__main__":
    main()
