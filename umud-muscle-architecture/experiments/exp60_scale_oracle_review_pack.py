"""Build a human-oracle review pack for test-set scale guesses.

The detector already assigns a scale to every Kaggle test image, but only part
of that set is independently confirmed by OCR/ruler/text evidence. This script
turns the current scale partition into a review manifest and a small starter
pack so a human can verify a few confident examples plus the unresolved rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT_DIR = RESULTS / "scale_oracle_review"
MANIFEST_OUT = OUT_DIR / "manifest.csv"
START_PACK_OUT = OUT_DIR / "start_pack.csv"


CONFIRMED_TIERS = {"verified", "text-confirmed"}
DETECTOR_TIERS = {"verified", "text-confirmed", "tick-only"}
URGENT_TIERS = {"flag", "mean"}


def _spread_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample rows evenly across the current order without randomness."""
    if len(df) <= n:
        return df.copy()
    idx = [round(i * (len(df) - 1) / (n - 1)) for i in range(n)]
    return df.iloc[sorted(set(idx))].copy()


def _review_group(row: pd.Series) -> str:
    tier = str(row["tier"])
    if tier in URGENT_TIERS:
        return "urgent_oracle"
    if tier == "tick-only":
        return "tick_only_oracle"
    if tier in CONFIRMED_TIERS:
        return "confidence_check"
    return "other"


def _reason(row: pd.Series) -> str:
    tier = str(row["tier"])
    if tier == "verified":
        return "independent ruler/text evidence agrees with detector scale"
    if tier == "text-confirmed":
        return "printed depth/text confirms the detector family"
    if tier == "tick-only":
        return "tick detector gives scale, but OCR did not independently confirm it"
    if tier == "mean":
        return "fallback mean scale; needs human check"
    if tier == "flag":
        return "detector flagged conflict; needs human check"
    return "unclassified scale evidence"


def main() -> None:
    partition_path = RESULTS / "scale_partition.csv"
    if not partition_path.exists():
        raise SystemExit(f"missing {partition_path}; run scale_ocr.py first")

    df = pd.read_csv(partition_path)
    if len(df) != 309:
        print(f"warning: expected 309 rows, found {len(df)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["image_path"] = df["image_id"].map(lambda name: str((TEST_IMAGES / str(name)).resolve()))
    df["exists"] = df["image_path"].map(lambda p: Path(p).exists())
    df["scale_px_per_mm"] = (df["scale_px_per_cm"] / 10.0).round(4)
    df["review_group"] = df.apply(_review_group, axis=1)
    df["reason"] = df.apply(_reason, axis=1)

    scale_med = df["scale_px_per_cm"].median()
    scale_mad = (df["scale_px_per_cm"] - scale_med).abs().median() or 1.0
    df["scale_robust_z"] = ((df["scale_px_per_cm"] - scale_med).abs() / scale_mad).round(3)

    group_rank = {"urgent_oracle": 0, "tick_only_oracle": 1000, "confidence_check": 2000, "other": 3000}
    tier_rank = {"flag": 0, "mean": 1, "tick-only": 2, "text-confirmed": 3, "verified": 4}
    df["priority"] = (
        df["review_group"].map(group_rank).fillna(3000).astype(int)
        + df["tier"].map(tier_rank).fillna(9).astype(int)
    )
    df = df.sort_values(["priority", "scale_robust_z", "image_id"], ascending=[True, False, True])
    df.to_csv(MANIFEST_OUT, index=False)

    urgent = df[df["review_group"] == "urgent_oracle"]
    tick = df[df["review_group"] == "tick_only_oracle"].sort_values(
        ["scale_robust_z", "scale_px_per_cm", "image_id"], ascending=[False, True, True]
    )
    confirmed = df[df["review_group"] == "confidence_check"].sort_values(
        ["tier", "scale_px_per_cm", "image_id"]
    )

    start_pack = pd.concat(
        [
            urgent,
            _spread_sample(tick, 18),
            _spread_sample(confirmed[confirmed["tier"] == "verified"], 4),
            _spread_sample(confirmed[confirmed["tier"] == "text-confirmed"], 4),
        ],
        ignore_index=True,
    )
    start_pack = start_pack.drop_duplicates("image_id").sort_values(
        ["review_group", "scale_robust_z", "image_id"], ascending=[True, False, True]
    )
    start_pack.to_csv(START_PACK_OUT, index=False)

    counts = df["review_group"].value_counts().rename_axis("review_group").reset_index(name="n")
    tiers = df["tier"].value_counts().rename_axis("tier").reset_index(name="n")
    print("\n=== EXP60 scale oracle review pack ===")
    print("tiers:")
    print(tiers.to_string(index=False))
    print("\nreview groups:")
    print(counts.to_string(index=False))
    print(f"\nstart pack rows: {len(start_pack)}")
    print(f"missing image files: {int((~df['exists']).sum())}")
    print(f"\nwrote:\n  {MANIFEST_OUT}\n  {START_PACK_OUT}")


if __name__ == "__main__":
    main()
