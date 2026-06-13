"""Summarize how much of the 309-image test scale is verified versus inferred.

This is label-free: it uses scale_partition.csv and calibration debug files, not
hidden target labels. The goal is to keep the language precise:

- verified/text-confirmed: independent UI text/ruler evidence supports the scale
- tick-only: detector scale exists, but lacks an independent text/ruler check
- flag/mean: unresolved or fallback rows
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "scale_status_summary.csv"
UNKNOWN_OUT = RESULTS / "scale_status_unverified_rows.csv"


def main() -> None:
    path = RESULTS / "scale_partition.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scale_ocr.py first")
    df = pd.read_csv(path)
    tier_counts = df["tier"].value_counts().rename_axis("tier").reset_index(name="n")
    tier_counts["pct"] = (100 * tier_counts["n"] / len(df)).round(2)

    verified_tiers = {"verified", "text-confirmed"}
    detector_tiers = {"verified", "text-confirmed", "tick-only"}
    summary = pd.DataFrame(
        [
            {
                "category": "independently_confirmed",
                "tiers": "verified + text-confirmed",
                "n": int(df["tier"].isin(verified_tiers).sum()),
                "pct": round(100 * df["tier"].isin(verified_tiers).mean(), 2),
            },
            {
                "category": "detector_scale_available",
                "tiers": "verified + text-confirmed + tick-only",
                "n": int(df["tier"].isin(detector_tiers).sum()),
                "pct": round(100 * df["tier"].isin(detector_tiers).mean(), 2),
            },
            {
                "category": "unresolved_or_fallback",
                "tiers": "flag + mean",
                "n": int((~df["tier"].isin(detector_tiers)).sum()),
                "pct": round(100 * (~df["tier"].isin(detector_tiers)).mean(), 2),
            },
        ]
    )
    unknown = df[~df["tier"].isin(verified_tiers)].copy()
    summary.to_csv(OUT, index=False)
    unknown.to_csv(UNKNOWN_OUT, index=False)

    print("\n=== scale status ===")
    print(tier_counts.to_string(index=False))
    print("\nrollup:")
    print(summary.to_string(index=False))
    print(f"\nwrote:\n  {OUT}\n  {UNKNOWN_OUT}")


if __name__ == "__main__":
    main()
