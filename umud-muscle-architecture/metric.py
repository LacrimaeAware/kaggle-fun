"""Local UMUD Score implementation for validation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("pa_deg", "fl_mm", "mt_mm")
TOLERANCES = {
    "pa_deg": 6.0,
    "fl_mm": 12.0,
    "mt_mm": 3.0,
}
WEIGHTS = {
    "pa_deg": 1.0,
    "fl_mm": 1.0,
    "mt_mm": 1.0,
}
EPS_SECONDARY = 1e-6
EPS_TERTIARY = 1e-9


class MetricInputError(ValueError):
    """Raised when solution or submission data cannot be scored."""


def umud_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> float:
    """Compute the UMUD Score, lower is better."""
    if row_id_column_name not in solution.columns:
        raise MetricInputError(f"Solution is missing '{row_id_column_name}'.")
    if row_id_column_name not in submission.columns:
        raise MetricInputError(f"Submission is missing '{row_id_column_name}'.")

    for col in TARGET_COLUMNS:
        if col not in solution.columns:
            raise MetricInputError(f"Solution is missing '{col}'.")
        if col not in submission.columns:
            raise MetricInputError(f"Submission is missing '{col}'.")

    if solution[row_id_column_name].duplicated().any():
        raise MetricInputError("Solution contains duplicate IDs.")
    if submission[row_id_column_name].duplicated().any():
        raise MetricInputError("Submission contains duplicate IDs.")

    merged = solution.merge(
        submission,
        on=row_id_column_name,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if len(merged) != len(solution):
        raise MetricInputError(f"Submission is missing {len(solution) - len(merged)} rows.")

    primary = 0.0
    secondary = 0.0
    tertiary = 0.0
    weight_sum = float(sum(WEIGHTS[col] for col in TARGET_COLUMNS))

    for col in TARGET_COLUMNS:
        pred_col = f"{col}_pred"
        true_col = f"{col}_true"
        merged[pred_col] = pd.to_numeric(merged[pred_col], errors="coerce")
        if merged[pred_col].isna().any():
            raise MetricInputError(f"Column '{col}' contains missing or non-numeric values.")

        y_true = merged[true_col].to_numpy(dtype=float)
        y_pred = merged[pred_col].to_numpy(dtype=float)
        if not np.isfinite(y_pred).all():
            raise MetricInputError(f"Column '{col}' contains infinite values.")

        err = y_pred - y_true
        normalized_abs = np.abs(err) / TOLERANCES[col]
        normalized_sq = (err / TOLERANCES[col]) ** 2
        weight = WEIGHTS[col]

        primary += weight * float(np.mean(normalized_abs))
        secondary += weight * float(np.median(normalized_abs))
        tertiary += weight * float(np.sqrt(np.mean(normalized_sq)))

    primary /= weight_sum
    secondary /= weight_sum
    tertiary /= weight_sum
    return primary + EPS_SECONDARY * secondary + EPS_TERTIARY * tertiary


def local_metric_report(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> pd.DataFrame:
    """Return per-target MAE, MedAE, RMSE, bias, and normalized MAE."""
    merged = solution.merge(
        submission,
        on=row_id_column_name,
        how="inner",
        suffixes=("_true", "_pred"),
    )
    rows = []
    for col in TARGET_COLUMNS:
        y_true = merged[f"{col}_true"].to_numpy(dtype=float)
        y_pred = pd.to_numeric(merged[f"{col}_pred"], errors="coerce").to_numpy(dtype=float)
        err = y_pred - y_true
        rows.append(
            {
                "target": col,
                "mae": float(np.mean(np.abs(err))),
                "medae": float(np.median(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "normalized_mae": float(np.mean(np.abs(err)) / TOLERANCES[col]),
            }
        )
    return pd.DataFrame(rows)
