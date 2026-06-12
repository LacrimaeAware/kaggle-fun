"""Robust consensus helpers for the 35-image expert benchmark.

The source spreadsheet is left untouched. These helpers only affect local
benchmark scoring/viewing by dropping a single obvious rater tail when the
remaining raters form a tight cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Iterable


RATERS = [f"R{i}" for i in range(1, 8)]
TOL_BY_SUFFIX = {"PA": 6.0, "FL": 12.0, "MT": 3.0}


@dataclass(frozen=True)
class DroppedExpertValue:
    rater: str
    value: float
    raw_mean: float
    robust_mean: float
    other_mean: float
    other_range: float
    distance_to_other_mean: float
    metric_suffix: str


def parse_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if isfinite(out) else None


def robust_mean(
    values: Iterable[tuple[str, object]],
    metric_suffix: str,
    *,
    outlier_distance_tol: float = 2.0,
    cluster_range_tol: float = 2.0,
) -> tuple[float | None, DroppedExpertValue | None]:
    """Return mean after dropping at most one obvious single-rater tail.

    A value is dropped only if it is at least two competition tolerances away
    from the other-rater mean and the other raters span no more than two
    tolerances. This catches the clear spreadsheet pathologies without
    rewriting broad expert disagreement as if it were a typo.
    """

    tol = TOL_BY_SUFFIX[metric_suffix]
    pairs = [(name, parsed) for name, value in values if (parsed := parse_float(value)) is not None]
    if not pairs:
        return None, None
    raw_values = [value for _, value in pairs]
    raw_mean = mean(raw_values)
    if len(pairs) < 3:
        return raw_mean, None

    candidates: list[tuple[float, float, DroppedExpertValue]] = []
    for name, value in pairs:
        others = [other_value for other_name, other_value in pairs if other_name != name]
        if len(others) < 2:
            continue
        other_mean = mean(others)
        other_range = max(others) - min(others)
        distance = abs(value - other_mean)
        if distance / tol >= outlier_distance_tol and other_range / tol <= cluster_range_tol:
            dropped = DroppedExpertValue(
                rater=name,
                value=value,
                raw_mean=raw_mean,
                robust_mean=other_mean,
                other_mean=other_mean,
                other_range=other_range,
                distance_to_other_mean=distance,
                metric_suffix=metric_suffix,
            )
            candidates.append((distance / tol, abs(raw_mean - other_mean) / tol, dropped))

    if not candidates:
        return raw_mean, None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = candidates[0][2]
    return chosen.robust_mean, chosen


def robust_mean_for_row(row: dict, metric_suffix: str) -> tuple[float | None, DroppedExpertValue | None]:
    return robust_mean(((rater, row.get(f"{rater}_{metric_suffix}")) for rater in RATERS), metric_suffix)
