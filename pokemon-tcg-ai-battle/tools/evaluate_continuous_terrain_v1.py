"""Validate and summarize Continuous Terrain Representation V1 evaluation JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = ROOT / "docs" / "workstreams" / "continuous_terrain_representation_v1_eval.json"


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(x) -> str:
    if x is None:
        return "-"
    return f"{float(x):.3f}"


def best_high_regret(report: dict) -> list[dict]:
    rows = []
    for name, metrics in (report.get("predictive_metrics", {}).get("high_regret") or {}).items():
        if name.endswith("_eval_only_seeds"):
            continue
        rows.append({
            "representation": name,
            "average_precision": metrics.get("average_precision"),
            "auroc": metrics.get("auroc"),
            "recall_at_fpr_10": metrics.get("recall_at_fpr_10"),
        })
    return sorted(rows, key=lambda r: -1 if r["average_precision"] is None else -float(r["average_precision"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    args = ap.parse_args()
    path = args.eval if args.eval.is_absolute() else ROOT / args.eval
    report = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "dataset", "split", "architecture", "training", "predictive_metrics",
        "signal_radius", "decision", "model_artifacts",
    ]
    missing = [k for k in required if k not in report]
    if missing:
        raise SystemExit(f"missing required keys: {missing}")
    summary = {
        "eval": display(path),
        "mode": report.get("mode"),
        "input": report.get("input"),
        "dataset": report.get("dataset"),
        "decision": report.get("decision"),
        "agent_search_modified": report.get("agent_search_modified"),
        "arena_screen": report.get("arena_screen"),
        "best_high_regret_predictive_rows": best_high_regret(report)[:8],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
