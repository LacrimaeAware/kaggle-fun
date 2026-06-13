"""Summarize viewer models by geometry class.

This is a reporting helper for the class-aware viewer. It scores every model
currently exposed by the expert benchmark API overall and within each class, so
we can see which models are broadly good versus class-specific.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_lab"))

import review_server as RS  # noqa: E402

OUT = ROOT / "results" / "exp54_model_class_matrix"


def term_norm(model: dict, term: str) -> float | None:
    delta = model.get("deltas", {}).get(term)
    if delta is None:
        return None
    tol = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}[term]
    return abs(float(delta)) / tol


def mean(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def model_scores_for_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        for model in row["models"]:
            out.append({
                "image_id": row["image_id"],
                "model_id": model["id"],
                "label": model["label"],
                "overall": model.get("overall_norm"),
                "pa": term_norm(model, "pa_deg"),
                "fl": term_norm(model, "fl_mm"),
                "mt": term_norm(model, "mt_mm"),
            })
    return out


def summarize(df: pd.DataFrame, label: str, class_name: str, n: int) -> list[dict]:
    rows = []
    for (model_id, model_label), g in df.groupby(["model_id", "label"], dropna=False):
        rows.append({
            "slice": label,
            "class": class_name,
            "n_images": n,
            "model_id": model_id,
            "model_label": model_label,
            "overall": g["overall"].mean(),
            "pa": g["pa"].mean(),
            "fl": g["fl"].mean(),
            "mt": g["mt"].mean(),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows([], candidates, ROOT / "results" / "human_benchmark" / "review_notes")
    RS.enrich_rows_for_v2(rows, summary)
    score_df = pd.DataFrame(model_scores_for_rows(rows))
    score_df.to_csv(OUT / "per_image_model_scores.csv", index=False)

    all_rows = summarize(score_df, "all", "all", rows and len(rows) or 0)
    class_rows = []
    class_names = sorted({name for row in rows for name, enabled in (row.get("class_flags") or {}).items() if enabled})
    for class_name in class_names:
        ids = [row["image_id"] for row in rows if row.get("class_flags", {}).get(class_name)]
        if not ids:
            continue
        sub = score_df[score_df["image_id"].isin(ids)]
        class_rows.extend(summarize(sub, "class", class_name, len(ids)))
    matrix = pd.DataFrame(all_rows + class_rows)
    matrix = matrix.sort_values(["slice", "class", "overall", "model_id"])
    matrix.to_csv(OUT / "model_class_matrix.csv", index=False)

    winners = []
    for (slice_name, class_name), g in matrix.groupby(["slice", "class"], dropna=False):
        for term in ("overall", "pa", "fl", "mt"):
            best = g.sort_values(term).iloc[0]
            winners.append({
                "slice": slice_name,
                "class": class_name,
                "n_images": int(best["n_images"]),
                "term": term,
                "best_model_id": best["model_id"],
                "best_model_label": best["model_label"],
                "score": best[term],
            })
    winners_df = pd.DataFrame(winners).sort_values(["slice", "class", "term"])
    winners_df.to_csv(OUT / "class_winners.csv", index=False)

    print("\n=== exp54 model class matrix ===")
    print("\nOverall model ranking:")
    print(matrix[matrix["slice"].eq("all")][["model_id", "overall", "pa", "fl", "mt"]].sort_values("overall").to_string(index=False))
    print("\nClass overall winners:")
    print(winners_df[winners_df["term"].eq("overall")][["class", "n_images", "best_model_id", "score"]].to_string(index=False))
    print(f"\nwrote bundle: {OUT}")


if __name__ == "__main__":
    main()
