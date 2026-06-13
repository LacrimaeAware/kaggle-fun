"""Greedy class-aware route search over viewer models.

This intentionally overfits the 35-image expert benchmark to estimate how much
headroom exists in class-aware routing. It is a diagnostic upper bound, not a
production rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_lab"))

import benchmark_validate as BV  # noqa: E402
import review_server as RS  # noqa: E402

OUT = ROOT / "results" / "exp55_class_route_search"
TERMS = ("pa_deg", "fl_mm", "mt_mm")
DISALLOWED_ROUTE_MODELS = {"DLTrack", "SMA", "our_pipeline_true_scale"}


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in TERMS:
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


def build_model_frames() -> tuple[list[dict], pd.DataFrame, dict[str, pd.DataFrame]]:
    truth, _floor = BV.load_truth()
    candidates = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows([], candidates, ROOT / "results" / "human_benchmark" / "review_notes")
    RS.enrich_rows_for_v2(rows, summary)
    frames: dict[str, list[dict]] = {}
    for row in rows:
        for model in row["models"]:
            pred = model.get("predictions") or {}
            if all(pred.get(term) is not None for term in TERMS):
                frames.setdefault(model["id"], []).append({
                    "image_id": row["image_id"],
                    **{term: pred[term] for term in TERMS},
                })
    out = {name: pd.DataFrame(records).set_index("image_id") for name, records in frames.items()}
    return rows, truth, out


def gate_series(rows: list[dict], class_name: str, invert: bool = False) -> pd.Series:
    values = {row["image_id"]: bool(row.get("class_flags", {}).get(class_name)) for row in rows}
    s = pd.Series(values)
    return ~s if invert else s


def candidate_gates(rows: list[dict]) -> list[tuple[str, pd.Series]]:
    class_names = sorted({name for row in rows for name, enabled in (row.get("class_flags") or {}).items() if enabled})
    gates = []
    for name in class_names:
        s = gate_series(rows, name)
        if 0 < int(s.sum()) < len(s):
            gates.append((name, s))
            gates.append((f"not_{name}", ~s))
    return gates


def greedy_route(
    base: pd.DataFrame,
    truth: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    gates: list[tuple[str, pd.Series]],
    term_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = base.copy()
    current_score = score_frame(current.reset_index(), truth)["overall"]
    steps = []
    for step in range(20):
        best = None
        for gate_name, gate in gates:
            gate = gate.reindex(current.index).fillna(False).astype(bool)
            if int(gate.sum()) == 0:
                continue
            for model_id, frame in frames.items():
                frame = frame.reindex(current.index)
                if term_level:
                    for term in TERMS:
                        trial = current.copy()
                        trial.loc[gate, term] = frame.loc[gate, term]
                        score = score_frame(trial.reset_index(), truth)
                        gain = current_score - score["overall"]
                        if best is None or gain > best["gain"]:
                            best = {"trial": trial, "score": score, "gain": gain, "gate": gate_name, "model_id": model_id, "term": term, "n": int(gate.sum())}
                else:
                    trial = current.copy()
                    trial.loc[gate, list(TERMS)] = frame.loc[gate, list(TERMS)]
                    score = score_frame(trial.reset_index(), truth)
                    gain = current_score - score["overall"]
                    if best is None or gain > best["gain"]:
                        best = {"trial": trial, "score": score, "gain": gain, "gate": gate_name, "model_id": model_id, "term": "all", "n": int(gate.sum())}
        if best is None or best["gain"] <= 1e-6:
            break
        current = best["trial"]
        current_score = best["score"]["overall"]
        steps.append({
            "step": step + 1,
            "gain": best["gain"],
            "overall": current_score,
            "pa": best["score"]["pa_deg"],
            "fl": best["score"]["fl_mm"],
            "mt": best["score"]["mt_mm"],
            "gate": best["gate"],
            "n": best["n"],
            "model_id": best["model_id"],
            "term": best["term"],
        })
    return current, pd.DataFrame(steps)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, truth, frames = build_model_frames()
    allowed_frames = {
        model_id: frame
        for model_id, frame in frames.items()
        if model_id not in DISALLOWED_ROUTE_MODELS
    }
    base_id = "median_weight_blend_best" if "median_weight_blend_best" in frames else "story_weight_grid_best"
    base = allowed_frames[base_id]
    gates = candidate_gates(rows)

    model_route, model_steps = greedy_route(base, truth, allowed_frames, gates, term_level=False)
    term_route, term_steps = greedy_route(base, truth, allowed_frames, gates, term_level=True)
    model_route.reset_index().to_csv(OUT / "best_model_route.csv", index=False)
    term_route.reset_index().to_csv(OUT / "best_term_route.csv", index=False)
    model_steps.to_csv(OUT / "model_route_steps.csv", index=False)
    term_steps.to_csv(OUT / "term_route_steps.csv", index=False)

    base_score = score_frame(base.reset_index(), truth)
    model_score = score_frame(model_route.reset_index(), truth)
    term_score = score_frame(term_route.reset_index(), truth)
    summary = pd.DataFrame([
        {"route": "base", "model": base_id, **base_score},
        {"route": "model_level_greedy", "model": "class route", **model_score},
        {"route": "term_level_greedy", "model": "class+term route", **term_score},
    ])
    summary.to_csv(OUT / "summary.csv", index=False)

    print("\n=== exp55 class route search ===")
    print(summary[["route", "overall", "pa_deg", "fl_mm", "mt_mm"]].to_string(index=False))
    print("\nModel-level route steps:")
    print(model_steps.to_string(index=False) if not model_steps.empty else "no improving steps")
    print("\nTerm-level route steps:")
    print(term_steps.to_string(index=False) if not term_steps.empty else "no improving steps")
    print(f"\nwrote bundle: {OUT}")


if __name__ == "__main__":
    main()
