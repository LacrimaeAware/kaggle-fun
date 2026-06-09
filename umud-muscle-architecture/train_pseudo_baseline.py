"""Train a first CPU image-feature baseline against mask-derived pseudo-labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from mask_geometry import DATA, HERE, build_geometry_table


TARGET_COLUMNS = ("pa_deg", "fl_px", "mt_px")
SUBMISSION_COLUMNS = ("pa_deg", "fl_mm", "mt_mm")
SAMPLE_MEAN_MM = {
    "pa_deg": 15.105,
    "fl_mm": 74.424,
    "mt_mm": 18.628,
}


def read_image(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"OpenCV failed to read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


def image_features(path: Path) -> np.ndarray:
    gray = read_image(path)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny((blurred * 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0

    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(float(hist.sum()), 1.0)

    row_profile = gray.mean(axis=1)
    col_profile = gray.mean(axis=0)
    edge_row_profile = edges.mean(axis=1)

    return np.concatenate(
        [
            gray.ravel(),
            edges.ravel(),
            hist.astype(np.float32),
            row_profile.astype(np.float32),
            col_profile.astype(np.float32),
            edge_row_profile.astype(np.float32),
        ]
    )


def train_image_paths(data_dir: Path, image_ids: pd.Series) -> list[Path]:
    image_dir = data_dir / "fasc_imgs_v1" / "fasc_images_new_model_v1"
    return [image_dir / image_id for image_id in image_ids]


def test_image_paths(data_dir: Path) -> list[Path]:
    test_dir = data_dir / "test_images_v2" / "test_set_v2"
    return sorted(
        path
        for path in test_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff", ".png"}
    )


def feature_matrix(paths: list[Path]) -> np.ndarray:
    return np.vstack([image_features(path) for path in paths])


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> MultiOutputRegressor:
    base = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.35,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base, n_jobs=None)
    model.fit(x_train, y_train)
    return model


def scaled_submission_predictions(
    pred: np.ndarray,
    train_targets: pd.DataFrame,
) -> np.ndarray:
    values = pred.copy()
    values[:, 0] = np.clip(values[:, 0], 5.0, 45.0)

    fl_scale = SAMPLE_MEAN_MM["fl_mm"] / float(train_targets["fl_px"].median())
    mt_scale = SAMPLE_MEAN_MM["mt_mm"] / float(train_targets["mt_px"].median())
    values[:, 1] = np.clip(values[:, 1] * fl_scale, 30.0, 200.0)
    values[:, 2] = np.clip(values[:, 2] * mt_scale, 10.0, 50.0)
    return values


def write_prediction_file(paths: list[Path], values: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("image_id", *SUBMISSION_COLUMNS))
        writer.writeheader()
        for path, row in zip(paths, values, strict=True):
            writer.writerow(
                {
                    "image_id": path.name,
                    "pa_deg": f"{row[0]:.3f}",
                    "fl_mm": f"{row[1]:.3f}",
                    "mt_mm": f"{row[2]:.3f}",
                }
            )


def median_baseline_metrics(y_val: np.ndarray, y_train: np.ndarray) -> list[dict[str, float | str]]:
    med = np.median(y_train, axis=0)
    pred = np.repeat(med[None, :], len(y_val), axis=0)
    rows: list[dict[str, float | str]] = []
    for i, col in enumerate(TARGET_COLUMNS):
        rows.append(
            {
                "target": col,
                "model": "median",
                "mae": mean_absolute_error(y_val[:, i], pred[:, i]),
                "r2": r2_score(y_val[:, i], pred[:, i]),
                "val_median": float(np.median(y_val[:, i])),
                "pred_median": float(np.median(pred[:, i])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--force-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.results_dir / "mask_geometry_pseudolabels.csv"
    if labels_path.exists() and not args.force_labels:
        labels = pd.read_csv(labels_path)
    else:
        labels = build_geometry_table(args.data_dir)
        labels.to_csv(labels_path, index=False)

    labels = labels.dropna(subset=list(TARGET_COLUMNS)).reset_index(drop=True)
    paths = train_image_paths(args.data_dir, labels["image_id"])
    x = feature_matrix(paths)
    y = labels[list(TARGET_COLUMNS)].to_numpy(dtype=float)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
    )
    model = fit_model(x[train_idx], y[train_idx])
    val_pred = model.predict(x[val_idx])

    rows = median_baseline_metrics(y[val_idx], y[train_idx])
    for i, col in enumerate(TARGET_COLUMNS):
        rows.append(
            {
                "target": col,
                "model": "extra_trees",
                "mae": mean_absolute_error(y[val_idx, i], val_pred[:, i]),
                "r2": r2_score(y[val_idx, i], val_pred[:, i]),
                "val_median": float(np.median(y[val_idx, i])),
                "pred_median": float(np.median(val_pred[:, i])),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics_path = args.results_dir / "pseudo_baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    final_model = fit_model(x, y)
    test_paths = test_image_paths(args.data_dir)
    test_x = feature_matrix(test_paths)
    test_pred = final_model.predict(test_x)
    sub_values = scaled_submission_predictions(test_pred, labels)
    prediction_path = args.results_dir / "submission_pseudo_baseline_comma_309.csv"
    write_prediction_file(test_paths, sub_values, prediction_path)

    hybrid_values = sub_values.copy()
    hybrid_values[:, 1] = SAMPLE_MEAN_MM["fl_mm"]
    hybrid_values[:, 2] = SAMPLE_MEAN_MM["mt_mm"]
    hybrid_path = args.results_dir / "submission_pseudo_pa_constant_flmt_comma_309.csv"
    write_prediction_file(test_paths, hybrid_values, hybrid_path)

    print(f"labels: {len(labels)} -> {labels_path}")
    print(f"metrics -> {metrics_path}")
    print(metrics.to_string(index=False))
    print(f"prediction rows: {len(test_paths)} -> {prediction_path}")
    print(f"hybrid rows: {len(test_paths)} -> {hybrid_path}")


if __name__ == "__main__":
    main()
