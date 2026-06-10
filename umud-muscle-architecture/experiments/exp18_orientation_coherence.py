"""Report fragment orientation coherence by control/target family.

The domain-gap probes showed target masks do not collapse, but presence is not
correctness. This script asks whether predicted fascicle fragments agree on a
single orientation within each image.

High coherence (near 1) means the extra fragments are aligned with the dominant
pennate structure. Low coherence suggests texture/noise pickup.

Run:
    python experiments/exp18_orientation_coherence.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import segment_then_measure as M  # noqa: E402
from exp16_fl_combiner import load, measure_components  # noqa: E402

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TRAIN_FASC = ROOT / "data" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
TEST_DIR = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "orientation_coherence.csv"


def family_for_path(p, debug_methods):
    if p.name in debug_methods:
        return debug_methods[p.name]
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return "unknown"
    return f"{g.shape[1]}x{g.shape[0]}"


def analyze_paths(apo, fasc, paths, group, rows, debug_methods=None):
    debug_methods = debug_methods or {}
    for p in paths:
        img = M.read_rgb(p)
        try:
            g = measure_components(M.predict_mask(apo, img), M.predict_mask(fasc, img))
        except Exception:
            g = None
        if g is None:
            rows.append({
                "group": group,
                "family": family_for_path(p, debug_methods),
                "image_id": p.name,
                "ok": False,
                "n_frag": 0,
                "coherence": np.nan,
                "pa_med": np.nan,
                "fl_ratio_identity_to_fragment": np.nan,
            })
            continue
        ratio = np.nan
        if g.get("fl_fragment_median_px") and g.get("fl_identity_gated_px"):
            ratio = float(g["fl_identity_gated_px"] / g["fl_fragment_median_px"])
        rows.append({
            "group": group,
            "family": family_for_path(p, debug_methods),
            "image_id": p.name,
            "ok": True,
            "n_frag": int(g["n_frag"]),
            "coherence": float(g["coherence"]),
            "pa_med": float(g["pa_med"]),
            "fl_ratio_identity_to_fragment": ratio,
        })


def summarize(df):
    groups = []
    for (group, family), sub in df.groupby(["group", "family"], dropna=False):
        ok = sub[sub["ok"]]
        groups.append({
            "group": group,
            "family": family,
            "n": len(sub),
            "ok_pct": 100.0 * len(ok) / max(len(sub), 1),
            "frag_mean": ok["n_frag"].mean(),
            "coh_mean": ok["coherence"].mean(),
            "coh_p10": ok["coherence"].quantile(0.10),
            "coh_min": ok["coherence"].min(),
            "pa_mean": ok["pa_med"].mean(),
            "ratio_mean": ok["fl_ratio_identity_to_fragment"].mean(),
        })
    return pd.DataFrame(groups).sort_values(["group", "family"])


def main():
    rng = np.random.default_rng(0)
    apo, fasc = load("apo"), load("fasc")
    rows = []

    train = sorted(p for p in TRAIN_FASC.iterdir() if p.suffix.lower() in EXTS)
    train = [train[i] for i in rng.choice(len(train), 40, replace=False)]
    analyze_paths(apo, fasc, train, "TRAIN-fasc", rows)

    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench:
        analyze_paths(apo, fasc, sorted(p for p in bench.iterdir() if p.suffix.lower() in EXTS), "BENCHMARK", rows)

    debug_methods = {}
    dbg_path = ROOT / "results" / "calibration_measurement_debug.csv"
    if dbg_path.exists():
        dbg = pd.read_csv(dbg_path)
        debug_methods = dict(zip(dbg["image_id"], dbg["calibration_method"]))

    test = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in EXTS)
    analyze_paths(apo, fasc, test, "TEST", rows, debug_methods=debug_methods)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    summary = summarize(df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print(f"\nwrote {OUT}")
    print("\nread: coherence near 1.0 means fragments are mutually aligned; low p10/min families need visual audit.")


if __name__ == "__main__":
    main()
