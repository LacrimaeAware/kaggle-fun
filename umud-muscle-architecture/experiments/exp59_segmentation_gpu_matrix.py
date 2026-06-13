"""Write the next GPU segmentation run matrix.

The actual training needs torch/CUDA/RunPod/Kaggle. This script is intentionally
lightweight: it records the ordered run plan and exact environment variables so
we can execute comparable submissions without losing track of what changed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp59_segmentation_gpu_matrix.csv"


RUNS = [
    {
        "run_id": "seg59_01_repro_384_unet",
        "priority": 1,
        "submit": "only if it unexpectedly differs from the known baseline",
        "UMUD_EPOCHS": 12,
        "UMUD_IMG_SIZE": 384,
        "UMUD_MODEL_ARCH": "unet",
        "UMUD_MODEL_ENCODER": "resnet34",
        "UMUD_LOSS_MODE": "dice_bce",
        "UMUD_AUG_LEVEL": "light",
        "UMUD_CLAHE": 0,
        "UMUD_FASC_POS_WEIGHT": 0,
        "UMUD_WEIGHTS_TAG": "seg59_01",
        "reason": "reproduction/control: verifies the new configurable code path matches the old setup",
    },
    {
        "run_id": "seg59_02_highres_512_unet",
        "priority": 2,
        "submit": "yes if sanity checks pass",
        "UMUD_EPOCHS": 18,
        "UMUD_IMG_SIZE": 512,
        "UMUD_MODEL_ARCH": "unet",
        "UMUD_MODEL_ENCODER": "resnet34",
        "UMUD_LOSS_MODE": "dice_bce",
        "UMUD_AUG_LEVEL": "light",
        "UMUD_CLAHE": 0,
        "UMUD_FASC_POS_WEIGHT": 0,
        "UMUD_WEIGHTS_TAG": "seg59_02",
        "reason": "most conservative mask-quality test: more pixels for thin fragments, no architecture/loss confound",
    },
    {
        "run_id": "seg59_03_highres_512_unetplusplus",
        "priority": 3,
        "submit": "yes if #2 is not clearly worse and masks look sane",
        "UMUD_EPOCHS": 18,
        "UMUD_IMG_SIZE": 512,
        "UMUD_MODEL_ARCH": "unetplusplus",
        "UMUD_MODEL_ENCODER": "resnet34",
        "UMUD_LOSS_MODE": "dice_bce",
        "UMUD_AUG_LEVEL": "light",
        "UMUD_CLAHE": 0,
        "UMUD_FASC_POS_WEIGHT": 0,
        "UMUD_WEIGHTS_TAG": "seg59_03",
        "reason": "tests stronger decoder/skip fusion while keeping encoder/loss fixed",
    },
    {
        "run_id": "seg59_04_highres_focal",
        "priority": 4,
        "submit": "only if fragment counts/support improve without PA drift",
        "UMUD_EPOCHS": 18,
        "UMUD_IMG_SIZE": 512,
        "UMUD_MODEL_ARCH": "unetplusplus",
        "UMUD_MODEL_ENCODER": "resnet34",
        "UMUD_LOSS_MODE": "dice_focal",
        "UMUD_AUG_LEVEL": "light",
        "UMUD_CLAHE": 0,
        "UMUD_FASC_POS_WEIGHT": 0,
        "UMUD_WEIGHTS_TAG": "seg59_04",
        "reason": "sparse-structure loss test; safer than crude positive class weighting",
    },
    {
        "run_id": "seg59_05_train_clahe",
        "priority": 5,
        "submit": "exploratory; submit only if QC looks substantially better",
        "UMUD_EPOCHS": 18,
        "UMUD_IMG_SIZE": 512,
        "UMUD_MODEL_ARCH": "unet",
        "UMUD_MODEL_ENCODER": "resnet34",
        "UMUD_LOSS_MODE": "dice_bce",
        "UMUD_AUG_LEVEL": "light",
        "UMUD_CLAHE": 1,
        "UMUD_FASC_POS_WEIGHT": 0,
        "UMUD_WEIGHTS_TAG": "seg59_05",
        "reason": "tests the earlier contrast signal correctly: train and inference both normalized",
    },
]


def command(row: dict) -> str:
    env = " ".join(
        f"{key}={row[key]}"
        for key in [
            "UMUD_EPOCHS",
            "UMUD_IMG_SIZE",
            "UMUD_MODEL_ARCH",
            "UMUD_MODEL_ENCODER",
            "UMUD_LOSS_MODE",
            "UMUD_AUG_LEVEL",
            "UMUD_CLAHE",
            "UMUD_FASC_POS_WEIGHT",
            "UMUD_WEIGHTS_TAG",
        ]
    )
    return f"{env} python segment_then_measure.py"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(RUNS)
    df["command"] = df.apply(lambda r: command(r.to_dict()), axis=1)
    df.to_csv(OUT, index=False)
    print(df[["priority", "run_id", "submit", "reason", "command"]].to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
