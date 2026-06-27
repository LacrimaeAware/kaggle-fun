"""Tests for the local meta analyzer (the trust anchor). Pure stats + file IO, no engine.

Verifies: Wilson/delta-CI/MDE sanity, correct pooling of wins/losses across stages (NOT averaging percentages),
primary-vs-sentinel separation, and -- most important -- that an interim look which crosses p<0.05 but does not
hold at the full sample triggers the early-stopping warning (the exact trap that turned a +15pp n=20 smoke into
+2.6pp at N500).

Run:  PYTHONIOENCODING=utf-8 python tests/test_local_meta_analyze_v1.py
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import local_meta_analyze_v1 as A  # noqa: E402


def _cell(win, loss):
    return {"win": win, "loss": loss, "draw": 0, "err": 0, "win_pct": round(100 * win / max(1, win + loss), 1)}


def _summary(s0, s3):
    # s0/s3: dict opp -> (win, loss)
    return {"modes": {"S0": "off", "S3": "treat"},
            "results": {"S0": {o: _cell(*wl) for o, wl in s0.items()},
                        "S3": {o: _cell(*wl) for o, wl in s3.items()}}}


def test_stat_helpers():
    lo, hi = A.wilson(50, 100)
    assert 39 < lo < 41 and 59 < hi < 61, (lo, hi)
    assert A.wilson(0, 0) == [None, None]
    clo, chi = A.dci(60, 100, 40, 100)   # +20pp diff
    assert clo < 20 < chi and clo > 0, (clo, chi)   # CI excludes 0 for a clear effect
    assert A.mde_pp(1000, 0.475) and 5 < A.mde_pp(1000, 0.475) < 8
    print("PASS stat-helpers")


def test_pooling_and_report_cell():
    # pooled must SUM wins/losses, not average percentages
    pooled = {"S0": {"deployed": {"win": 10, "loss": 10}, "mirror": {"win": 12, "loss": 8}},
              "S3": {"deployed": {"win": 14, "loss": 6}, "mirror": {"win": 13, "loss": 7}}}
    c = A._report_cell(pooled, "S0", "S3", ["deployed", "mirror"])
    assert c["baseline"] == "22-18" and c["treatment"] == "27-13", c
    assert c["n_per_arm"] == 40 and c["delta_pp"] == round(100 * 27 / 40 - 100 * 22 / 40, 1), c
    print("PASS pooling + report-cell")


def _write_run(d, stageA, stageB):
    for s, (s0, s3) in (("A", stageA), ("B", stageB)):
        (d / f"stage_{s}_summary.json").write_text(json.dumps(_summary(s0, s3)), encoding="utf-8")
        (d / f"stage_{s}_changed_decisions.jsonl").write_text("", encoding="utf-8")
        (d / f"stage_{s}_game_summary.jsonl").write_text("", encoding="utf-8")


def test_early_stopping_warning_fires():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Stage A: treatment strongly ahead on primary (interim "significant"); Stage B reverses -> pooled ~even
        _write_run(d,
                   stageA=({"deployed": (3, 17), "mirror": (4, 16)}, {"deployed": (16, 4), "mirror": (15, 5)}),
                   stageB=({"deployed": (30, 10), "mirror": (30, 10)}, {"deployed": (17, 23), "mirror": (18, 22)}))
        subprocess.run([sys.executable, str(ROOT / "tools" / "local_meta_analyze_v1.py"), "--dir", str(d),
                        "--baseline", "off", "--treatment", "treat", "--primary", "deployed,mirror",
                        "--sentinels", "", "--neg", ""], check=True, capture_output=True, text=True)
        rep = json.loads((d / "analysis_report.json").read_text(encoding="utf-8"))
        traj = rep["early_stopping_trajectory"]
        assert len(traj) == 2, traj
        assert traj[0]["fisher_p"] < 0.05, ("stage-A interim should look significant", traj[0])
        assert rep["PRIMARY_combined_deployed_mirror"]["fisher_p"] > 0.05, "pooled should NOT be significant"
        assert "interim look crossed" in rep["early_stopping_warning"], rep["early_stopping_warning"]
        print("PASS early-stopping-warning fires (interim sig, full not)")


def test_primary_not_inflated_by_field():
    # field landslide must NOT enter the primary cell
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_run(d,
                   stageA=({"deployed": (10, 10), "mirror": (10, 10), "first": (1, 19)},
                           {"deployed": (10, 10), "mirror": (10, 10), "first": (20, 0)}),
                   stageB=({"deployed": (10, 10), "mirror": (10, 10), "first": (1, 19)},
                           {"deployed": (10, 10), "mirror": (10, 10), "first": (20, 0)}))
        subprocess.run([sys.executable, str(ROOT / "tools" / "local_meta_analyze_v1.py"), "--dir", str(d),
                        "--baseline", "off", "--treatment", "treat", "--primary", "deployed,mirror",
                        "--sentinels", "", "--neg", "first"], check=True, capture_output=True, text=True)
        rep = json.loads((d / "analysis_report.json").read_text(encoding="utf-8"))
        assert rep["PRIMARY_combined_deployed_mirror"]["delta_pp"] == 0.0, "primary must ignore the field landslide"
        assert rep["negative_controls"]["first"]["delta_pp"] > 50, "field win belongs in negative controls"
        print("PASS primary-isolated-from-field")


def main():
    fns = [test_stat_helpers, test_pooling_and_report_cell, test_early_stopping_warning_fires,
           test_primary_not_inflated_by_field]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print("ALL LOCAL-META-ANALYZE TESTS PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
