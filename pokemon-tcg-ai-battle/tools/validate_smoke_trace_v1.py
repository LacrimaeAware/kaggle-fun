"""Section 2: trace validation for the repaired smoke harness. Proves the per-game changed-decision logging is
reliable BEFORE any future Selector V3 / N500 smoke. Runs on the micro-smoke trace + summary. Read-only.

  PYTHONIOENCODING=utf-8 python tools/validate_smoke_trace_v1.py
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "generated" / "starmie_smoke_trace_repair_v1"
NORMALIZED_MODES = {"off", "top1_gate", "c3_family_limited", "selector_v3"}
TERMINAL_FAMS = {"ATTACK", "END", "RETREAT"}


def _is_blocked(r):
    return bool(r.get("blocked_terminal"))


def _is_override(r):
    return (not _is_blocked(r)) and r.get("selector_raw") != r.get("baseline_raw")


def main() -> int:
    rows = [json.loads(l) for l in open(DIR / "micro_smoke_trace.jsonl", encoding="utf-8")]
    summary = json.load(open(DIR / "micro_smoke_summary.json", encoding="utf-8"))
    checks = {}

    # 1. game_id unique per game: each game_id maps to ONE consistent (mode, matchup, game_result)
    by_gid = collections.defaultdict(list)
    for r in rows:
        by_gid[r.get("game_id")].append(r)
    gid_consistent = all(len({r["mode"] for r in rs}) == 1 and len({r["matchup"] for r in rs}) == 1
                         and len({r.get("game_result") for r in rs}) == 1 for rs in by_gid.values())
    checks["game_id_unique_consistent"] = {"pass": gid_consistent, "distinct_game_ids": len(by_gid),
                                           "note": "each game_id -> one (mode,matchup,result)"}

    # 2. every changed/blocked decision has a non-null game_result
    no_result = [r for r in rows if r.get("game_result") in (None, "")]
    checks["every_decision_has_outcome"] = {"pass": len(no_result) == 0, "missing_outcome": len(no_result)}

    # 3. blocked proposals are logged + count matches summary metrics
    blk_rows = sum(1 for r in rows if _is_blocked(r))
    blk_metric = sum(v.get("blocked_terminal", 0) for k, v in (summary.get("metrics") or {}).items())
    checks["blocked_proposals_logged"] = {"pass": blk_rows == blk_metric and blk_rows > 0,
                                          "blocked_rows": blk_rows, "summary_blocked_counter": blk_metric}

    # 4. first_changed appears exactly once per game that has an override
    fc_ok = True
    fc_detail = []
    for gid, rs in by_gid.items():
        n_fc = sum(1 for r in rs if r.get("first_changed"))
        n_ov = sum(1 for r in rs if _is_override(r))
        ok = (n_fc == 1 and n_ov >= 1) or (n_fc == 0 and n_ov == 0)
        if not ok:
            fc_ok = False
            fc_detail.append({"game_id": gid, "first_changed": n_fc, "overrides": n_ov})
    checks["first_changed_once_per_changed_game"] = {"pass": fc_ok, "violations": fc_detail[:5]}

    # 5. mode labels normalized
    modes_seen = {r["mode"] for r in rows}
    checks["mode_labels_normalized"] = {"pass": modes_seen <= NORMALIZED_MODES, "modes_seen": sorted(modes_seen)}

    # 6. terminal-flag correctness: c3 OVERRIDES never into a terminal family; blocked rows flagged blocked
    bad_term_overrides = [r for r in rows if r["mode"] == "c3_family_limited" and _is_override(r)
                          and r.get("selector_family") in TERMINAL_FAMS]
    blocked_flag_ok = all(r.get("terminal_override_blocked") for r in rows if _is_blocked(r))
    checks["terminal_flags_correct"] = {"pass": len(bad_term_overrides) == 0 and blocked_flag_ok,
                                        "c3_terminal_overrides": len(bad_term_overrides)}

    # 7. override counter matches override rows
    ov_rows = sum(1 for r in rows if _is_override(r))
    ov_metric = sum(v.get("overrides", 0) for k, v in (summary.get("metrics") or {}).items())
    checks["override_count_matches"] = {"pass": ov_rows == ov_metric, "override_rows": ov_rows, "summary_overrides": ov_metric}

    # 8. no forbidden metadata leak in the live trace (it is a runtime log; must carry no pilot/outcome-of-replay)
    FORBIDDEN = ("pilot_name", "replay", "future_same_turn")
    leaks = sum(1 for r in rows if any(b in json.dumps(r).lower() for b in FORBIDDEN))
    checks["no_metadata_leak"] = {"pass": leaks == 0, "leaks": leaks}

    overall = all(c["pass"] for c in checks.values())
    report = {"trace_rows": len(rows), "games_with_records": len(by_gid), "ALL_PASS": overall, "checks": checks}
    (DIR / "trace_validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"trace rows {len(rows)} | games {len(by_gid)} | ALL_PASS={overall}")
    for name, c in checks.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: {{" +
              ", ".join(f'{k}={v}' for k, v in c.items() if k != 'pass' and not isinstance(v, list)) + "}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
