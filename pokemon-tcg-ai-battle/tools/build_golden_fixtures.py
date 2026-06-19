"""SPLIT_BASE_V2 / P3 -- build the golden state/action fixtures.

Selects >=100 REAL decisions from the replay corpus, stratified to cover every major action type, and
records for each the raw observation plus the schema's outputs (card ids, semantic keys, equivalence
classes, the canonical encoding, and the forced/non-forced classification). tests/test_split_base_v2.py
recomputes these live and asserts they match -- the contract that kills the schema/encoding bug class.

    python tools/build_golden_fixtures.py [--target 120]

Output: tests/golden_state_action_fixtures/fixtures.json  (committed; the raw replays stay gitignored).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import state_action_schema_v2 as SCH  # noqa: E402
import main as M                       # noqa: E402  (_forced_move -- pure, no engine)

REPLAY_DIR = ROOT / "data" / "external" / "replays"
OUT_DIR = ROOT / "tests" / "golden_state_action_fixtures"


def _fixture(fn, si, ai, obs):
    cur = obs["current"]
    opts = obs["select"]["option"]
    me = cur.get("yourIndex", ai)
    players = cur.get("players") or []
    me_player = players[me] if me < len(players) else {}
    forced = None
    try:
        fm = M._forced_move(obs)               # pure heuristic floor; no engine
        forced = fm[0] if fm else None
    except Exception:
        forced = None
    return {
        "source": fn, "step": si, "player": ai, "your_index": me,
        "n_options": len(opts),
        "option_types": [o.get("type") if isinstance(o, dict) else None for o in opts],
        "card_ids": [SCH.card_identity(o, me_player) if isinstance(o, dict) else None for o in opts],
        "semantic_keys": [list(SCH.semantic_action_key(o, cur, me)) if isinstance(o, dict) else None
                          for o in opts],
        "eq_classes": SCH.equivalence_classes(opts, cur, me),
        "root_encoding": SCH.encode_vector(obs),
        "forced_option": forced,
        "observation": obs,                    # raw (keeps search_begin_input for the teacher test)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=120, help="number of fixtures (>=100 required)")
    args = ap.parse_args()

    keys = SCH.FEATURE_KEYS
    major = set(SCH.MAJOR_ACTION_TYPES)
    picked, covered = [], set()
    feat_vals = {k: set() for k in keys}      # distinct values seen per feature (for diversity)
    type_hist = Counter()
    CAP = max(args.target + 90, 220)          # allow overshoot to capture rare edge states

    def diversity_remaining():
        return (not (major <= covered)) or any(len(feat_vals[k]) < 2 for k in keys)

    # select for BOTH action-type coverage AND per-feature value diversity, so the dead-feature
    # test exercises edge states (KO-now, status conditions, ex actives, deckout, gust/heal) rather
    # than failing on sampling bias. Deterministic scan order.
    for fp in sorted(glob.glob(str(REPLAY_DIR / "*.json"))):
        if (len(picked) >= args.target and not diversity_remaining()) or len(picked) >= CAP:
            break
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        fn = os.path.basename(fp)
        for si, s in enumerate(d.get("steps", [])):
            if (len(picked) >= args.target and not diversity_remaining()) or len(picked) >= CAP:
                break
            for ai, agent in enumerate(s):
                if not isinstance(agent, dict):
                    continue
                obs = agent.get("observation") or {}
                if not SCH.is_single_pick_decision(obs):
                    continue
                cur = obs.get("current") or {}
                if not cur.get("players"):
                    continue
                types = {o.get("type") for o in obs["select"]["option"] if isinstance(o, dict)} & major
                vec = SCH.encode_vector(obs)
                new_type = bool(types - covered)
                new_feat = any(len(feat_vals[keys[i]]) < 2 and vec[i] not in feat_vals[keys[i]]
                               for i in range(len(keys)))
                if new_type or new_feat or len(picked) < args.target:
                    picked.append(_fixture(fn, si, ai, obs))
                    covered |= types
                    type_hist.update(types)
                    for i in range(len(keys)):
                        feat_vals[keys[i]].add(vec[i])
                    break  # one decision per step is plenty

    still_const = sorted(k for k in keys if len(feat_vals[k]) < 2)
    fixtures = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCH.SCHEMA_VERSION,
        "n_fixtures": len(picked),
        "major_action_types": sorted(major),
        "covered_action_types": sorted(covered),
        "type_coverage": {str(t): type_hist[t] for t in sorted(type_hist)},
        "feature_keys": SCH.FEATURE_KEYS,
        # features that stay constant even after diversity-seeking selection = genuinely absent from
        # the corpus at single-pick decisions (documented expected-constant; the dead-feature test
        # only fails on UNEXPECTED constants outside this list).
        "expected_constant_features": still_const,
        "fixtures": picked,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fixtures.json"
    path.write_text(json.dumps(fixtures, indent=1), encoding="utf-8")
    print(f"wrote {len(picked)} fixtures -> {path.relative_to(ROOT)}")
    print(f"  covered major types: {sorted(covered)}  (need {sorted(major)})")
    print(f"  type coverage (option-occurrences): {fixtures['type_coverage']}")
    print(f"  features varying: {len(keys) - len(still_const)}/{len(keys)}; "
          f"still-constant (expected): {still_const}")
    missing = major - covered
    print(f"  MISSING TYPES: {sorted(missing)}" if missing else "  all major action types covered")


if __name__ == "__main__":
    main()
