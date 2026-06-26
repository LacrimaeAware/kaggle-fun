"""Validate the CABT->Feature-V2 adapter (learned_selector_bridge) against Model A's exported ground-truth
payloads on the SAME parity decisions, then run end-to-end (my payload -> official packer -> portable runtime)
and compare the selected action to Model A's expected selector output.

Read-only. Resolves each parity decision's raw CABT observation from replays.

  PYTHONIOENCODING=utf-8 python tools/validate_selector_bridge_v1.py
"""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "agent" / "vendor" / "portable_selector_v1"))
import learned_selector_bridge as BR  # noqa: E402
import starmie_feature_v2_packer as PK  # noqa: E402
import starmie_selector_runtime as RT  # noqa: E402

MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v1/export")
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
_EPC: dict = {}

STATE_KEYS = ("our_hand_size", "our_deck_count", "our_prize_count", "our_bench_count", "our_attack_ready_count",
              "opp_hand_size", "opp_deck_count", "opp_prize_count", "opp_attack_ready_count", "option_count")
ENT_KEYS = ("card_id", "hp", "damage", "attached_energy_count")
OPT_KEYS = ("type_id", "attack_id", "ability_id", "source_card_id", "target_card_id", "context_card_id")


def _raw_only(action):
    """Keep only the pipeline-invariant raw-index fields of a baseline/search action reference."""
    if isinstance(action, dict):
        out = {}
        for k in ("raw_option_index", "raw_option_indexes"):
            if action.get(k) is not None:
                out[k] = action[k]
        return out or None
    return action


def _obs(e, s, seat):
    if e not in _EPC:
        if len(_EPC) > 12:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][s][seat].get("observation")
    except Exception:
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rt = RT.StarmieSelectorRuntime.from_dir(str(ROOT / "agent" / "vendor" / "portable_selector_v1"))
    exp = {json.loads(l)["decision_id"]: json.loads(l)
           for l in open(MA / "packer_parity_expected_packed.jsonl", encoding="utf-8")}

    n = resolved = sel_match = top3_match = 0
    state_mm = collections.Counter()
    state_n = collections.Counter()
    tac_mm = collections.Counter()
    tac_n = collections.Counter()
    ent_present = ent_match = 0
    ent_field_mm = collections.Counter()
    opt_mm = collections.Counter()
    opt_n = collections.Counter()
    sel_examples = []

    for line in open(MA / "packer_parity_raw_inputs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        did = r["decision_id"]
        try:
            ep, step, seat = (int(x) for x in did.split(":"))
        except Exception:
            continue
        ob = _obs(ep, step, seat)
        if not ob:
            continue
        resolved += 1
        their_ro = r["raw_observation"]
        # Map the baseline/search by RAW option index only: semantic keys are format-specific to Model A's
        # packer and would not match this adapter's keys. Raw indices are pipeline-invariant.
        baseline = _raw_only(their_ro.get("baseline_action"))
        search = _raw_only(their_ro.get("search_action"))
        payload = BR.cabt_to_payload(ob, baseline_action=baseline, search_action=search)

        # --- tactical_state_features diff (selector-critical) ---
        mine_tac = payload.get("tactical_state_features") or {}
        theirs_tac = their_ro.get("tactical_state_features") or {}
        if not theirs_tac:
            for o in (their_ro.get("raw_legal_options") or []):
                if isinstance(o.get("tactical_state_features"), dict):
                    theirs_tac = o["tactical_state_features"]
                    break
        for k, tv in theirs_tac.items():
            tac_n[k] += 1
            if mine_tac.get(k) != tv:
                tac_mm[k] += 1

        # --- state_features diff ---
        mine_sf = payload["state_features"]
        theirs_sf = their_ro.get("state_features") or {}
        for k in STATE_KEYS:
            if k in theirs_sf:
                state_n[k] += 1
                if mine_sf.get(k) != theirs_sf.get(k):
                    state_mm[k] += 1

        # --- board_entities diff (keyed by role/zone/slot) ---
        mine_be = {(e["player_role"], e["zone"], e["slot_index"]): e for e in payload["board_entities"]}
        theirs_be = {(e.get("player_role"), e.get("zone"), e.get("slot_index")): e
                     for e in (their_ro.get("board_entities") or [])}
        for key, te in theirs_be.items():
            ent_present += 1
            me = mine_be.get(key)
            if me is None:
                ent_field_mm["MISSING_ENTITY"] += 1
                continue
            ok = True
            for f in ENT_KEYS:
                if me.get(f) != te.get(f):
                    ent_field_mm[f] += 1
                    ok = False
            if ok:
                ent_match += 1

        # --- pack + run end-to-end ---
        packed = PK.pack_cabt_observation(payload, payload["raw_legal_options"])
        out = rt.rank_and_select(packed, packed["packed_options"], packed.get("baseline_action"),
                                 packed.get("search_action"), packed.get("top_k", 5))
        e_out = r.get("expected_selector_output") or {}
        n += 1
        # grouping-invariant action identity: the selected RAW option index
        my_raw = out.get("selected_raw_option_index")
        their_raw = e_out.get("selected_raw_option_index")
        if my_raw == their_raw:
            sel_match += 1
        else:
            if len(sel_examples) < 14:
                sel_examples.append({"decision_id": did, "my_raw": my_raw, "their_raw": their_raw,
                                     "my_packed": out.get("selected_packed_option_index"),
                                     "their_packed": e_out.get("selected_packed_option_index"),
                                     "my_key": out.get("selected_semantic_key"),
                                     "their_key": e_out.get("selected_semantic_key")})
        # top-3 packed
        their_top3 = exp.get(did, {}).get("expected_top3_packed_option_indexes") or []
        my_ranked = out.get("ranked_actions") or []
        my_top3 = [a.get("packed_option_index") for a in my_ranked[:3]]
        if their_top3 and my_top3 and their_top3[0] == my_top3[0]:
            top3_match += 1

        # --- per-option exact-ID parity vs expected packed ---
        epo = {po.get("raw_option_index"): po for po in (exp.get(did, {}).get("expected_packed_options") or [])}
        for po in packed["packed_options"]:
            ri = po.get("raw_option_index")
            tp = epo.get(ri)
            if tp is None:
                continue
            tf = tp.get("features") or {}
            mf = po.get("features") or {}
            for k in OPT_KEYS:
                opt_n[k] += 1
                if (po.get(k) if k in po else mf.get(k)) != (tp.get(k) if k in tp else tf.get(k)):
                    opt_mm[k] += 1

    def pct(a, b):
        return round(100 * a / max(1, b), 1)

    report = {
        "decisions_resolved": resolved, "scored": n,
        "selected_match_pct": pct(sel_match, n), "top1_of_top3_match_pct": pct(top3_match, n),
        "state_feature_match_pct": {k: pct(state_n[k] - state_mm[k], state_n[k]) for k in STATE_KEYS},
        "tactical_feature_match_pct": {k: pct(tac_n[k] - tac_mm[k], tac_n[k]) for k in sorted(tac_n)},
        "board_entity_exact_row_match_pct": pct(ent_match, ent_present),
        "board_entity_field_mismatches": dict(ent_field_mm),
        "option_feature_match_pct": {k: pct(opt_n[k] - opt_mm[k], opt_n[k]) for k in OPT_KEYS},
        "selection_mismatch_examples": sel_examples,
    }
    (OUT / "bridge_adapter_validation_v1.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "selection_mismatch_examples"}, indent=2, default=str))
    print("\nselection mismatches (first few):")
    for e in sel_examples[:8]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
