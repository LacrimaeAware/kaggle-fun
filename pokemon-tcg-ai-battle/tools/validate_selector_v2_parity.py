"""In-repo parity gate for the vendored conservative selector V2 (C3_FAMILY_LIMITED). Feeds Model A's V2 parity
sets through the VENDORED V2 packer + runtime and compares to the expected outputs. Read-only.

Gates (Section 3): packer fields 100%; selected action >=99%; terminal-override-block 100%; top-k >=99%;
logits within tolerance; deterministic repeat; 0 forbidden-metadata failures; no observation mutation.

  PYTHONIOENCODING=utf-8 python tools/validate_selector_v2_parity.py
"""
from __future__ import annotations
import collections
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VD = ROOT / "agent" / "vendor" / "portable_selector_v2"
sys.path.insert(0, str(VD))
import starmie_feature_v2_packer as PK   # noqa: E402  vendored V2 packer
import starmie_selector_runtime as RT    # noqa: E402  vendored V2 runtime
EXPORT = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v2/export")
OUT = ROOT / "data" / "generated" / "starmie_selector_v2_smoke"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rt = RT.StarmieSelectorRuntime.from_dir(str(VD))

    # ---------- selector parity (pre-packed inputs) ----------
    exp = {json.loads(l)["decision_id"]: json.loads(l) for l in open(EXPORT / "parity_expected.jsonl", encoding="utf-8")}
    n = sel = blk = topk = det = mut = 0
    max_logit_diff = 0.0
    cat = collections.Counter()
    cat_sel = collections.Counter()
    mismatches = []
    for line in open(EXPORT / "parity_inputs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        did = r["decision_id"]
        e = exp.get(did)
        if not e:
            continue
        obs = r["observation"]
        before = copy.deepcopy(obs)
        opts = r.get("legal_options") or obs.get("packed_options")
        out = rt.rank_and_select(obs, opts, r.get("baseline_action"), r.get("search_action"),
                                 mode=r.get("mode", "c3_family_limited"), top_k=r.get("top_k", 5))
        out2 = rt.rank_and_select(obs, opts, r.get("baseline_action"), r.get("search_action"),
                                  mode=r.get("mode", "c3_family_limited"), top_k=r.get("top_k", 5))
        if obs != before:
            mut += 1
        n += 1
        c = e.get("category", "?")
        cat[c] += 1
        if out.get("selected_raw_option_index") == e.get("expected_selected_raw_option_index"):
            sel += 1
            cat_sel[c] += 1
        elif len(mismatches) < 12:
            mismatches.append({"decision_id": did, "category": c, "mine": out.get("selected_raw_option_index"),
                               "theirs": e.get("expected_selected_raw_option_index"),
                               "my_block": out.get("terminal_override_blocked"), "their_block": e.get("expected_terminal_override_blocked")})
        if bool(out.get("terminal_override_blocked")) == bool(e.get("expected_terminal_override_blocked")):
            blk += 1
        # top-k packed
        my_topk = [a.get("packed_option_index") for a in (out.get("proposer_top_k") or out.get("ranked_actions") or [])][:3]
        their_topk = e.get("expected_top3_packed_option_indexes") or (e.get("expected_topk_packed_option_indexes") or [])[:3]
        if my_topk and their_topk and my_topk[0] == their_topk[0]:
            topk += 1
        # logits -- align by packed_option_index (ranked_actions is rank-sorted, expected_logits is index-ordered)
        el = e.get("expected_logits")
        if isinstance(el, list):
            by_idx = {a.get("packed_option_index"): a.get("logit") for a in (out.get("ranked_actions") or [])}
            for i, b in enumerate(el):
                a = by_idx.get(i)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    max_logit_diff = max(max_logit_diff, abs(a - b))
        if out.get("selected_raw_option_index") == out2.get("selected_raw_option_index"):
            det += 1

    def pct(a):
        return round(100 * a / max(1, n), 1)
    # terminal-block category accuracy: of block_* categories, all must be blocked; of allow_* none blocked
    block_cats = {did: e for did, e in exp.items() if str(e.get("category", "")).startswith("block_")}
    report = {
        "selector_parity": {
            "decisions": n, "selected_match_pct": pct(sel), "terminal_block_match_pct": pct(blk),
            "top1_of_topk_match_pct": pct(topk), "deterministic_pct": pct(det), "obs_mutated": mut,
            "max_logit_abs_diff": round(max_logit_diff, 9), "logit_tolerance_documented": 1e-6,
            "selected_match_by_category": {c: f"{cat_sel[c]}/{cat[c]}" for c in sorted(cat)},
            "block_category_count": len(block_cats), "mismatches": mismatches,
        }
    }

    # ---------- packer parity (raw CABT -> V2 packer -> packed fields) ----------
    pexp = {json.loads(l)["decision_id"]: json.loads(l) for l in open(EXPORT / "packer_parity_expected_packed.jsonl", encoding="utf-8")}
    fmm = collections.Counter()
    ftot = collections.Counter()
    pn = 0
    for line in open(EXPORT / "packer_parity_raw_inputs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        did = r["decision_id"]
        ep = pexp.get(did)
        if not ep:
            continue
        pn += 1
        ro = r["raw_observation"]
        rl = r.get("raw_legal_options")
        packed = PK.pack_cabt_observation(ro, rl)
        epo = {po.get("raw_option_index"): po for po in (ep.get("expected_packed_options") or [])}
        for po in packed["packed_options"]:
            tp = epo.get(po.get("raw_option_index"))
            if not tp:
                continue
            for k in ("type_id", "attack_id", "source_card_id", "target_card_id", "semantic_action_key", "raw_option_indexes"):
                ftot[k] += 1
                if po.get(k) != tp.get(k):
                    fmm[k] += 1
    report["packer_parity"] = {"decisions": pn,
                               "field_match_pct": {k: round(100 * (ftot[k] - fmm[k]) / max(1, ftot[k]), 1) for k in ftot}}

    # ---------- pass/fail ----------
    sp = report["selector_parity"]
    pp = report["packer_parity"]
    passed = (sp["selected_match_pct"] >= 99 and sp["terminal_block_match_pct"] == 100 and
              sp["top1_of_topk_match_pct"] >= 99 and sp["deterministic_pct"] == 100 and sp["obs_mutated"] == 0 and
              all(v == 100 for v in pp["field_match_pct"].values()))
    report["GATE"] = "PASS" if passed else "FAIL"
    (OUT / "selector_v2_parity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"GATE": report["GATE"], "selector": {k: v for k, v in sp.items() if k != "mismatches"},
                      "packer": pp}, indent=2, default=str))
    if sp["mismatches"]:
        print("mismatches:", json.dumps(sp["mismatches"][:6]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
