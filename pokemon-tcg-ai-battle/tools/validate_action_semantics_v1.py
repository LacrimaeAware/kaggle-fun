"""Continuous Terrain V1 -- A5 semantic validation on the 50 most frequent acting cards.

Reports, for each top-50 acting card: name, frequency, decoded coverage tier, and the decoded effect fields,
so misdecoded/under-decoded frequent cards can get a deterministic override. Coverage = fraction of the top-50
(frequency-weighted and unweighted) whose semantics are trustworthy (decoded or override, not unknown).

    python tools/validate_action_semantics_v1.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import action_semantics_v1 as AS   # noqa: E402

MAN = ROOT / "data" / "manifests"


def main():
    path = MAN / "continuous_terrain_v1.jsonl"
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    freq = Counter()
    cov_by_card = defaultdict(Counter)
    eff_by_card = {}
    for r in recs:
        for o in r["options"]:
            sv = o["semantic_vector"]
            cid = sv["acting_card_id"]
            if cid is None or cid < 0:
                continue
            freq[cid] += 1
            cov_by_card[cid][sv["semantic_coverage"]] += 1
            if cid not in eff_by_card:
                eff_by_card[cid] = {k: v for k, v in sv.items() if k.startswith("eff_") or k in ("own_switch", "opp_gust", "atk_damage")}
    top = freq.most_common(50)
    trust = 0
    wtrust = 0
    wtot = 0
    print(f"{'card':38s} {'freq':>5s} {'cov':>9s}  effects")
    rows = []
    TRUST = {"decoded", "override", "energy", "pokemon_meta", "tool"}
    for cid, fr in top:
        cov = cov_by_card[cid].most_common(1)[0][0]
        ok = cov in TRUST
        trust += int(ok)
        wtrust += fr * int(ok)
        wtot += fr
        nm = AS.CARD_NAME.get(cid, "?")[:36]
        active_eff = {k: v for k, v in eff_by_card.get(cid, {}).items() if v}
        print(f"{nm:38s} {fr:5d} {cov:>9s}  {active_eff}")
        rows.append({"card_id": cid, "name": AS.CARD_NAME.get(cid), "freq": fr, "coverage": cov, "effects": active_eff})
    summary = {"top50_trust_unweighted": round(trust / max(1, len(top)), 3),
               "top50_trust_freq_weighted": round(wtrust / max(1, wtot), 3),
               "n_top": len(top), "overrides_active": len(AS.OVERRIDES),
               "rows": rows}
    json.dump(summary, open(MAN / "continuous_terrain_v1_semantic_validation.json", "w", encoding="utf-8"), indent=1)
    print(f"\nTOP-50 trustworthy (decoded/override): unweighted {summary['top50_trust_unweighted']} | "
          f"freq-weighted {summary['top50_trust_freq_weighted']} | overrides {len(AS.OVERRIDES)}")


if __name__ == "__main__":
    main()
