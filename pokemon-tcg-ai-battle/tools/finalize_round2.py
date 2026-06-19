"""Branch A -- post-process the round-2 risk labels per the adversarial verification findings.

Adds (no relabeling; all derivable from the stored values):
  group_id                          : source replay file = GAME, so B can do a group-held-out split
                                      (c1 positives cluster by game; consecutive steps are near-duplicates).
  selected_option_high_regret_flag  : convenience -- is agent_search's pick high_regret this label.
  selected_option_unacceptable_flag : convenience -- ... and is it unacceptable.
  c1_candidate                      : this state was SAMPLED as a search-selected-high-regret candidate
                                      (mined or the catastrophic seed). B's #1-priority positive pool.
  c1_reproduced_this_label          : did the c1 property actually realize on THIS (noisy) label.
                                      c1_candidate & !c1_reproduced_this_label == a "flipped/dead" c1 row
                                      (the ~53% non-reproducibility, made explicit so B can down-weight).
  eval_only                         : the 2 B-failure SEED states -- hold out, do not train.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "data" / "manifests" / "teacher_v2_residual_risk_labels_round2.jsonl"
C1_SEED = "80251230.json:12"  # B's catastrophic-miss seed (c1 failure mode)

recs = [json.loads(l) for l in open(ART, encoding="utf-8")]
for r in recs:
    tags = r.get("criterion_tags", [])
    so = next((o for o in r["options"] if o["index"] == r["search_selected_option"]), None)
    r["group_id"] = (r.get("source") or {}).get("file")
    r["selected_option_high_regret_flag"] = so["high_regret_flag"] if so else None
    r["selected_option_unacceptable_flag"] = so["unacceptable_flag"] if so else None
    r["c1_candidate"] = ("MINED_c1" in tags) or (r["decision_id"] == C1_SEED)
    r["c1_reproduced_this_label"] = bool(so and so["high_regret_flag"] == 1)
    r["eval_only"] = "SEED" in tags

with open(ART, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

# report
from collections import Counter
games = Counter(r["group_id"] for r in recs)
c1_cand = [r for r in recs if r["c1_candidate"]]
c1_repro = [r for r in c1_cand if r["c1_reproduced_this_label"]]
c1_games = Counter(r["group_id"] for r in c1_repro)
print(f"records: {len(recs)} | games: {len(games)}")
print(f"c1_candidate: {len(c1_cand)} | reproduced this label: {len(c1_repro)} | dead/flipped: {len(c1_cand)-len(c1_repro)}")
print(f"reproduced-c1 by game ({len(c1_games)} games): {dict(c1_games)}")
print(f"eval_only (seeds): {sum(1 for r in recs if r['eval_only'])}")
print(f"sel high_regret states: {sum(1 for r in recs if r['selected_option_high_regret_flag']==1)} | "
      f"sel high_regret & not-unacc: {sum(1 for r in recs if r['selected_option_high_regret_flag']==1 and r['selected_option_unacceptable_flag']==0)}")
print(f"games with >=1 c2 (false-pos calib): "
      f"{len({r['group_id'] for r in recs if 'c2_safe_search_false_positive' in r['criterion_tags']})}")
