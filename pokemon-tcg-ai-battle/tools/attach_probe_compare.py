"""Compare the baseline atlas table (attach off) vs the candidate (ATTACH_MEGA_NOT_ENGINE_V1 on) on the Yushin
#1-pilot cohort: trigger count, ATTACH agreement before/after, attach target distribution before/after, overall
agreement. Read-only. Writes data/starmie_audit/v2_behavior_atlas/attach_probe_audit.json.

  python tools/attach_probe_compare.py
"""
from __future__ import annotations
import collections, json
from pathlib import Path

ATLAS = Path(__file__).resolve().parent.parent / "data" / "starmie_audit" / "v2_behavior_atlas"
BASE = ATLAS / "newtop1_decisions.jsonl"
CAND = ATLAS / "newtop1_attachmega.jsonl"


def _load(p):
    return {json.loads(l)["decision_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def _agree(rows, fam=None):
    rs = [r for r in rows if fam is None or r["family"] == fam]
    n = sum(1 for r in rs if "agree" in r)
    a = sum(1 for r in rs if r.get("agree"))
    return round(100 * a / n, 1) if n else None, n


def main():
    base, cand = _load(BASE), _load(CAND)
    ids = [i for i in base if i in cand]
    b = [base[i] for i in ids]; c = [cand[i] for i in ids]
    # trigger count: decisions where the agent's pick changed
    trig = [i for i in ids if base[i].get("agent_choice") != cand[i].get("agent_choice")]
    # attach target distribution (agent) before/after
    def att_tgt(rows):
        ah = [r for r in rows if r["family"] == "ATTACH"]
        return dict(collections.Counter(r.get("agent_target_role") for r in ah))
    res = {
        "n_decisions": len(ids), "trigger_count": len(trig),
        "overall_agreement": {"baseline": _agree(b)[0], "attach_on": _agree(c)[0], "n": _agree(b)[1]},
        "attach_agreement": {"baseline": _agree(b, "ATTACH")[0], "attach_on": _agree(c, "ATTACH")[0], "n": _agree(b, "ATTACH")[1]},
        "agent_attach_target_dist_baseline": att_tgt(b),
        "agent_attach_target_dist_attach_on": att_tgt(c),
        "trigger_examples": trig[:8],
        "note": "baseline and candidate tables both built with R15 default-on; only ATTACH_MEGA differs.",
    }
    (ATLAS / "attach_probe_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "trigger_examples"}, indent=2))
    print(f"wrote {ATLAS/'attach_probe_audit.json'}")


if __name__ == "__main__":
    main()
