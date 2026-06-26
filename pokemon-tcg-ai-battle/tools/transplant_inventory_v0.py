"""Section 1: inventory the trace/corpus/smoke files for Model A's replay-transplant value prior. Read-only.

  PYTHONIOENCODING=utf-8 python tools/transplant_inventory_v0.py
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_transplant_support_v0"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")

FILES = {
    "corpus_v1": "data/starmie_corpus/starmie_specialist_corpus_v1.jsonl",
    "corpus_v2": "data/starmie_corpus/starmie_specialist_corpus_v2.jsonl",
    "zero_shot_newtop1": "data/starmie_corpus/starmie_v2_zero_shot_newtop1.jsonl",
    "yushin_trace": "data/generated/starmie_bridge_trace_v0/yushin_trace.jsonl",
    "keidroid_trace": "data/generated/starmie_bridge_trace_v0/keidroid_trace.jsonl",
    "old_exact_trace": "data/generated/starmie_bridge_trace_v0/old_exact_trace_sample.jsonl",
    "model_a_bridge_input": "data/generated/starmie_bridge_trace_v0/model_a_bridge_input.jsonl",
    "v1_smoke_changed_classes": "data/generated/starmie_selector_live_smoke_v1/changed_decision_classes.jsonl",
    "v2_smoke_changed_decisions": "data/generated/starmie_selector_v2_smoke/changed_decisions.jsonl",
    "v1_smoke_summary": "data/generated/starmie_selector_live_smoke_v1/live_smoke_report.json",
    "v2_smoke_summary": "data/generated/starmie_selector_v2_smoke/live_smoke_summary.json",
}
RUNTIME_KEYS = {"runtime", "runtime_tactical", "tactical_state", "option_index_to_semantic_key"}
EVAL_KEYS = {"eval_meta", "pilot_action", "pilot_name", "outcome_won", "future_same_turn_sequence"}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _profile_jsonl(p, cap=200000):
    n = 0
    eps = set()
    decs = set()
    fams = collections.Counter()
    has_runtime = has_eval = False
    first_keys = None
    with open(p, encoding="utf-8") as f:
        for line in f:
            if n >= cap:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            if first_keys is None:
                first_keys = sorted(r.keys())
                has_runtime = bool(RUNTIME_KEYS & set(r.keys())) or "runtime" in r
                has_eval = bool(EVAL_KEYS & set(r.keys())) or "eval_meta" in r
            did = r.get("decision_id") or r.get("id")
            if did is not None:
                decs.add(str(did))
            ep = r.get("episode_id") or r.get("episode") or (r.get("eval_meta") or {}).get("episode_id")
            if ep is not None:
                eps.add(str(ep))
            fam = (r.get("family") or r.get("baseline_family")
                   or (r.get("eval_meta") or {}).get("pilot_action_family")
                   or (r.get("runtime") or {}).get("current_agent_action_family"))
            if fam:
                fams[fam] += 1
    return {"rows": n, "capped": n >= cap, "decisions": len(decs), "episodes": len(eps),
            "family_counts": dict(fams.most_common()), "has_runtime_fields": has_runtime,
            "has_eval_fields": has_eval, "runtime_eval_separation": has_runtime and has_eval,
            "first_row_keys": first_keys}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = {}
    for name, rel in FILES.items():
        p = ROOT / rel
        rec = {"path": str(rel), "exists": p.exists()}
        if p.exists():
            rec["size_bytes"] = p.stat().st_size
            rec["sha256_16"] = _sha(p)
            if rel.endswith(".jsonl"):
                rec.update(_profile_jsonl(p))
        inv[name] = rec
    # replays dir
    n_replays = len([f for f in os.listdir(REPLAYS) if f.endswith(".json")]) if REPLAYS.exists() else 0
    inv["replays_dir"] = {"path": str(REPLAYS), "exists": REPLAYS.exists(), "n_replay_files": n_replays}
    (OUT / "trace_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print("trace_inventory.json:")
    for name, rec in inv.items():
        if rec.get("exists"):
            extra = (f" rows={rec.get('rows')} dec={rec.get('decisions')} ep={rec.get('episodes')} "
                     f"sep={rec.get('runtime_eval_separation')}" if rec.get("rows") is not None
                     else f" n_replays={rec.get('n_replay_files')}" if "n_replay_files" in rec else "")
            print(f"  {name:26s} OK{extra}")
        else:
            print(f"  {name:26s} MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
