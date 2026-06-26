"""STARMIE PROPOSER BRIDGE ADAPTER -- offline DRY-RUN CLI (Section 4). Read-only; never alters gameplay.

Modes:
  A NO_PROPOSER       -- verify the adapter is disabled and cannot change the agent action; compute baseline
                         trace agreement; sanity-run the safety filters on the CURRENT agent's actions.
  B LOGITS_ONLY       -- join a Model A logits/ranks file by decision_id + semantic key; candidate-set coverage;
                         apply safety filters to hypothetical proposer top-1/top-3. (waits if no logits file)
  C RUNTIME_PROPOSER  -- if a Model A runtime artifact exists, call it on recorded obs. (waits if absent)

  python tools/dry_run_proposer_bridge_v0.py --mode A --trace yushin_trace.jsonl --safety-sample 500
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import learned_proposer_adapter as AD   # noqa: E402
import deck_policy_v3 as DP             # noqa: E402

TRACE_DIR = ROOT / "data" / "generated" / "starmie_bridge_trace_v0"
OUT = ROOT / "data" / "generated" / "starmie_bridge_adapter_v0"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
PROPOSER_EXPORT = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/generated/starmie_specialist/starmie_behavior_proposer_runtime_v1.json")
_EPC = {}


def _obs(e, s, seat):
    if e not in _EPC:
        if len(_EPC) > 16:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][s][seat].get("observation")
    except Exception:
        return None


def mode_a(trace, safety_sample):
    rows = [json.loads(l) for l in open(TRACE_DIR / trace, encoding="utf-8")]
    # 1. adapter disabled: rank_actions returns no ranked actions on every row -> cannot change action
    handle = AD.load_proposer(None)
    disabled_ok = True
    for r in rows[:200]:
        em = r["eval_meta"]
        ob = _obs(em["episode_id"], em["step"], em["seat"])
        if ob is None:
            continue
        res = AD.rank_actions(handle, ob)
        if res["status"] not in ("MISSING", "DISABLED") or res["ranked_actions"]:
            disabled_ok = False
            break
    # 2. baseline agreement (from traces)
    agree = sum(1 for r in rows if r["eval_meta"]["agreement"])
    # 3. safety spec sanity: how often does the CURRENT agent's action trip a HARD veto? (should be ~0)
    hard = soft = checked = 0
    bytype = {}
    for r in rows[:safety_sample]:
        em = r["eval_meta"]; rt = r["runtime"]
        a = rt.get("current_agent_action")
        if not a:
            continue
        ob = _obs(em["episode_id"], em["step"], em["seat"])
        if ob is None:
            continue
        chk = AD.safety_check(ob, a[0])
        checked += 1
        if chk["hard_veto"]:
            hard += 1
        if chk["soft_flags"]:
            soft += 1
            for f in chk["soft_flags"]:
                bytype[f] = bytype.get(f, 0) + 1
    return {
        "mode": "A_NO_PROPOSER", "trace": trace, "rows": len(rows),
        "adapter_disabled_cannot_change_action": disabled_ok,
        "baseline_trace_agreement_pct": round(100 * agree / max(1, len(rows)), 1),
        "safety_sanity_on_agent_actions": {
            "checked": checked, "hard_veto_count": hard, "soft_flag_count": soft, "soft_by_filter": bytype,
            "interpretation": "the current agent's own actions should hard-veto ~never; a high hard count would mean the safety spec is too aggressive. Soft flags are advisory.",
        },
    }


def mode_b(logits_path):
    if not logits_path or not Path(logits_path).exists():
        return {"mode": "B_LOGITS_ONLY", "status": "WAITING", "expected_logits": logits_path,
                "note": "provide a Model A logits/ranks file (decision_id -> ranked semantic keys) to join."}
    return {"mode": "B_LOGITS_ONLY", "status": "LOGITS_PRESENT_BUT_JOIN_NOT_IMPLEMENTED_V0",
            "logits": logits_path, "note": "join by decision_id + semantic key; compute C0..C3 coverage + safety on top-k."}


def mode_c():
    if not PROPOSER_EXPORT.exists():
        return {"mode": "C_RUNTIME_PROPOSER", "status": "WAITING_FOR_MODEL_A_RUNTIME_PROPOSER",
                "expected_artifact": str(PROPOSER_EXPORT)}
    return {"mode": "C_RUNTIME_PROPOSER", "status": "ARTIFACT_PRESENT", "artifact": str(PROPOSER_EXPORT),
            "note": "load via adapter.load_proposer; run on recorded obs; compare to cached logits."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="A", choices=["A", "B", "C"])
    ap.add_argument("--trace", default="yushin_trace.jsonl")
    ap.add_argument("--logits", default=None)
    ap.add_argument("--safety-sample", type=int, default=500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "A":
        res = mode_a(a.trace, a.safety_sample)
        (OUT / "dry_run_no_proposer.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    elif a.mode == "B":
        res = mode_b(a.logits)
        (OUT / "proposer_bridge_dry_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    else:
        res = mode_c()
        (OUT / "proposer_bridge_dry_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
