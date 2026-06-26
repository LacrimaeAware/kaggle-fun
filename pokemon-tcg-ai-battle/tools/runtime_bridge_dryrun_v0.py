"""STARMIE RUNTIME PROPOSER BRIDGE -- LOGITS_ONLY dry-run (Sections 1-3). Read-only; consumes Model A's exported
runtime-proposer logits and Model B traces. Does NOT run Model A's model in-repo (its feature pipeline lives in
the 0557 worktree; re-implementing it would be a different model) and does NOT change gameplay.

Compatibility: Model A semantic keys are structured JSON (different format from Model B's compact keys), and
Model A PACKS semantically-equivalent options. So the join is by decision_id, and the comparison is at the
SEMANTIC level using Model B's own key mapper applied to Model A's per-packed-option raw_option_index.

  python tools/runtime_bridge_dryrun_v0.py
"""
from __future__ import annotations
import collections, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import learned_proposer_adapter as AD   # semantic keys + offline safety filters
import deck_policy_v3 as DP             # noqa

MA = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist")
EXPORT = MA / "starmie_behavior_proposer_runtime_v1.json"
LOGITS = [MA / "proposer_runtime_v1" / "validation_logits.jsonl", MA / "proposer_runtime_v1" / "test_logits.jsonl"]
TRACE = ROOT / "data" / "generated" / "starmie_bridge_trace_v0" / "yushin_trace.jsonl"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
OUT = ROOT / "data" / "generated" / "starmie_runtime_bridge_v0"
_EPC = {}


def _sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except Exception:
        return None


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- baseline freeze ----
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=ROOT).strip()
    import starmie_heuristics as SH
    baseline = {"git_head": head, "branch": branch, "starmie_heuristics_sha256": _sha(ROOT / "agent" / "starmie_heuristics.py"),
                "search_v3_sha256": _sha(ROOT / "agent" / "search_v3.py"), "eval_sha256": _sha(ROOT / "agent" / "eval.py"),
                "R15_default_on": SH._on("R15"), "ATTACH_MEGA_default_off": not SH.ATTACH_MEGA,
                "traced_agent": "deployed baseline = sub_starmie2 (R15 disabled in traces, ATTACH_MEGA off)"}
    (OUT / "baseline_manifest.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    # ---- compatibility audit ----
    export = json.loads(EXPORT.read_text(encoding="utf-8")) if EXPORT.exists() else None
    compat = {
        "proposer_export_exists": EXPORT.exists(), "export_sha256": _sha(EXPORT),
        "live_enabled": (export or {}).get("live_enabled"),
        "system_id": (export or {}).get("system_id"), "status": (export or {}).get("status"),
        "logits_present": [str(p) for p in LOGITS if p.exists()],
        "runtime_inference_in_repo": False,
        "runtime_blocker": "Model A's feature pipeline + card data + model live in the 0557 worktree; running it in kaggle-fun would require importing Model A code or re-implementing the feature pipeline (a different model). Using LOGITS_ONLY.",
        "semantic_key_format_mismatch": "Model A keys are structured JSON; Model B keys are compact strings -> join by decision_id, compare semantics via Model B key mapper on Model A's raw_option_index.",
        "join_key": "decision_id (':' normalised to '_') + raw_option_index", "mode": "LOGITS_ONLY",
    }
    (OUT / "proposer_compatibility.json").write_text(json.dumps(compat, indent=2), encoding="utf-8")

    # ---- load my trace ----
    TR = {}
    for l in open(TRACE, encoding="utf-8"):
        r = json.loads(l); rt = r["runtime"]; em = r["eval_meta"]
        a = rt.get("current_agent_action"); p = em.get("pilot_action")
        TR[r["decision_id"]] = {"agent": a[0] if a else None, "pilot": p[0] if p else None,
                                "fam": em["pilot_action_family"], "ep": em["episode_id"], "step": em["step"], "seat": em["seat"]}

    # ---- LOGITS_ONLY dry-run ----
    n = agent_cov = prop1 = prop3 = prop5 = union3 = adds = omits = 0
    safety_veto = ood = no_obs = 0
    fam = collections.defaultdict(lambda: {"n": 0, "agent": 0, "prop3": 0, "union3": 0})
    examples = []
    for lf in LOGITS:
        for line in open(lf, encoding="utf-8"):
            r = json.loads(line)
            if r.get("record_type") == "manifest" or "decision_id" not in r:
                continue
            did = r["decision_id"].replace(":", "_")
            t = TR.get(did)
            if not t:
                continue
            ob = _obs(t["ep"], t["step"], t["seat"])
            if not ob:
                no_obs += 1; continue
            i2k = AD.option_index_to_key(ob)
            # proposer ranking: packed options sorted by probability -> their raw_option_index -> Model B semantic key
            probs = r.get("final_probabilities") or []
            opts = r.get("options") or []
            order = sorted(range(len(opts)), key=lambda k: -(probs[k] if k < len(probs) else -1))
            prop_keys = []
            for packed in order:
                ri = opts[packed].get("raw_option_index")
                prop_keys.append(i2k.get(ri))
            agent_key = i2k.get(t["agent"]) if t["agent"] is not None else None
            pilot_key = i2k.get(t["pilot"]) if t["pilot"] is not None else None
            if pilot_key is None:
                continue
            n += 1
            top1, top3, top5 = prop_keys[:1], prop_keys[:3], prop_keys[:5]
            a_cov = agent_key == pilot_key
            p1 = pilot_key in top1; p3 = pilot_key in top3; p5 = pilot_key in top5
            u3 = a_cov or p3
            agent_cov += a_cov; prop1 += p1; prop3 += p3; prop5 += p5; union3 += u3
            if (not a_cov) and p3:
                adds += 1
            if a_cov and not p3:
                omits += 1
            # safety on proposer top-1 (its raw index)
            ri1 = opts[order[0]].get("raw_option_index") if order else None
            if ri1 is not None:
                chk = AD.safety_check(ob, ri1)
                if chk["hard_veto"]:
                    safety_veto += 1
            if r.get("support_status") and str(r.get("support_status")).upper() not in ("SUPPORTED", "IN_SUPPORT", "OK"):
                ood += 1
            f = fam[t["fam"]]; f["n"] += 1; f["agent"] += a_cov; f["prop3"] += p3; f["union3"] += u3
            if (not a_cov) and p3 and len(examples) < 12:
                examples.append({"decision_id": did, "family": t["fam"], "agent_key": agent_key,
                                 "pilot_key": pilot_key, "proposer_top3": top3})

    def pct(x):
        return round(100 * x / max(1, n), 1)
    report = {
        "mode": "LOGITS_ONLY", "cohort": "Yushin (val+test logit decisions)", "joined_decisions": n, "no_obs": no_obs,
        "coverage_pct": {"agent_alone": pct(agent_cov), "proposer_top1": pct(prop1), "proposer_top3": pct(prop3),
                         "proposer_top5": pct(prop5), "agent_union_proposer_top3": pct(union3)},
        "proposer_adds_pilot_when_agent_missed": adds, "proposer_omits_pilot_when_agent_had_it": omits,
        "safety_hard_veto_on_proposer_top1": safety_veto, "ood_unsupported": ood,
        "by_family": {k: {"n": v["n"], "agent_pct": round(100*v["agent"]/max(1, v["n"]), 1),
                          "proposer_top3_pct": round(100*v["prop3"]/max(1, v["n"]), 1),
                          "union3_pct": round(100*v["union3"]/max(1, v["n"]), 1)} for k, v in sorted(fam.items())},
        "examples_proposer_adds": examples,
    }
    (OUT / "dry_run_bridge_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_family", "examples_proposer_adds")}, indent=2))
    print("by family:", json.dumps(report["by_family"]))


if __name__ == "__main__":
    main()
