"""STARMIE PROPOSER-BRIDGE TRACE LOGGER V0 -- analysis/derivation (Sections 6-9).

Consumes the cohort traces and writes: agreement_gap_report.json, tactical_feature_summary.json,
model_a_bridge_input.jsonl, proposer_bridge_eval.json OR proposer_bridge_waiting.json, review_examples.jsonl,
review_examples.html, closeout.json. Read-only.

  python tools/build_bridge_analysis_v0.py
"""
from __future__ import annotations
import collections, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_bridge_trace_v0"
PROPOSER = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/generated/starmie_specialist/starmie_behavior_proposer_v0_export.json")
COHORTS = {"D0_YUSHIN_TOP1": "yushin_trace.jsonl", "D1_KEIDROID_TOP2": "keidroid_trace.jsonl",
           "D2_OLD_EXACT_ALL": "old_exact_trace_sample.jsonl"}


def _load(name):
    p = OUT / name
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def _rate(n, d):
    return round(100 * n / d, 1) if d else None


def gap_report(rows):
    fam = collections.defaultdict(lambda: [0, 0])
    src = collections.Counter()
    for r in rows:
        f = r["eval_meta"]["pilot_action_family"]
        fam[f][0] += int(r["eval_meta"]["agreement"]); fam[f][1] += 1
        src[r["runtime"]["active_source"]] += 1
    ov = [sum(v[0] for v in fam.values()), sum(v[1] for v in fam.values())]
    # specific gaps
    av = [r for r in rows if r["runtime"]["tactical_state"]["COMMITMENT_STATE"].get("nonterminal_attack_available")
          or r["runtime"]["tactical_state"]["COMMITMENT_STATE"].get("guaranteed_ko_available")]
    pilot_atk = sum(1 for r in av if r["eval_meta"]["pilot_action_family"] == "ATTACK")
    agent_atk = sum(1 for r in av if str(r["runtime"]["current_agent_action_key"] or "").startswith("ATTACK"))
    ah = [r for r in rows if r["eval_meta"]["pilot_action_family"] == "ATTACH"]
    pilot_mega = sum(1 for r in ah if r["eval_meta"]["pilot_choice_meta"].get("attach_target_role") == "Mega")
    agent_cind = sum(1 for r in ah if r["runtime"]["agent_choice_meta"].get("attach_target_role") == "Cinderace")
    return {
        "overall_agreement_pct": _rate(*ov), "n": ov[1],
        "by_family": {k: {"agree_pct": _rate(v[0], v[1]), "n": v[1]} for k, v in sorted(fam.items())},
        "active_source": dict(src),
        "attack_when_available": {"n": len(av), "pilot_attack_pct": _rate(pilot_atk, len(av)),
                                  "agent_attack_pct": _rate(agent_atk, len(av))},
        "attach_target": {"n_attach": len(ah), "pilot_mega_pct": _rate(pilot_mega, len(ah)),
                          "agent_cinderace_pct": _rate(agent_cind, len(ah))},
    }


def main():
    data = {c: _load(f) for c, f in COHORTS.items()}
    # 6. agreement/gap report
    gaps = {c: gap_report(rows) for c, rows in data.items() if rows}
    (OUT / "agreement_gap_report.json").write_text(json.dumps(gaps, indent=2, default=str), encoding="utf-8")

    # 7. model_a bridge input (lean projection, join by decision_id)
    n_bridge = 0
    with open(OUT / "model_a_bridge_input.jsonl", "w", encoding="utf-8") as o:
        for c, rows in data.items():
            for r in rows:
                rt = r["runtime"]; em = r["eval_meta"]
                o.write(json.dumps({
                    "decision_id": r["decision_id"], "cohort": c,
                    "option_index_to_semantic_key": rt["option_index_to_semantic_key"],
                    "legal_semantic_keys": rt["legal_semantic_keys"],
                    "current_agent_action_key": rt["current_agent_action_key"],
                    "current_agent_action_family": rt["current_agent_action_family"],
                    "active_source": rt["active_source"],
                    "runtime_tactical": {"board": rt["tactical_state"]["board"],
                                         "RACE_STATE": rt["tactical_state"]["RACE_STATE"],
                                         "COMMITMENT_STATE": rt["tactical_state"]["COMMITMENT_STATE"],
                                         "VALUE_STATE": rt["tactical_state"]["VALUE_STATE"]},
                    "eval_meta": {"pilot_action_key": em["pilot_action_key"], "pilot_action_family": em["pilot_action_family"],
                                  "pilot_name": em["pilot_name"], "outcome_won": em["outcome_won"], "split": em["split"]},
                }, default=str) + "\n")
                n_bridge += 1

    # 8. proposer bridge eval OR waiting
    if PROPOSER.exists():
        bridge = {"status": "PROPOSER_FOUND_BUT_EVAL_NOT_IMPLEMENTED_HERE",
                  "proposer_path": str(PROPOSER),
                  "note": "Model A proposer export exists; run the bridge candidate-set eval (Section 8)."}
        (OUT / "proposer_bridge_eval.json").write_text(json.dumps(bridge, indent=2), encoding="utf-8")
        bridge_verdict = "A_PROPOSER_BRIDGE_EVAL_READY (artifact present; eval pass pending)"
    else:
        waiting = {
            "status": "WAITING_FOR_MODEL_A_PROPOSER",
            "expected_proposer_export": str(PROPOSER),
            "also_expected": [
                "pokemon-ai-agent/data/generated/starmie_specialist/proposer_v0_quality_report.json",
                "pokemon-ai-agent/data/generated/starmie_specialist/proposer_v0_candidate_set_report.json"],
            "bridge_input_ready": "data/generated/starmie_bridge_trace_v0/model_a_bridge_input.jsonl",
            "join_key": "decision_id",
            "instructions": "When the proposer export + per-decision logits exist, join by decision_id to the bridge input; compute candidate-set coverage C0..C4 (current_agent_action +/- proposer_top1/3/5) per Section 8.",
        }
        (OUT / "proposer_bridge_waiting.json").write_text(json.dumps(waiting, indent=2), encoding="utf-8")
        bridge_verdict = "B_WAITING_FOR_MODEL_A_PROPOSER"

    # 9. review pack (agent-vs-pilot disagreements by family)
    pack = []
    for c, rows in data.items():
        for fam, k in (("ATTACH", 8), ("SELECT_CARD", 6), ("ATTACK", 6), ("PLAY", 6), ("RETREAT", 3)):
            dis = [r for r in rows if r["eval_meta"]["pilot_action_family"] == fam and not r["eval_meta"]["agreement"]]
            for r in dis[:k]:
                pack.append({"cohort": c, "decision_id": r["decision_id"], "family": fam,
                             "pilot_key": r["eval_meta"]["pilot_action_key"], "agent_key": r["runtime"]["current_agent_action_key"],
                             "source": r["runtime"]["active_source"],
                             "board": {kk: r["runtime"]["tactical_state"]["board"].get(kk) for kk in ("prize_diff", "my_ready_main_attackers", "my_units")},
                             "replay_link": r["eval_meta"]["replay_link"], "episode": r["eval_meta"]["episode_id"], "step": r["eval_meta"]["step"]})
    with open(OUT / "review_examples.jsonl", "w", encoding="utf-8") as o:
        for p in pack:
            o.write(json.dumps(p, default=str) + "\n")
    htmlrows = "".join(
        f"<tr><td>{html.escape(p['cohort'])}</td><td>{html.escape(p['family'])}</td>"
        f"<td><b>{html.escape(str(p['pilot_key']))}</b></td><td>{html.escape(str(p['agent_key']))}</td>"
        f"<td>{html.escape(str(p['source']))}</td><td><small>{html.escape(json.dumps(p['board']))}</small></td>"
        f"<td><a href='{html.escape(p['replay_link'])}'>{p['episode']}</a>:{p['step']}</td></tr>" for p in pack)
    (OUT / "review_examples.html").write_text(
        "<html><head><meta charset='utf-8'><style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;font:13px sans-serif}b{color:#0a7}</style></head>"
        "<body><h2>Bridge trace: current agent vs pilot disagreements</h2><table>"
        "<tr><th>cohort</th><th>family</th><th>pilot</th><th>agent</th><th>src</th><th>board</th><th>episode:step</th></tr>"
        + htmlrows + "</table></body></html>", encoding="utf-8")

    # tactical feature summary
    tfs = {c: {"n": len(rows),
               "commitment_true_rate": {k: round(sum(1 for r in rows if r["runtime"]["tactical_state"]["COMMITMENT_STATE"].get(k)) / max(1, len(rows)), 3)
                                        for k in ("guaranteed_ko_available", "nonterminal_attack_available", "attachment_unused", "information_action_available")}}
              for c, rows in data.items() if rows}
    (OUT / "tactical_feature_summary.json").write_text(json.dumps(tfs, indent=2), encoding="utf-8")

    trace_verdict = ("A_TRACE_DATA_READY_FOR_MODEL_A" if all(data.values()) and n_bridge > 1000
                     else "B_TRACE_DATA_DIRECTIONAL_ONLY" if any(data.values()) else "C_TRACE_PIPELINE_INVALID")
    closeout = {"TRACE_VERDICT": trace_verdict, "BRIDGE_VERDICT": bridge_verdict,
                "cohort_rows": {c: len(rows) for c, rows in data.items()},
                "bridge_input_rows": n_bridge, "review_examples": len(pack),
                "agreement": {c: g["overall_agreement_pct"] for c, g in gaps.items()},
                "artifacts": [p.name for p in sorted(OUT.glob("*.json*")) + sorted(OUT.glob("*.html"))]}
    (OUT / "closeout.json").write_text(json.dumps(closeout, indent=2, default=str), encoding="utf-8")
    print(json.dumps(closeout, indent=2))


if __name__ == "__main__":
    main()
