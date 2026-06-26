"""Sections 5-7: the Model A join file (runtime state vs eval-only targets, strictly separated), the data-quality
report, and the review pack. Consumes the cohort traces + same_turn_sequences + turn_end_deltas + action_semantics.

  PYTHONIOENCODING=utf-8 python tools/transplant_join_v0.py
"""
from __future__ import annotations
import collections
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_transplant_support_v0"
TRACES = {
    "yushin": ROOT / "data/generated/starmie_bridge_trace_v0/yushin_trace.jsonl",
    "keidroid": ROOT / "data/generated/starmie_bridge_trace_v0/keidroid_trace.jsonl",
    "old_exact": ROOT / "data/generated/starmie_bridge_trace_v0/old_exact_trace_sample.jsonl",
}
# eval-only signals that must NEVER appear in the runtime section
FORBIDDEN_RUNTIME = ("pilot", "outcome", "won", "future", "result", "replay_link")


def _index(path, key="decision_id"):
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r[key]] = r
    return out


def main() -> int:
    seqs = _index(OUT / "same_turn_sequences.jsonl")
    dels = _index(OUT / "turn_end_deltas.jsonl")
    sems = _index(OUT / "action_semantics.jsonl")

    join = open(OUT / "model_a_transplant_join.jsonl", "w", encoding="utf-8")
    n = 0
    fam_counts = collections.Counter()
    have_seq = have_del = have_sem = 0
    dup = collections.Counter()
    ep_split = collections.defaultdict(set)   # episode -> {splits}
    ep_cohort = collections.defaultdict(set)
    runtime_leaks = 0
    missing_by_family = collections.defaultdict(lambda: {"seq": 0, "del": 0, "n": 0})
    examples = collections.defaultdict(list)

    for cohort, path in TRACES.items():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            did = r["decision_id"]
            dup[did] += 1
            em = r.get("eval_meta") or {}
            rt = r.get("runtime") or {}
            fam = em.get("pilot_action_family") or "?"
            fam_counts[fam] += 1
            split = em.get("split")
            ep = em.get("episode_id")
            if ep is not None:
                ep_split[ep].add(split)
                ep_cohort[ep].add(cohort)
            sq = seqs.get(did)
            dl = dels.get(did)
            sm = sems.get(did)
            have_seq += sq is not None
            have_del += dl is not None and not dl.get("missing", {}).get("turn_end", True)
            have_sem += sm is not None
            mb = missing_by_family[fam]
            mb["n"] += 1
            mb["seq"] += sq is None
            mb["del"] += dl is None or dl.get("missing", {}).get("turn_end", True)

            runtime = {
                "tactical_state": rt.get("tactical_state"),
                "option_index_to_semantic_key": rt.get("option_index_to_semantic_key"),
                "legal_semantic_keys": rt.get("legal_semantic_keys"),
                "current_agent_action_key": rt.get("current_agent_action_key"),
                "current_agent_action_family": rt.get("current_agent_action_family"),
                "search_action": rt.get("search_action"),
                "search_candidate_order": rt.get("search_candidate_order"),
                "support_status": rt.get("support_status"),
                "hard_safety_flags": rt.get("hard_safety_flags"),
                "action_semantics": (sm or {}).get("options"),
            }
            # leakage guard: the runtime section must carry no pilot/outcome/future signal
            blob = json.dumps(runtime).lower()
            if any(b in blob for b in FORBIDDEN_RUNTIME):
                runtime_leaks += 1
            eval_only = {
                "pilot_action_key": em.get("pilot_action_key"), "pilot_action_family": em.get("pilot_action_family"),
                "pilot_name": em.get("pilot_name"), "outcome_won": em.get("outcome_won"),
                "agreement": em.get("agreement"), "replay_link": em.get("replay_link"),
                "same_turn": {k: sq.get(k) for k in (sq or {}) if k not in ("decision_id", "cohort", "episode_id", "step", "seat")} if sq else None,
                "turn_end_delta": (dl or {}).get("turn_end_delta"), "next_own_delta": (dl or {}).get("next_own_delta"),
                "turn_end_step": (dl or {}).get("turn_end_step"), "delta_missing": (dl or {}).get("missing"),
            }
            row = {"decision_id": did, "episode_id": ep, "step": em.get("step"), "seat": em.get("seat"),
                   "split": split, "cohort": cohort, "runtime": runtime, "eval_only": eval_only}
            join.write(json.dumps(row, default=str) + "\n")
            n += 1
            # review examples
            if sq and dl and not dl.get("missing", {}).get("turn_end", True):
                te = dl.get("turn_end_delta") or {}
                tag = None
                if fam == "ATTACK" and sq.get("dev_actions_before_attack") == 0:
                    tag = "attack_now_no_dev"
                elif fam in ("ATTACH", "EVOLVE") and te.get("my_ready_main_attackers", 0) > 0:
                    tag = "develop_enabled_attacker"
                elif fam == "ATTACH" and ":Mega" in (em.get("pilot_action_key") or ""):
                    tag = "attach_mega_line"
                elif fam == "SELECT_CARD":
                    tag = "select_card"
                elif sq.get("pilot_attacked_later_same_turn") and fam in ("ATTACH", "PLAY", "EVOLVE"):
                    tag = "develop_then_attack"
                if tag and len(examples[tag]) < 12:
                    examples[tag].append({"decision_id": did, "cohort": cohort, "pilot_action": em.get("pilot_action_key"),
                                          "family": fam, "future": sq.get("future_same_turn_sequence"),
                                          "dev_before_attack": sq.get("dev_actions_before_attack"),
                                          "turn_end_delta": {k: te.get(k) for k in ("opp_board_hp", "my_ready_main_attackers", "my_units", "opp_prizes_left")},
                                          "outcome_won": em.get("outcome_won")})
    join.close()

    # split-leakage: an episode appearing in >1 split is leakage
    split_leak_eps = [str(e) for e, s in ep_split.items() if len([x for x in s if x]) > 1]
    cross_cohort_eps = [str(e) for e, c in ep_cohort.items() if len(c) > 1]
    dups = {k: v for k, v in dup.items() if v > 1}
    quality = {
        "decisions": n,
        "coverage": {"same_turn_seq": have_seq, "turn_end_delta_resolved": have_del, "action_semantics": have_sem,
                     "same_turn_pct": round(100 * have_seq / max(1, n), 1),
                     "turn_end_pct": round(100 * have_del / max(1, n), 1)},
        "family_support_counts": dict(fam_counts.most_common()),
        "missingness_by_family": {k: {"n": v["n"], "no_seq": v["seq"], "no_turn_end": v["del"],
                                      "turn_end_resolved_pct": round(100 * (v["n"] - v["del"]) / max(1, v["n"]), 1)}
                                  for k, v in sorted(missing_by_family.items(), key=lambda x: -x[1]["n"])},
        "duplicate_decision_ids": len(dups), "duplicate_examples": dict(list(dups.items())[:5]),
        "split_leakage_episodes": len(split_leak_eps), "split_leakage_examples": split_leak_eps[:5],
        "cross_cohort_episodes": len(cross_cohort_eps), "cross_cohort_examples": cross_cohort_eps[:5],
        "runtime_metadata_leaks": runtime_leaks,
        "runtime_eval_separation": "PASS" if runtime_leaks == 0 else "FAIL",
    }
    (OUT / "data_quality_report.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")

    # review pack
    review_rows = [{"category": tag, **ex} for tag, exs in examples.items() for ex in exs]
    with open(OUT / "review_examples.jsonl", "w", encoding="utf-8") as f:
        for r in review_rows:
            f.write(json.dumps(r, default=str) + "\n")
    css = ("body{font:13px/1.5 system-ui,sans-serif;margin:18px;background:#0f1117;color:#dde}h2{font-size:15px;"
           "border-bottom:1px solid #333;margin-top:22px}.c{border:1px solid #2a2f3a;border-radius:7px;padding:8px 11px;"
           "margin:6px 0;background:#161a22}.tag{display:inline-block;padding:0 6px;border-radius:4px;background:#22303f;"
           "margin-right:4px;font-size:11px}.k{color:#9fc5ff}.win{color:#7ee787}.loss{color:#ff7a7a}")
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
             "<h1>Transplant support pack: replay decisions with same-turn + turn-end consequences</h1>",
             f"<p class='tag'>decisions {n} | same-turn {quality['coverage']['same_turn_pct']}% | "
             f"turn-end {quality['coverage']['turn_end_pct']}% | runtime/eval separation {quality['runtime_eval_separation']}</p>"]
    for tag, exs in examples.items():
        parts.append(f"<h2>{html.escape(tag)} ({len(exs)})</h2>")
        for e in exs:
            wl = "win" if e.get("outcome_won") else "loss"
            parts.append("<div class='c'>"
                         f"<span class='tag {wl}'>{'won' if e.get('outcome_won') else 'lost'}</span>"
                         f"<span class='tag'>{e['cohort']}</span><span class='k'>{e['decision_id']}</span> "
                         f"<b>{html.escape(str(e['pilot_action']))}</b> "
                         f"future={html.escape(str(e.get('future')))} dev_before_attack={e.get('dev_before_attack')} "
                         f"turn_end_delta={html.escape(str(e.get('turn_end_delta')))}</div>")
    parts.append("</body></html>")
    (OUT / "review_examples.html").write_text("\n".join(parts), encoding="utf-8")

    print(json.dumps({"decisions": n, "coverage": quality["coverage"], "runtime_eval_separation": quality["runtime_eval_separation"],
                      "split_leakage_eps": quality["split_leakage_episodes"], "dup_ids": quality["duplicate_decision_ids"],
                      "cross_cohort_eps": quality["cross_cohort_episodes"], "review_examples": len(review_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
