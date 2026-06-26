"""STARMIE V2 BEHAVIORAL ATLAS -- analysis + interaction mining (Sections 2-9).

Consumes the foundation tables (newtop1_decisions.jsonl, keidroid_decisions.jsonl) and writes the atlas
artifacts: agent-gap, cross-pilot comparison, ranked interaction candidates, review pack, verdict. Read-only;
no training, no heuristic changes, public pre-action features only.

  python tools/build_v2_atlas_analysis.py
"""
from __future__ import annotations
import collections, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATLAS = ROOT / "data" / "starmie_audit" / "v2_behavior_atlas"
YU = ATLAS / "newtop1_decisions.jsonl"
KE = ATLAS / "keidroid_decisions.jsonl"


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def _rate(num, den):
    return round(100 * num / den, 1) if den else None


def _lift(rows, pred, outcome):
    """Univariate association: outcome rate when pred true vs false, + coverage + lift."""
    a = [r for r in rows if pred(r)]
    b = [r for r in rows if not pred(r)]
    pa = sum(1 for r in a if outcome(r)); pb = sum(1 for r in b if outcome(r))
    ra = pa / len(a) if a else 0.0
    rb = pb / len(b) if b else 0.0
    return {"coverage": len(a), "outcome_rate_pred": round(ra, 3), "outcome_rate_not": round(rb, 3),
            "lift": round(ra - rb, 3)}


def agent_gap(yu):
    fam_agree = collections.defaultdict(lambda: [0, 0])
    src = collections.Counter()
    for r in yu:
        if "agree" not in r:
            continue
        fam_agree[r["family"]][0] += int(r["agree"]); fam_agree[r["family"]][1] += 1
        src[r.get("agent_source")] += 1
    overall = [sum(v[0] for v in fam_agree.values()), sum(v[1] for v in fam_agree.values())]
    # develop-vs-attack: among decisions where an attack is available, did the pilot/agent ATTACK?
    atk_avail = [r for r in yu if (r["feat"].get("ko_avail") or r["feat"].get("nonterm_attack"))]
    pilot_atk = sum(1 for r in atk_avail if r["family"] == "ATTACK")
    agent_atk = sum(1 for r in atk_avail if str(r.get("agent_choice", "")).startswith("ATTACK"))
    # disagreement patterns by family
    dis = collections.Counter()
    for r in yu:
        if r.get("agree") is False:
            dis[(r["family"], r.get("pilot_choice"), r.get("agent_choice"))] += 1
    return {
        "overall_agreement_pct": _rate(*overall), "overall_n": overall[1],
        "by_family": {k: {"agree_pct": _rate(v[0], v[1]), "n": v[1]} for k, v in sorted(fam_agree.items())},
        "agent_source_breakdown": dict(src),
        "develop_vs_attack_when_attack_available": {
            "n_decisions_attack_available": len(atk_avail),
            "pilot_attack_rate_pct": _rate(pilot_atk, len(atk_avail)),
            "agent_attack_rate_pct": _rate(agent_atk, len(atk_avail)),
            "ko_available_pilot_attack_pct": _rate(
                sum(1 for r in atk_avail if r["feat"].get("ko_avail") and r["family"] == "ATTACK"),
                sum(1 for r in atk_avail if r["feat"].get("ko_avail"))),
            "chip_only_pilot_attack_pct": _rate(
                sum(1 for r in atk_avail if not r["feat"].get("ko_avail") and r["family"] == "ATTACK"),
                sum(1 for r in atk_avail if not r["feat"].get("ko_avail"))),
        },
        "top_disagreement_patterns": [
            {"family": k[0], "pilot": k[1], "agent": k[2], "count": c}
            for k, c in dis.most_common(25)],
    }


def _choice_dist(rows, family, keyfn):
    sub = [r for r in rows if r["family"] == family]
    c = collections.Counter(keyfn(r) for r in sub)
    n = sum(c.values())
    return {"n": n, "dist_pct": {k: _rate(v, n) for k, v in c.most_common(8)}}


def cross_pilot(yu, ke):
    def block(rows):
        atk_avail = [r for r in rows if (r["feat"].get("ko_avail") or r["feat"].get("nonterm_attack"))]
        return {
            "n_decisions": len(rows),
            "family_mix_pct": {k: _rate(v, len(rows)) for k, v in collections.Counter(r["family"] for r in rows).most_common()},
            "attack_choice_dist": _choice_dist(rows, "ATTACK", lambda r: r["pilot_choice"]),
            "attach_target_dist": _choice_dist(rows, "ATTACH", lambda r: r.get("pilot_target_role") or "?"),
            "play_card_dist": _choice_dist(rows, "PLAY", lambda r: r["pilot_choice"]),
            "select_card_dist": _choice_dist(rows, "SELECT_CARD", lambda r: r["pilot_choice"]),
            "develop_vs_attack": {
                "attack_available_n": len(atk_avail),
                "attack_rate_pct": _rate(sum(1 for r in atk_avail if r["family"] == "ATTACK"), len(atk_avail)),
                "chip_only_attack_rate_pct": _rate(
                    sum(1 for r in atk_avail if not r["feat"].get("ko_avail") and r["family"] == "ATTACK"),
                    sum(1 for r in atk_avail if not r["feat"].get("ko_avail"))),
            },
            "retreat_rate_pct": _rate(sum(1 for r in rows if r["family"] == "RETREAT"), len(rows)),
        }
    return {"yushin": block(yu), "keidroid": block(ke)}


def mine_candidates(yu, ke):
    cands = []

    def add(cid, family, hyp, variables, support, agent_gap, cross, use, risk, examples):
        cands.append({"id": cid, "family": family, "hypothesis": hyp, "variables": variables,
                      "support": support, "agent_gap": agent_gap, "cross_pilot": cross,
                      "recommended_use": use, "risk": risk, "examples": examples})

    # ---- ATTACK: Nebula vs Jetting by energy units ----
    atk = [r for r in yu if r["family"] == "ATTACK"]
    atk_ke = [r for r in ke if r["family"] == "ATTACK"]
    def is_neb(r): return "Nebula" in str(r["pilot_choice"])
    yu_neb = _lift(atk, lambda r: (r["feat"].get("my_units") or 0) >= 3, is_neb)
    ke_neb = _lift(atk_ke, lambda r: (r["feat"].get("my_units") or 0) >= 3, is_neb)
    add("attack_nebula_at_3plus_units_v1", "ATTACK",
        "Pilot chooses Nebula Beam (vs Jetting) when the active has >=3 energy units.",
        ["my_units>=3", "choice=Nebula"], yu_neb,
        {"agent_choice_overlap": "compare agent Nebula rate in same states"}, {"keidroid": ke_neb},
        "search_leaf_or_heuristic", "Nebula spends Ignition; only worth it when it KOs or out-damages Jetting",
        [r["decision_id"] for r in atk if (r["feat"].get("my_units") or 0) >= 3 and is_neb(r)][:5])

    # ---- ATTACH: target is the Mega attacker ----
    ah = [r for r in yu if r["family"] == "ATTACH"]
    ah_ke = [r for r in ke if r["family"] == "ATTACH"]
    def tgt_mega(r): return r.get("pilot_target_role") == "Mega"
    yu_am = _lift(ah, lambda r: (r["feat"].get("my_main_one_short") or 0) >= 1, tgt_mega)
    ke_am = _lift(ah_ke, lambda r: (r["feat"].get("my_main_one_short") or 0) >= 1, tgt_mega)
    add("attach_to_mega_when_one_short_v1", "ATTACH",
        "Pilot attaches energy to the Mega Starmie (not engine/basic) when a Mega is one attachment from ready.",
        ["my_main_one_short>=1", "target=Mega"], yu_am,
        {"note": "compare to engine_overinvest in agent"}, {"keidroid": ke_am},
        "heuristic_or_feature", "target-role detection must be correct; bench vs active matters",
        [r["decision_id"] for r in ah if (r["feat"].get("my_main_one_short") or 0) >= 1 and tgt_mega(r)][:5])

    # ---- ATTACH target: Mega attacker vs Cinderace engine (resolved targets) ----
    def pmega(r): return r.get("pilot_target_role") == "Mega"
    yu_pm = sum(1 for r in ah if pmega(r))
    ag_cind = sum(1 for r in ah if r.get("agent_target_role") == "Cinderace")
    ke_pm = sum(1 for r in ah_ke if r.get("pilot_target_role") == "Mega")
    add("attach_energy_to_mega_not_engine_v1", "ATTACH",
        "Pilot attaches energy to the Mega Starmie attacker (~80%), almost never to the Cinderace engine; our agent attaches to Cinderace far more.",
        ["target=Mega", "target!=Cinderace"],
        {"coverage": len(ah), "pilot_mega_target_rate": round(yu_pm / max(1, len(ah)), 3),
         "agent_cinderace_target_rate": round(ag_cind / max(1, len(ah)), 3)},
        {"gap": "agent over-attaches the engine"},
        {"keidroid_mega_target_rate": round(ke_pm / max(1, len(ah_ke)), 3)},
        "heuristic_or_search_leaf", "respect Ignition-for-Nebula timing; don't starve a needed Cinderace ramp turn",
        [r["decision_id"] for r in ah if pmega(r) and r.get("agent_target_role") == "Cinderace"][:5])

    # ---- DEVELOP-vs-ATTACK: chip attack discipline ----
    av = [r for r in yu if (r["feat"].get("ko_avail") or r["feat"].get("nonterm_attack"))]
    av_ke = [r for r in ke if (r["feat"].get("ko_avail") or r["feat"].get("nonterm_attack"))]
    def did_attack(r): return r["family"] == "ATTACK"
    yu_ko = _lift(av, lambda r: bool(r["feat"].get("ko_avail")), did_attack)
    ke_ko = _lift(av_ke, lambda r: bool(r["feat"].get("ko_avail")), did_attack)
    add("attack_when_ko_available_v1", "ATTACK",
        "Pilot takes the attack on the turn-ending decision when a guaranteed KO is available (vs developing).",
        ["ko_avail", "chose ATTACK"], yu_ko,
        {"agent": "agent already has a KO floor; gap measures chip discipline"}, {"keidroid": ke_ko},
        "heuristic", "KO detection must be correct; a develop-first move can still precede the same-turn KO",
        [r["decision_id"] for r in av if r["feat"].get("ko_avail") and did_attack(r)][:5])

    # ---- SELECT_CARD / tutor: dominant fetch targets ----
    sc = [r for r in yu if r["family"] == "SELECT_CARD"]
    top_fetch = collections.Counter(r["pilot_choice"] for r in sc).most_common(5)
    add("tutor_target_priority_v1", "SELECT_CARD",
        "Pilot's tutor/select-card targets concentrate on specific cards (priority order observed).",
        ["select_card target"], {"coverage": len(sc), "top_targets": top_fetch},
        {"note": "compare to agent tutor target agreement"},
        {"keidroid_top": collections.Counter(r["pilot_choice"] for r in ke if r["family"] == "SELECT_CARD").most_common(5)},
        "heuristic_or_feature", "depends on board need; static priority may misfire", [r["decision_id"] for r in sc][:5])

    # ---- PLAY: Wally/Hammer/Boss usage ----
    pl = [r for r in yu if r["family"] == "PLAY"]
    play_dist = collections.Counter(r["pilot_choice"] for r in pl).most_common(8)
    add("play_card_priority_v1", "PLAY",
        "Pilot's PLAY usage (Wally/Hammer/Boss/Cape/draw/search) frequency profile.",
        ["play card"], {"coverage": len(pl), "dist": play_dist},
        {"note": "compare to agent PLAY agreement"},
        {"keidroid": collections.Counter(r["pilot_choice"] for r in ke if r["family"] == "PLAY").most_common(8)},
        "visual_review_or_feature", "many of these are situational; review before encoding", [r["decision_id"] for r in pl][:5])

    return cands


def review_pack(yu):
    pack = []
    for fam, k in (("ATTACH", 10), ("SELECT_CARD", 10), ("ATTACK", 10), ("PLAY", 10), ("RETREAT", 5)):
        dis = [r for r in yu if r["family"] == fam and r.get("agree") is False]
        for r in dis[:k]:
            pack.append({"family": fam, "decision_id": r["decision_id"], "episode": r["episode"], "step": r["step"],
                         "pilot_choice": r.get("pilot_choice"), "agent_choice": r.get("agent_choice"),
                         "agent_source": r.get("agent_source"), "feat": r["feat"]})
    return pack


def write_html(pack):
    rows = []
    for p in pack:
        f = p["feat"]
        feat = ", ".join(f"{k}={f[k]}" for k in ("prize_diff", "my_ready_main", "my_units", "ko_avail", "nonterm_attack", "my_main_one_short") if k in f)
        rows.append(f"<tr><td>{html.escape(p['family'])}</td><td><b>{html.escape(str(p['pilot_choice']))}</b></td>"
                    f"<td>{html.escape(str(p['agent_choice']))}</td><td>{html.escape(str(p['agent_source']))}</td>"
                    f"<td><small>{html.escape(feat)}</small></td>"
                    f"<td><a href='https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?episodeId={p['episode']}'>{p['episode']}</a>:{p['step']}</td></tr>")
    return ("<html><head><meta charset='utf-8'><style>table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:4px;font-family:sans-serif;font-size:13px}b{color:#0a7}</style></head><body>"
            "<h2>Yushin Ito (#1) vs current agent -- top disagreements</h2>"
            "<p>pilot choice (green) vs agent choice, with pre-action features. Episode links to Kaggle.</p>"
            "<table><tr><th>family</th><th>pilot</th><th>agent</th><th>agent src</th><th>features</th><th>episode:step</th></tr>"
            + "".join(rows) + "</table></body></html>")


def main():
    yu, ke = _load(YU), _load(KE)
    if not yu:
        raise SystemExit("run build_v2_behavior_atlas.py first (newtop1_decisions.jsonl missing)")

    gap = agent_gap(yu)
    (ATLAS / "yushin_vs_agent_gap.json").write_text(json.dumps(gap, indent=2, default=str), encoding="utf-8")
    cx = cross_pilot(yu, ke)
    (ATLAS / "yushin_vs_old_pilots.json").write_text(json.dumps(cx, indent=2, default=str), encoding="utf-8")
    cands = mine_candidates(yu, ke)
    with open(ATLAS / "interaction_candidates.jsonl", "w", encoding="utf-8") as o:
        for c in cands:
            o.write(json.dumps(c, default=str) + "\n")
    summary = {"n_candidates": len(cands),
               "candidates": [{"id": c["id"], "family": c["family"], "hypothesis": c["hypothesis"],
                               "support_coverage": (c["support"] or {}).get("coverage"),
                               "lift": (c["support"] or {}).get("lift"),
                               "recommended_use": c["recommended_use"]} for c in cands]}
    (ATLAS / "interaction_candidate_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pack = review_pack(yu)
    with open(ATLAS / "top_review_examples.jsonl", "w", encoding="utf-8") as o:
        for p in pack:
            o.write(json.dumps(p, default=str) + "\n")
    (ATLAS / "top_review_examples.html").write_text(write_html(pack), encoding="utf-8")

    # verdict
    matches_old = abs((cx["yushin"]["develop_vs_attack"]["attack_rate_pct"] or 0)
                      - (cx["keidroid"]["develop_vs_attack"]["attack_rate_pct"] or 0)) < 5
    has_lift = any(abs((c["support"] or {}).get("lift", 0) or 0) >= 0.15 for c in cands)
    verdict = ("A_CLEAR_INTERACTION_CANDIDATES_FOUND" if has_lift else
               ("C_NEW_PILOT_MATCHES_OLD_CORPUS" if matches_old else "B_DIRECTIONAL_CANDIDATES_ONLY"))
    closeout = {"verdict": verdict, "n_yushin_decisions": len(yu), "n_keidroid_decisions": len(ke),
                "overall_agreement_pct": gap["overall_agreement_pct"],
                "develop_vs_attack": {"yushin": cx["yushin"]["develop_vs_attack"],
                                      "keidroid": cx["keidroid"]["develop_vs_attack"],
                                      "agent_attack_rate_when_available_pct": gap["develop_vs_attack_when_attack_available"]["agent_attack_rate_pct"]},
                "top5_for_heuristic_test": [c["id"] for c in cands][:5],
                "artifacts": [p.name for p in sorted(ATLAS.glob("*")) if p.suffix in (".json", ".jsonl", ".html")]}
    (ATLAS / "closeout.json").write_text(json.dumps(closeout, indent=2, default=str), encoding="utf-8")

    print(f"VERDICT: {verdict}")
    print(f"overall agent agreement: {gap['overall_agreement_pct']}% over {gap['overall_n']} decisions")
    print(f"by family: " + " ".join(f"{k}={v['agree_pct']}%" for k, v in gap['by_family'].items()))
    dva = gap["develop_vs_attack_when_attack_available"]
    print(f"attack-available: pilot attacks {dva['pilot_attack_rate_pct']}% vs agent {dva['agent_attack_rate_pct']}% "
          f"(KO-avail pilot {dva['ko_available_pilot_attack_pct']}%, chip-only pilot {dva['chip_only_pilot_attack_pct']}%)")
    print(f"cross-pilot chip-attack rate: yushin {cx['yushin']['develop_vs_attack']['chip_only_attack_rate_pct']}% "
          f"keidroid {cx['keidroid']['develop_vs_attack']['chip_only_attack_rate_pct']}%")
    print(f"candidates: {len(cands)} | wrote atlas artifacts to {ATLAS}")


if __name__ == "__main__":
    main()
