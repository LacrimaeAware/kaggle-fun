"""STARMIE TACTICAL-LEAF V1 -- Section 6: OFFLINE SELECTION AUDIT (P0 baseline vs P1 = baseline + continuity).

On TRIGGERED, search-decided corpus roots, run the real search with the leaf OFF (P0) and ON (P1) and measure
whether the term re-ranks the pick toward MAIN-ATTACKER CONTINUITY (the intended mechanism), plus pilot-agreement
deltas (secondary, NOT the promotion criterion). Stratified by tactical state. Uses replay-grouped wins+losses.

A "continuity-favoring" option = one that builds/uses the Mega Starmie line: ATTACH onto a Mega, EVOLVE into a
Mega, RETREAT (promote a ready benched Mega), or ATTACK with a Mega. (Heuristic direction label, not a claim of
optimality.)

  python tools/starmie_leaf_offline_audit_v1.py --sample 300 --budget 0.15
Output: data/generated/starmie_tactical_leaf_v1/offline_selection_audit.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
EXPORT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1" / "starmie_tactical_state_v1.jsonl"
OUT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1" / "offline_selection_audit.json"
REPLAY_DIRS = [ROOT / "data" / "external" / "replays",
               Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")]
MEGA, CIND, PLAY, ATTACH, EVOLVE, RETREAT, ATTACK = 1031, 666, 7, 8, 9, 12, 13
_EPC = {}


def _episode(epid):
    if epid in _EPC:
        return _EPC[epid]
    ep = None
    for d in REPLAY_DIRS:
        p = d / f"{epid}.json"
        if p.exists():
            try:
                ep = json.load(open(p, encoding="utf-8"))
            except Exception:
                ep = None
            break
    if len(_EPC) > 48:
        _EPC.clear()
    _EPC[epid] = ep
    return ep


def _obs(ptr):
    ep = _episode(ptr.get("episode"))
    if not ep:
        return None
    steps = ep.get("steps") if isinstance(ep, dict) else ep
    try:
        return steps[ptr["step"]][ptr["seat"]].get("observation")
    except Exception:
        return None


def _is_triggered(rt):
    bf = rt["board_features"]
    ents = rt["entity_features"]
    my_active = next((e for e in ents if e["owner"] == "me" and e["slot"] == "active"), None)
    mega_bench_ready = any(e["owner"] == "me" and e["slot"] == "bench" and e["is_main_attacker"]
                           and (e["attack_ready"] or e["one_attachment_from_ready"]) for e in ents)
    return (bf["my_main_one_short"] >= 1 or bf["engine_overinvestment_units"] > 0
            or (my_active and my_active["is_energy_engine"] and mega_bench_ready)
            or (bf["my_ready_main_attackers"] == 0 and bf["my_main_one_short"] >= 1))


def _continuity_favoring(opt, obs, DP):
    """Does this option build/use the Mega line? ATTACH onto Mega / EVOLVE into Mega / RETREAT(promote) / ATTACK Mega."""
    t = opt.get("type")
    if t == RETREAT:
        return True   # in this deck retreating the active is to promote a (ready) Mega/attacker
    if t == ATTACK:
        # Mega attacks are Jetting(1487)/Nebula(1488); Turbo(965) is the Cinderace engine
        return opt.get("attackId") in (1487, 1488)
    cid = None
    try:
        cid = DP.option_card_id(opt, obs)
    except Exception:
        cid = None
    if t == EVOLVE:
        return cid == MEGA
    if t == ATTACH:
        tgt = None
        try:
            tgt = DP.option_target_entity(opt, obs)
        except Exception:
            tgt = None
        return bool(tgt and DP._cid(tgt) == MEGA)
    return False


def _strata(coords):
    out = []
    rs = coords.get("RACE_STATE", {})
    vs = coords.get("VALUE_STATE", {})
    if rs.get("my_immediate_ko") or rs.get("opp_immediate_ko"):
        out.append("race")
    if (vs.get("prize_diff") or 0) > 0:
        out.append("value_ahead")
    elif (vs.get("prize_diff") or 0) < 0:
        out.append("value_behind")
    else:
        out.append("value_even")
    return out or ["other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--budget", type=float, default=0.15)
    a = ap.parse_args()
    sys.path.insert(0, str(AGENT))
    import deck_policy_v3 as DP, search_v3 as S, eval as EV, starmie_heuristics as SH
    S.USE_DYNAMIC_ATTACKS = True
    S.DEFAULT_BUDGET = a.budget
    deck = list(SH.STARMIE_DECK)

    rows = []
    for line in open(EXPORT, encoding="utf-8"):
        r = json.loads(line)
        if r["runtime"]["baseline_source"] == "search_or_default" and _is_triggered(r["runtime"]):
            rows.append(r)
    # deterministic spread across splits without RNG: take an even stride
    if len(rows) > a.sample:
        stride = len(rows) / a.sample
        rows = [rows[int(i * stride)] for i in range(a.sample)]
    print(f"triggered+search-decided roots sampled: {len(rows)} (budget {a.budget}s)", flush=True)

    n = changed = cont_dir = pilot0 = pilot1 = no_obs = 0
    strat = {}
    examples = []
    for k, r in enumerate(rows):
        obs = _obs(r["eval_meta"] | {"episode": r["eval_meta"]["replay_id"]})
        if not obs:
            no_obs += 1; continue
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not opts:
            continue
        try:
            EV.ATTACKER_CONTINUITY_ON = False
            p0 = S.best_option(obs, deck, leaf_mode="deckout", rollout_mode="develop")
            EV.ATTACKER_CONTINUITY_ON = True
            p1 = S.best_option(obs, deck, leaf_mode="deckout", rollout_mode="develop")
        except Exception:
            continue
        finally:
            EV.ATTACKER_CONTINUITY_ON = False
        if p0 is None or p1 is None:
            continue
        n += 1
        pilot = r["eval_meta"]["pilot_action"]
        ch = list(p0) != list(p1)
        for st in _strata(r["runtime"]["tactical_coordinates"]):
            d = strat.setdefault(st, {"n": 0, "changed": 0, "cont_dir": 0})
            d["n"] += 1
            if ch:
                d["changed"] += 1
        if ch:
            changed += 1
            # direction: did p1 move to a continuity-favoring option that p0 did NOT pick?
            p1_opt = opts[p1[0]] if p1 and 0 <= p1[0] < len(opts) else None
            p0_opt = opts[p0[0]] if p0 and 0 <= p0[0] < len(opts) else None
            if p1_opt is not None and _continuity_favoring(p1_opt, obs, DP) and not (p0_opt is not None and _continuity_favoring(p0_opt, obs, DP)):
                cont_dir += 1
                for st in _strata(r["runtime"]["tactical_coordinates"]):
                    strat[st]["cont_dir"] += 1
                if len(examples) < 12:
                    examples.append({"decision_id": r["decision_id"], "family": r["runtime"]["action_family"],
                                     "p0": list(p0), "p1": list(p1), "pilot": pilot,
                                     "p1_option_type": p1_opt.get("type")})
        if pilot is not None:
            if list(p0) == list(pilot):
                pilot0 += 1
            if list(p1) == list(pilot):
                pilot1 += 1
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(rows)} | changed {changed} cont_dir {cont_dir}", flush=True)

    res = {
        "section": "S6_offline_selection_audit", "budget": a.budget,
        "roots_evaluated": n, "no_obs": no_obs,
        "picks_changed": changed, "picks_changed_pct": round(100 * changed / max(1, n), 1),
        "changed_toward_continuity": cont_dir,
        "changed_toward_continuity_pct_of_changed": round(100 * cont_dir / max(1, changed), 1),
        "pilot_agreement_triggered_baseline": round(100 * pilot0 / max(1, n), 1),
        "pilot_agreement_triggered_continuity": round(100 * pilot1 / max(1, n), 1),
        "by_stratum": strat,
        "examples_changed_toward_continuity": examples,
        "note": ("Pilot agreement is reported but is NOT the promotion criterion (Section 6). The mechanism signal "
                 "is picks_changed_toward_continuity on triggered states."),
    }
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_stratum", "examples_changed_toward_continuity")}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
