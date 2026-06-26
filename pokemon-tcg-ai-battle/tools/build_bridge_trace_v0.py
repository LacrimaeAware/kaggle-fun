"""STARMIE PROPOSER-BRIDGE TRACE LOGGER V0 (Sections 2-5).

Produces aligned decision traces so Model A can later join its learned-proposer logits by decision_id. Runs the
CURRENT Starmie heuristic/search agent (deployed baseline = sub_starmie2 behaviour: R15 disabled, ATTACH_MEGA
off) on recorded pilot states -- READ-ONLY, does NOT change gameplay. Each row strictly separates a public
`runtime` payload from `eval_meta` (pilot/outcome/future/split are eval-only). Adds canonical SEMANTIC ACTION
KEYS + an option-index->key map so the proposer can be joined.

Trace mode = just running the agent + logging; no behaviour change (search candidate order via a SEPARATE
read-only option_evals call, off by default to keep it fast).

  python tools/build_bridge_trace_v0.py --cohort yushin --out yushin_trace.jsonl
  python tools/build_bridge_trace_v0.py --cohort keidroid --max 2500 --out keidroid_trace.jsonl
  python tools/build_bridge_trace_v0.py --cohort old_exact --max 2500 --out old_exact_trace_sample.jsonl
"""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated" / "starmie_bridge_trace_v0"
V2 = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v2.jsonl"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, CARD, YES, NO = 7, 8, 9, 10, 11, 12, 13, 14, 3, 1, 2
ROLE = {1031: "Mega", 666: "Cinderace", 1030: "Staryu", 17: "Ignition", 3: "Water"}
_G, _EPC = {}, {}


@contextlib.contextmanager
def _quiet():
    old = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2); yield
    finally:
        os.dup2(old, 2); os.close(dn); os.close(old)


def _winit(budget):
    os.environ["STARMIE_DISABLE"] = "R15"   # deployed baseline (sub_starmie2) behaviour
    sys.path.insert(0, str(ROOT / "agent"))
    with _quiet(), contextlib.redirect_stdout(io.StringIO()):
        import deck_policy_v3 as DP, starmie_heuristics as SH, starmie_tactical_state as TS, search_v3 as S
        S.USE_DYNAMIC_ATTACKS = True
        try: S.DEFAULT_BUDGET = budget
        except Exception: pass
    _G.update(DP=DP, SH=SH, TS=TS)


def _obs(e, step, seat):
    if e not in _EPC:
        if len(_EPC) > 16:
            _EPC.clear()
        try:
            _EPC[e] = json.load(open(REPLAYS / f"{e}.json", encoding="utf-8"))
        except Exception:
            _EPC[e] = None
    try:
        return _EPC[e]["steps"][step][seat].get("observation")
    except Exception:
        return None


def _name(cid):
    return (_G["SH"].CDB.get(str(cid), {}) or {}).get("n", f"#{cid}") if cid is not None else None


def _semkey(o, obs):
    """Canonical semantic action key for a RAW obs option (joins to the proposer's semantic outputs)."""
    DP, SH = _G["DP"], _G["SH"]
    t = o.get("type")
    if t == ATTACK:
        aid = o.get("attackId")
        nm = {SH.JETTING_BLOW: "Jetting", SH.NEBULA_BEAM: "Nebula", SH.TURBO_FLARE: "Turbo"}.get(aid)
        return f"ATTACK:{nm or aid}"
    if t in (PLAY, ATTACH, EVOLVE, ABILITY, CARD):
        try: cid = DP.option_card_id(o, obs)
        except Exception: cid = None
        fam = {PLAY: "PLAY", ATTACH: "ATTACH", EVOLVE: "EVOLVE", ABILITY: "ABILITY", CARD: "SELECT_CARD"}[t]
        if t == ATTACH:
            try: tgt = DP.option_target_entity(o, obs)
            except Exception: tgt = None
            trole = ROLE.get(DP._cid(tgt)) if tgt else "?"
            return f"ATTACH:{ROLE.get(cid, _name(cid))}:{trole}"
        return f"{fam}:{ROLE.get(cid) or _name(cid)}"
    return {RETREAT: "RETREAT", END: "END", YES: "YES", NO: "NO", DISCARD: "DISCARD"}.get(t, str(t))


def _tactical(obs):
    feats = _G["TS"].extract(obs)
    co = feats.get("tactical_coordinates", {})
    return {
        "board": feats.get("board_features", {}),
        "RACE_STATE": co.get("RACE_STATE", {}), "SWEEP_PRESSURE": co.get("SWEEP_PRESSURE", {}),
        "WALL_PRESSURE": co.get("WALL_PRESSURE", {}), "VALUE_STATE": co.get("VALUE_STATE", {}),
        "COMMITMENT_STATE": co.get("COMMITMENT_STATE", {}),
        "entities": feats.get("entity_features", []),
    }


def _choice_meta(o, obs):
    """attach_target_role / attack_name / play_card / select_card for a chosen option."""
    DP = _G["DP"]; t = o.get("type")
    m = {"attach_target_role": None, "attack_name": None, "play_card_name": None, "select_card_name": None}
    if t == ATTACK:
        m["attack_name"] = _semkey(o, obs).split(":", 1)[-1]
    elif t == ATTACH:
        try: tgt = DP.option_target_entity(o, obs)
        except Exception: tgt = None
        m["attach_target_role"] = ROLE.get(DP._cid(tgt)) if tgt else None
    elif t == PLAY:
        m["play_card_name"] = _name(DP.option_card_id(o, obs))
    elif t == CARD:
        m["select_card_name"] = _name(DP.option_card_id(o, obs))
    return m


def run_chunk(task):
    rows, _budget = task
    DP, SH, TS = _G["DP"], _G["SH"], _G["TS"]
    out = []
    for r in rows:
        obs = _obs(r["episode"], r["step"], r["seat"])
        if not obs:
            continue
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if len(opts) < 2:
            continue
        try:
            agent = SH.agent(obs)
        except Exception:
            agent = None
        try:
            src = "heuristic" if SH.choose(obs) is not None else "search_or_default"
        except Exception:
            src = "unknown"
        pilot = r["pilot_action"]
        ai = agent[0] if agent and 0 <= agent[0] < len(opts) else None
        pi = pilot[0] if pilot and 0 <= pilot[0] < len(opts) else None
        idx2key = {i: _semkey(o, obs) for i, o in enumerate(opts)}
        agent_key = idx2key.get(ai)
        pilot_key = idx2key.get(pi)
        cur = obs.get("current") or {}
        ohash = hashlib.sha1(json.dumps(cur, sort_keys=True, default=str).encode()).hexdigest()[:16]
        runtime = {
            "observation_hash": ohash,
            "n_legal": len(opts),
            "option_index_to_semantic_key": idx2key,
            "legal_semantic_keys": sorted(set(idx2key.values())),
            "grouped_legal_actions": r["legal_actions"],
            "current_agent_action": agent, "current_agent_action_key": agent_key,
            "current_agent_action_family": r["family"] if ai == pi else (_semkey(opts[ai], obs).split(":")[0] if ai is not None else None),
            "active_source": src,
            "search_action": None, "search_scores": None, "search_candidate_order": None,  # available via TRACE option_evals (off by default)
            "eval_leaf_terms": None,
            "tactical_state": _tactical(obs),
            "agent_choice_meta": _choice_meta(opts[ai], obs) if ai is not None else {},
            "support_status": "SUPPORTED" if r.get("deck_distance") == 0 else "NEAR" if (r.get("deck_distance") or 9) <= 2 else "OOD",
            "hard_safety_flags": [],
        }
        eval_meta = {
            "decision_id": r["decision_id"], "episode_id": r["episode"], "step": r["step"], "seat": r["seat"],
            "split": r.get("split"), "cohort": r.get("cohort"), "deck_distance": r.get("deck_distance"),
            "pilot_action": pilot, "pilot_action_key": pilot_key, "pilot_action_family": r["family"],
            "pilot_name": r.get("pilot"), "outcome_won": r.get("won"),
            "future_same_turn_sequence": r.get("same_turn_sequence"),
            "agreement": (sorted(agent) == sorted(pilot)) if (agent and pilot) else False,
            "pilot_choice_meta": _choice_meta(opts[pi], obs) if pi is not None else {},
            "replay_link": f"https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?episodeId={r['episode']}",
        }
        out.append({"decision_id": r["decision_id"], "runtime": runtime, "eval_meta": eval_meta,
                    "schema_version": "starmie_bridge_trace_v0"})
    return out


def _select_rows(cohort, maxn):
    rows = []
    for line in open(V2, encoding="utf-8"):
        r = json.loads(line)
        m = r.get("meta") or {}
        if cohort == "yushin":
            if not r.get("new_top1_zero_shot"):
                continue
        elif cohort == "keidroid":
            if m.get("pilot") != "keidroid" or r.get("deck_distance") != 0 or r.get("is_opponent"):
                continue
        elif cohort == "old_exact":
            if r.get("cohort") != "C0" or r.get("is_opponent") or m.get("pilot") in ("keidroid",):
                continue
        rows.append({"decision_id": r["id"], "episode": r["episode"], "step": r["step"], "seat": r["seat"],
                     "family": r["family"], "legal_actions": r["legal_actions"], "pilot_action": r["pilot_action"],
                     "split": r.get("split"), "cohort": r.get("cohort"), "deck_distance": r.get("deck_distance"),
                     "pilot": m.get("pilot"), "won": m.get("won"),
                     "same_turn_sequence": m.get("same_turn_family_sequence")})
        if maxn and len(rows) >= maxn:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["yushin", "keidroid", "old_exact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = _select_rows(a.cohort, a.max)
    print(f"cohort={a.cohort} decisions={len(rows)} budget={a.budget}", flush=True)
    chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
    res, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(a.budget,)) as ex:
        for f in as_completed([ex.submit(run_chunk, (c, a.budget)) for c in chunks]):
            done += 1; res.extend(f.result())
            if done % 15 == 0:
                print(f"  {done}/{len(chunks)} ({len(res)} rows)", flush=True)
    outp = OUT / a.out
    with open(outp, "w", encoding="utf-8") as o:
        for r in res:
            o.write(json.dumps(r, default=str) + "\n")
    agree = sum(1 for r in res if r["eval_meta"]["agreement"])
    print(f"agreement: {agree}/{len(res)} = {round(100*agree/max(1,len(res)),1)}% | wrote {outp} ({len(res)} rows)")


if __name__ == "__main__":
    main()
