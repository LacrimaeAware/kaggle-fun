"""STARMIE TACTICAL-LEAF V1 -- Section 2: EXPORT A TACTICAL-STATE DATASET.

Replays the Starmie specialist corpus and emits one row per supported decision with the public tactical-state
features (agent/starmie_tactical_state.py). The RUNTIME feature payload is strictly separated from
EVALUATION-ONLY metadata (pilot identity, outcome, future same-turn sequence, replay id, split id).

Source of truth for full observations: the raw replay episodes (the corpus carries only a compact root_state +
a root_obs_pointer). Replays are read READ-ONLY from the authorized pokemon-ai-agent mirror; the corpus row's
pilot_action / meta are carried as eval-only metadata.

Output: data/generated/starmie_tactical_leaf_v1/starmie_tactical_state_v1.jsonl

  python tools/starmie_tactical_export_v1.py --max-rows 8000 --budget 0.0
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
CORPUS = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v1.jsonl"
OUT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1" / "starmie_tactical_state_v1.jsonl"
REPLAY_DIRS = [
    ROOT / "data" / "external" / "replays",
    Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays"),
]
# Trivial decision families (coin flips / forced setup) carry little tactical signal -> skipped by default.
TRIVIAL_FAMILIES = {"YES_NO"}


def _episode_path(epid):
    for d in REPLAY_DIRS:
        p = d / f"{epid}.json"
        if p.exists():
            return p
    return None


_EP_CACHE = {}


def _load_episode(epid):
    if epid in _EP_CACHE:
        return _EP_CACHE[epid]
    p = _episode_path(epid)
    ep = None
    if p is not None:
        try:
            ep = json.load(open(p, encoding="utf-8"))
        except Exception:
            ep = None
    # bound cache memory
    if len(_EP_CACHE) > 64:
        _EP_CACHE.clear()
    _EP_CACHE[epid] = ep
    return ep


def _obs_from_pointer(ptr):
    ep = _load_episode(ptr.get("episode"))
    if not ep:
        return None
    steps = ep.get("steps") if isinstance(ep, dict) else ep
    t, seat = ptr.get("step"), ptr.get("seat")
    try:
        return steps[t][seat].get("observation")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--include-trivial", action="store_true")
    ap.add_argument("--budget", type=float, default=0.0, help="search budget for baseline action; 0 = heuristic-only")
    a = ap.parse_args()

    sys.path.insert(0, str(AGENT))
    import deck_policy_v3 as DP  # noqa
    import starmie_heuristics as SH
    import starmie_tactical_state as TS
    if a.budget > 0:
        import search_v3 as S
        S.USE_DYNAMIC_ATTACKS = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_no_obs = n_err = n_trivial = 0
    fam_counts, split_counts, cohort_counts = {}, {}, {}
    with open(CORPUS, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            if n_out >= a.max_rows:
                break
            n_in += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            fam = row.get("family")
            if row.get("is_opponent"):
                continue   # extractor is Starmie-specific; only OUR-seat decisions are meaningful here
            if not a.include_trivial and fam in TRIVIAL_FAMILIES:
                n_trivial += 1
                continue
            obs = _obs_from_pointer(row.get("root_obs_pointer") or {})
            if not obs:
                n_no_obs += 1
                continue
            feats = TS.extract(obs)
            if not feats or "error" in feats:
                n_err += 1
                continue

            # baseline action + source (RUNTIME): heuristic pick, or full agent at a small budget if requested.
            src = "search_or_default"
            base_action = None
            try:
                h = SH.choose(obs)
                if h is not None:
                    src, base_action = "heuristic", h
                elif a.budget > 0:
                    base_action = SH.choose_action(obs)
            except Exception:
                pass
            pilot_action = row.get("pilot_action")
            in_disagree = (base_action is not None and pilot_action is not None and list(base_action) != list(pilot_action))

            obs_hash = hashlib.sha1(json.dumps(obs.get("current") or {}, sort_keys=True, default=str).encode()).hexdigest()[:16]
            out_row = {
                "decision_id": row.get("id"),
                # ---- RUNTIME feature payload (public; no hidden info, no outcome, no pilot, no replay id) ----
                "runtime": {
                    "observable_state_hash": obs_hash,
                    "action_family": fam,
                    "n_options": feats.get("n_options"),
                    "option_types": feats.get("option_types"),
                    "legal_grouped_actions": row.get("legal_actions"),
                    "baseline_action": base_action, "baseline_source": src,
                    "entity_features": feats.get("entity_features"),
                    "board_features": feats.get("board_features"),
                    "tactical_coordinates": feats.get("tactical_coordinates"),
                },
                # ---- EVALUATION-ONLY metadata (must NOT be a runtime input) ----
                "eval_meta": {
                    "replay_id": row.get("episode"), "step": row.get("step"), "seat": row.get("seat"),
                    "split": row.get("split"), "cohort": row.get("cohort"), "deck_distance": row.get("deck_distance"),
                    "is_opponent_seat": row.get("is_opponent"),
                    "pilot_action": pilot_action, "pilot": (row.get("meta") or {}).get("pilot"),
                    "won": (row.get("meta") or {}).get("won"),
                    "same_turn_sequence": (row.get("meta") or {}).get("same_turn_family_sequence"),
                    "in_disagreement_class": in_disagree,
                },
                "schema_version": "starmie_tactical_state_v1",
            }
            fout.write(json.dumps(out_row, default=str) + "\n")
            n_out += 1
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            split_counts[row.get("split")] = split_counts.get(row.get("split"), 0) + 1
            cohort_counts[row.get("cohort")] = cohort_counts.get(row.get("cohort"), 0) + 1
            if n_out % 1000 == 0:
                print(f"  exported {n_out} (scanned {n_in}, no_obs {n_no_obs}, err {n_err})", flush=True)

    summary = {"rows_in_scanned": n_in, "rows_exported": n_out, "skipped_trivial": n_trivial,
               "skipped_no_obs": n_no_obs, "skipped_err": n_err,
               "by_family": fam_counts, "by_split": split_counts, "by_cohort": cohort_counts,
               "output": str(OUT)}
    (OUT.parent / "tactical_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
