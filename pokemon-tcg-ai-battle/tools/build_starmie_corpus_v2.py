"""STARMIE SPECIALIST CORPUS V2 -- add the new #1 public pilot (Yushin Ito, leaderboard Elo 1397.2, submission
54038721) as a SEPARATE cohort WITHOUT overwriting V1.

V2 = all V1 rows (verbatim) + Yushin Ito's decisions from the 65 newly fetched games (submission 54038721), tagged
cohort="C_NEW_TOP1". Those 65 episodes are NOT in V1 (the V1 corpus was built before this fetch), so the new
rows are a clean HELD-OUT zero-shot set for the existing teacher (which trained on V1). Yushin Ito's deck is an
EXACT match to our Starmie deck (deck_distance 0, zero card diffs), so the cohort is exact.

Outputs (V1 untouched):
  data/starmie_corpus/starmie_specialist_corpus_v2.jsonl    (V1 rows + new C_NEW_TOP1 rows; gitignored, large)
  data/starmie_audit/starmie_corpus_manifest_v2.json        (references V1; new-cohort stats; split-view defs)

  python tools/build_starmie_corpus_v2.py
"""
from __future__ import annotations
import collections, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tools"))
import build_starmie_corpus_v1 as B  # noqa: E402  (helpers: _deck_of, _deck_distance, _seat_rows, _winner, _split)

REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
FETCH_REPORT = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/generated/last_replay_fetch.json")
V1_JSONL = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v1.jsonl"
V2_JSONL = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v2.jsonl"
V2_MANIFEST = ROOT / "data" / "starmie_audit" / "starmie_corpus_manifest_v2.json"

NEW_PILOT = "Yushin Ito"
NEW_SUBMISSION = 54038721
NEW_PILOT_RANK = {"source": "kaggle competition_leaderboard_view (frozen)", "rank": 1, "elo_like_score": 1397.2}


def _new_episode_ids():
    rpt = json.loads(FETCH_REPORT.read_text(encoding="utf-8"))
    return [r["episode_id"] for r in rpt.get("results", []) if r.get("status") == "fetched"]


def main():
    if not V1_JSONL.exists():
        raise SystemExit(f"V1 corpus missing: {V1_JSONL}")
    V2_JSONL.parent.mkdir(parents=True, exist_ok=True)
    V2_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # ---- build new-pilot rows from the 65 fetched episodes (Yushin Ito's seat only) ----
    new_rows = []
    new_dists = collections.Counter()
    new_fams = collections.Counter()
    new_wl = collections.Counter()
    new_decks = []
    n_games = 0
    for epid in _new_episode_ids():
        p = REPLAYS / f"{epid}.json"
        if not p.exists():
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        tn = (d.get("info") or {}).get("TeamNames") or []
        seat = next((s for s in (0, 1) if s < len(tn) and str(tn[s]) == NEW_PILOT), None)
        if seat is None:
            continue
        dk = B._deck_of(d, seat)
        if not dk or B.MEGA not in dk:
            continue
        dist = B._deck_distance(dk)
        new_dists[dist] += 1
        new_decks.append(tuple(sorted(dk)))
        w = B._winner(d)
        won = (seat == w)
        new_wl["win" if won else "loss"] += 1
        n_games += 1
        cohort = "C_NEW_TOP1_EXACT" if dist == 0 else ("C_NEW_TOP1_NEAR1" if dist == 1 else
                 "C_NEW_TOP1_NEAR2PLUS" if dist <= 10 else "OUT_OF_SCOPE")
        if cohort == "OUT_OF_SCOPE":
            continue
        rows = B._seat_rows(d, seat, cohort, dist, NEW_PILOT, won)
        for r in rows:
            r["source"] = "new_top1_fetch_v2"          # provenance (NOT a model input)
            r["new_top1_zero_shot"] = True             # held out from the V1-trained teacher
            new_fams[r["family"]] += 1
            new_rows.append(r)

    # ---- write V2 = V1 (verbatim) + new rows ----
    v1_count = 0
    v1_cohorts = collections.Counter()
    v1_yushin_exact = 0
    with open(V2_JSONL, "w", encoding="utf-8") as out:
        for line in open(V1_JSONL, encoding="utf-8"):
            out.write(line)
            v1_count += 1
            try:
                r = json.loads(line)
                v1_cohorts[r.get("cohort")] += 1
                if (r.get("meta") or {}).get("pilot") == NEW_PILOT and r.get("deck_distance") == 0 and not r.get("is_opponent"):
                    v1_yushin_exact += 1
            except Exception:
                pass
        for r in new_rows:
            out.write(json.dumps(r) + "\n")

    new_split = collections.Counter(r["split"] for r in new_rows)
    modal_deck = collections.Counter(new_decks).most_common(1)[0][0] if new_decks else None
    deck_diff = {}
    if modal_deck:
        a, b = collections.Counter(modal_deck), collections.Counter(B.EXACT)
        deck_diff = {str(c): a[c] - b[c] for c in set(a) | set(b) if a[c] != b[c]}

    manifest = {
        "corpus_version": "starmie_specialist_corpus_v2",
        "built_on": "V1 (preserved verbatim) + new #1 pilot cohort",
        "v1_reference": {"jsonl": "data/starmie_corpus/starmie_specialist_corpus_v1.jsonl",
                         "manifest": "data/starmie_audit/starmie_corpus_manifest_v1.json",
                         "rows": v1_count, "cohorts": dict(v1_cohorts),
                         "yushin_ito_exact_rows_already_in_v1": v1_yushin_exact},
        "new_top1_pilot": {
            "pilot": NEW_PILOT, "submission_id": NEW_SUBMISSION, "rank_source": NEW_PILOT_RANK,
            "new_games": n_games, "win_loss": dict(new_wl),
            "deck_distance_distribution": dict(new_dists),
            "deck_exact_match": deck_diff == {} and set(new_dists) == {0},
            "card_diffs_vs_our_deck": deck_diff,
            "new_decisions": len(new_rows), "action_families": dict(new_fams),
            "split_distribution_new_rows": dict(new_split),
            "cohort": "C_NEW_TOP1_EXACT",
        },
        "exact_deck_sha1": hashlib.sha1(str(B.EXACT).encode()).hexdigest()[:12],
        "rows_total_v2": v1_count + len(new_rows),
        "split_views": {
            "A_NEW_TOP1_EXACT_SINGLE_PILOT": {
                "desc": "Yushin Ito exact-deck rows (V1-origin + new), partitioned by each row's `split` (train/val/test). For training a single-pilot specialist.",
                "filter": "meta.pilot=='Yushin Ito' AND deck_distance==0 AND is_opponent==false",
                "split_field": "split"},
            "B_OLD_STARMIE_EXACT_ALL_PILOTS": {
                "desc": "V1 exact-deck cohort (all pilots), preserved unchanged.",
                "filter": "cohort=='C0'", "split_field": "split"},
            "C_OLD_PLUS_NEW_EXACT": {
                "desc": "All exact-deck rows: V1 C0 + new C_NEW_TOP1_EXACT.",
                "filter": "cohort=='C0' OR cohort=='C_NEW_TOP1_EXACT'", "split_field": "split"},
            "D_NEW_TOP1_ZERO_SHOT_EVAL_ONLY": {
                "desc": "Yushin Ito decisions from the 65 NEW games only (not in V1 -> teacher never trained on them). EVAL ONLY, use all rows regardless of split.",
                "filter": "new_top1_zero_shot==true", "split_field": None, "n_rows": len(new_rows)},
        },
        "underpowered": len(new_rows) < 200,
        "notes": ("Forbidden as model inputs (metadata only): meta.pilot, episode/replay id, meta.won (outcome), "
                  "split, meta.same_turn_family_sequence, source, new_top1_zero_shot. V1 rows are byte-identical "
                  "to V1; only new rows are appended. Replay-grouped splits via sha1(episode)%100; the 65 new "
                  "episodes are disjoint from V1 episodes, so no replay appears in two splits."),
    }
    V2_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"V1 rows preserved: {v1_count}")
    print(f"new {NEW_PILOT} rows added: {len(new_rows)} from {n_games} games (W/L {dict(new_wl)})")
    print(f"deck_distance dist (new): {dict(new_dists)} | exact match: {manifest['new_top1_pilot']['deck_exact_match']}")
    print(f"new-row families: {dict(new_fams)}")
    print(f"new-row split dist: {dict(new_split)}")
    print(f"Yushin Ito exact rows already in V1: {v1_yushin_exact}")
    print(f"V2 total rows: {v1_count + len(new_rows)}")
    print(f"wrote {V2_JSONL}\nwrote {V2_MANIFEST}")


if __name__ == "__main__":
    main()
