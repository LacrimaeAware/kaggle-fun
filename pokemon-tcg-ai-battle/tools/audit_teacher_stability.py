"""Branch A / A2 -- Teacher V1 stability audit.

Measures how stable Teacher V1's verdict is when the SAME decision is queried repeatedly, and -- the
key contribution -- DECOMPOSES the instability into its two sources:

  * cross-seed queries  (different determinization seeds): determinization draw + engine rollout RNG
  * same-seed repeats   (one fixed determinization):        engine rollout RNG ONLY (coins/shuffles,
                                                            not Python-seedable; ~93% of decisions, per
                                                            the SPLIT_BASE_V2 finding)

The gap tells us how much teacher wobble is FIXABLE by sampling harder / shared worlds (A3) versus
baked into the engine. It records, per decision, everything Branch B needs to weight noisy labels:
repeated-query top-action stability, value variance, completed determinizations, top-two margin,
acceptable-action sets, averaged soft-policy + advantage targets, and a confidence weight. It does NOT
build Teacher V2 or rerun search sweeps.

Reproducible: samples from a FROZEN snapshot and verifies each replay's sha256 against the manifest.

    python tools/audit_teacher_stability.py --n 8                 # throughput smoke
    python tools/audit_teacher_stability.py --n 150 --seeds 16 --engine-repeats 8 --out teacher_v1_stability_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import teacher_api_v1 as T          # noqa: E402
import state_action_schema_v2 as SCH  # noqa: E402
import main as M                    # noqa: E402

MANIFEST_DIR = ROOT / "data" / "manifests"
SPLIT_DIR = ROOT / "data" / "splits"
REPLAY_DIR = ROOT / "data" / "external" / "replays"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _deciding_player_deck(d: dict, player_idx: int):
    for s in d.get("steps", []):
        if player_idx < len(s) and isinstance(s[player_idx], dict):
            a = s[player_idx].get("action")
            if isinstance(a, list) and len(a) == 60:
                return a
    return None


def sample_decisions(manifest: dict, split: dict, n: int, verify: bool) -> list:
    """Non-forced single-pick decisions from the FROZEN train games, deck-stratified. Verifies each
    sampled file's sha256 against the manifest (so the audit truly consumes the immutable snapshot)."""
    sha_by_file = {r["file"]: r["sha256"] for r in manifest["files"] if r.get("sha256")}
    per_deck_cap = max(2, n // 8)
    decisions, by_deck, mism = [], defaultdict(int), 0
    for fn in split["train"]:
        if len(decisions) >= n:
            break
        fp = REPLAY_DIR / fn
        if not fp.exists():
            continue
        if verify and sha_by_file.get(fn) and _sha256(str(fp)) != sha_by_file[fn]:
            mism += 1
            continue
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for si, s in enumerate(d.get("steps", [])):
            if len(decisions) >= n:
                break
            for ai, ag in enumerate(s):
                if not isinstance(ag, dict):
                    continue
                obs = ag.get("observation") or {}
                if not SCH.is_single_pick_decision(obs):
                    continue
                cur = obs.get("current") or {}
                if not cur.get("players"):
                    continue
                me = cur.get("yourIndex", ai)
                opts = obs["select"]["option"]
                if len(set(SCH.equivalence_classes(opts, cur, me))) < 2:
                    continue
                try:
                    if M._forced_move(obs) is not None:        # non-forced only (forced is deterministic)
                        continue
                except Exception:
                    pass
                deck = _deciding_player_deck(d, me)
                if not deck:
                    continue
                sig = tuple(sorted(deck))
                if by_deck[sig] >= per_deck_cap:
                    continue
                by_deck[sig] += 1
                decisions.append({"file": fn, "step": si, "player": ai, "obs": obs, "deck": deck,
                                  "turn": cur.get("turn"), "n_eq": len(set(SCH.equivalence_classes(opts, cur, me))),
                                  "types": sorted(t for t in {o.get("type") for o in opts if isinstance(o, dict)}
                                                  if t in SCH.MAJOR_ACTION_TYPES)})
                break
    if mism:
        print(f"  WARNING: {mism} files failed hash verification (corpus changed under the snapshot)", flush=True)
    return decisions


def _top_class(rs):
    cls = [r["argmax_eq_class"] for r in rs]
    if not cls:
        return None, None
    modal, cnt = Counter(cls).most_common(1)[0]
    return cnt / len(cls), modal


def audit_decision(dec, seeds, engine_repeats, n_determ, budget):
    obs, deck = dec["obs"], dec["deck"]
    cross = [r for r in (T.query(obs, deck, n_determ=n_determ, time_budget=budget, leaf_mode="hand", seed=s)
                         for s in seeds) if r.get("applicable")]
    if not cross:
        return None
    same = [r for r in (T.query(obs, deck, n_determ=n_determ, time_budget=budget, leaf_mode="hand", seed=seeds[0])
                        for _ in range(engine_repeats)) if r.get("applicable")]
    cross_stab, modal = _top_class(cross)
    eng_stab, _ = _top_class(same)
    margins = [r["top_two_margin"] for r in cross if r.get("top_two_margin") is not None]
    accept = [len(r["acceptable_action_set"]) for r in cross]
    # value spread of the modal-argmax class across seeds; within-query variance; completed worlds
    modal_vals = [ec["mean_value"] for r in cross for ec in r["eq_classes"]
                  if ec["eq_class"] == modal and ec["mean_value"] is not None]
    within_var = [ec["value_variance"] for r in cross for ec in r["eq_classes"]
                  if ec.get("value_variance") is not None]
    completed = [ec["completed_determinizations"] for r in cross for ec in r["eq_classes"]]
    # robust targets for B: average soft policy + advantage per eq-class across seeds (indices are
    # consistent across seeds -- same option list). Down-weight by confidence elsewhere.
    soft_acc, adv_acc = defaultdict(list), defaultdict(list)
    for r in cross:
        for ec_idx, p in r["soft_policy_target"].items():
            soft_acc[ec_idx].append(p)
        for ec in r["eq_classes"]:
            if ec.get("mean_value") is not None:
                na = next((o["normalized_advantage"] for o in r["options"]
                           if o["eq_class"] == ec["eq_class"] and o["normalized_advantage"] is not None), None)
                if na is not None:
                    adv_acc[ec["eq_class"]].append(na)
    avg_soft = {str(k): round(statistics.fmean(v), 3) for k, v in soft_acc.items()}
    avg_adv = {str(k): round(statistics.fmean(v), 2) for k, v in adv_acc.items()}
    scls = "stable" if cross_stab >= 0.9 else ("unstable" if cross_stab <= 0.6 else "near_tie")
    return {
        "file": dec["file"], "step": dec["step"], "turn": dec["turn"], "n_eq": dec["n_eq"], "types": dec["types"],
        "cross_seed_top_stability": round(cross_stab, 3),
        "engine_only_top_stability": round(eng_stab, 3) if eng_stab is not None else None,
        "determinization_extra_instability": round(eng_stab - cross_stab, 3) if eng_stab is not None else None,
        "mean_top_two_margin": round(statistics.fmean(margins), 3) if margins else None,
        "mean_acceptable_set_size": round(statistics.fmean(accept), 2) if accept else None,
        "across_seed_modal_value_std": round(statistics.pstdev(modal_vals), 2) if len(modal_vals) > 1 else 0.0,
        "mean_within_value_variance": round(statistics.fmean(within_var), 2) if within_var else None,
        "mean_completed_determinizations": round(statistics.fmean(completed), 2) if completed else None,
        "confidence_weight": round(cross_stab, 3),
        "stability_class": scls,
        "modal_argmax_eq_class": modal,
        "avg_soft_policy": avg_soft,
        "avg_advantage": avg_adv,
        "n_cross": len(cross), "n_same": len(same),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="replays_20260618.json")
    ap.add_argument("--split", default="replays_20260618_split.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--engine-repeats", type=int, default=8)
    ap.add_argument("--n-determ", type=int, default=8)
    ap.add_argument("--budget", type=float, default=8.0)     # large so all worlds finish (offline)
    ap.add_argument("--no-verify-hashes", action="store_true")
    ap.add_argument("--out", default="teacher_v1_stability_smoke")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST_DIR / args.snapshot, encoding="utf-8"))
    split = json.load(open(SPLIT_DIR / args.split, encoding="utf-8"))
    print(f"[A2] frozen snapshot {args.snapshot} (corpus_sha256 {manifest.get('corpus_sha256')}, "
          f"{manifest['n_included']} games). Sampling {args.n} non-forced decisions...", flush=True)
    decisions = sample_decisions(manifest, split, args.n, not args.no_verify_hashes)
    n_decks = len({tuple(sorted(d["deck"])) for d in decisions})
    print(f"[A2] {len(decisions)} decisions across {n_decks} decks; querying "
          f"{args.seeds} cross-seed + {args.engine_repeats} same-seed each (n_determ={args.n_determ})...", flush=True)

    seeds = list(range(1000, 1000 + args.seeds))
    recs, t0, q = [], time.time(), 0
    for i, dec in enumerate(decisions):
        r = audit_decision(dec, seeds, args.engine_repeats, args.n_determ, args.budget)
        q += args.seeds + args.engine_repeats
        if r:
            recs.append(r)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(decisions)} | {q} queries | {el:.0f}s | {q/el:.1f} q/s", flush=True)
    dt = time.time() - t0

    def fmean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.fmean(xs), 3) if xs else None

    by_type = defaultdict(list)
    for r in recs:
        for t in r["types"]:
            by_type[t].append(r["cross_seed_top_stability"])
    agg = {
        "n_decisions": len(recs), "total_queries": q, "seconds": round(dt, 1),
        "queries_per_sec": round(q / dt, 2) if dt else 0,
        "projected_full_1000_hours": round((dt / max(1, len(recs))) * 1000 / 3600, 2) if recs else None,
        "stable_frac": round(sum(r["stability_class"] == "stable" for r in recs) / max(1, len(recs)), 3),
        "near_tie_frac": round(sum(r["stability_class"] == "near_tie" for r in recs) / max(1, len(recs)), 3),
        "unstable_frac": round(sum(r["stability_class"] == "unstable" for r in recs) / max(1, len(recs)), 3),
        "mean_cross_seed_top_stability": fmean([r["cross_seed_top_stability"] for r in recs]),
        "mean_engine_only_top_stability": fmean([r["engine_only_top_stability"] for r in recs]),
        "mean_determinization_extra_instability": fmean([r["determinization_extra_instability"] for r in recs]),
        "mean_top_two_margin": fmean([r["mean_top_two_margin"] for r in recs]),
        "mean_acceptable_set_size": fmean([r["mean_acceptable_set_size"] for r in recs]),
        "stability_by_action_type": {str(t): round(statistics.fmean(v), 3) for t, v in sorted(by_type.items())},
    }
    recpath = MANIFEST_DIR / f"{args.out}.jsonl"
    with open(recpath, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summ = {"audit": "teacher_v1_stability_v1", "snapshot": args.snapshot,
            "config": {"seeds": args.seeds, "engine_repeats": args.engine_repeats,
                       "n_determ": args.n_determ, "budget": args.budget}, "aggregate": agg}
    json.dump(summ, open(MANIFEST_DIR / f"{args.out}_summary.json", "w", encoding="utf-8"), indent=1)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=1))
    print(f"\n[A2] wrote {recpath.name} (+ summary). Small-n directional read, not a final conclusion.", flush=True)


if __name__ == "__main__":
    main()
