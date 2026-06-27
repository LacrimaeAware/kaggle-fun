"""Natural Heuristic Rule Reconstruction -- PHASE 1: trace generation (Model B, READ-ONLY diagnostic).

Runs the known rule agent pi_R (submissions/sub_archaludon) in NATURAL games vs a small opponent mix and dumps,
per decision, the raw public obs + pi_R's chosen action + option-zero. We do NOT extract features or change any
gameplay here -- Phase 2 (heuristic_rule_recon_v0.py) consumes natural_trace_raw.jsonl in the worktree env to
avoid the sub_archaludon `cg` import colliding with the worktree's engine.

Opponents: random + first (kaggle builtins; compatible, no module conflict). The Starmie baseline is NOT used as
an opponent here to keep pi_R's bundled `cg` isolated; the diagnostic needs pi_R's natural decisions, which these
opponents elicit across every decision type. Natural states only (no artificial/controlled states).

  PYTHONIOENCODING=utf-8 python tools/heuristic_trace_gen_v0.py --games 20
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

WT = Path(__file__).resolve().parent.parent                      # worktree repo root
ARCH_DIR = WT.parent.parent.parent / "pokemon-tcg-ai-battle" / "submissions" / "sub_archaludon"  # main checkout
if not ARCH_DIR.exists():
    ARCH_DIR = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/pokemon-tcg-ai-battle/submissions/sub_archaludon")
OUT = WT / "data" / "generated" / "heuristic_rule_reconstruction_v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-games", type=int, default=60)
    args = ap.parse_args()
    n = min(args.games, args.max_games)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (ARCH_DIR / "main.py").exists():
        (OUT / "trace_manifest.json").write_text(json.dumps({"VERDICT": "WAITING_FOR_NOTEBOOK",
            "reason": f"rule agent not found at {ARCH_DIR}"}, indent=2), encoding="utf-8")
        print("WAITING_FOR_NOTEBOOK")
        return 0

    os.chdir(ARCH_DIR)                       # pi_R reads deck.csv from CWD
    sys.path.insert(0, str(ARCH_DIR))        # for bundled cg + main
    records = []
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        import main as ARCH
        from kaggle_environments import make

        def make_recorder(game_id, opp_name, pi_seat):
            def rec(obs):
                act = ARCH.agent(obs)
                sel = obs.get("select")
                if sel and (sel.get("option") or []):
                    k = int(sel.get("maxCount") or 1)
                    records.append({
                        "game_id": game_id, "opponent": opp_name, "seat": pi_seat,
                        "step": int(obs.get("step") or len(records)),
                        "maxCount": k, "n_options": len(sel.get("option") or []),
                        "pi_rule_action": list(act) if isinstance(act, list) else [act],
                        "pi_zero_action": list(range(min(k, len(sel.get("option") or [])))),
                        "obs": json.loads(json.dumps(obs)),     # deep copy of the public obs
                    })
                return act
            return rec

        opps = ["random", "first"]
        gi = 0
        for g in range(n):
            opp = opps[g % len(opps)]
            pi_seat = g % 2                                   # alternate seats
            rec = make_recorder(f"g{gi}", opp, pi_seat)
            pair = [rec, opp] if pi_seat == 0 else [opp, rec]
            with contextlib.suppress(Exception):
                env = make("cabt")
                env.run(pair)
            gi += 1

    with open(OUT / "natural_trace_raw.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    manifest = {"pi_R": "submissions/sub_archaludon (Archaludon ex rule agent)", "games": n,
                "opponents": opps, "decisions_captured": len(records),
                "note": "natural games only; raw public obs captured; features computed in Phase 2 (worktree env). "
                        "Starmie baseline not used as opponent to isolate pi_R's bundled cg."}
    (OUT / "trace_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"games": n, "decisions_captured": len(records),
                      "out": str(OUT / "natural_trace_raw.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
