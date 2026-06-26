"""STARMIE CORPUS V2 -- Section 4: prepare the new-#1-pilot ZERO-SHOT eval set (do NOT train).

Extracts the held-out new-top-1 (Yushin Ito) decisions from V2 into a self-contained eval set and writes a
manifest with family counts + the exact command Model A should run against its frozen Starmie teacher. Model B
cannot run the teacher (it lives in Model A's .codex worktree), so this only PREPARES the eval.

  python tools/build_starmie_v2_zeroshot.py
"""
from __future__ import annotations
import collections, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v2.jsonl"
OUT = ROOT / "data" / "starmie_corpus" / "starmie_v2_zero_shot_newtop1.jsonl"
MAN = ROOT / "data" / "starmie_audit" / "starmie_v2_zero_shot_eval.json"

CMD = [
    "# Run the FROZEN Starmie teacher (do NOT retrain) zero-shot on this held-out new-#1-pilot set.",
    "# Group candidates by legal_actions; label = pilot_action_label; report top-1/3/5, MRR, NLL, option-zero",
    "# rate, predicted option-zero rate, nonzero-label top-1/3, and a by-family breakdown",
    "# (PLAY / SELECT_CARD / ATTACH / EVOLVE / ATTACK / RETREAT / ABILITY / END).",
    "python tools/eval_starmie_teacher.py --teacher <frozen_starmie_teacher_ckpt>"
    " --eval-jsonl <KF>/pokemon-tcg-ai-battle/data/starmie_corpus/starmie_v2_zero_shot_newtop1.jsonl"
    " --replays <PAA>/data/external/replays"
    " --group-by legal_actions --label pilot_action_label --by-family"
    " --report data/generated/starmie_v2_newtop1_zeroshot_metrics.json",
    "# <KF>  = C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun",
    "# <PAA> = C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent",
]


def main():
    fams = collections.Counter()
    splits = collections.Counter()
    n = choice = 0
    with open(OUT, "w", encoding="utf-8") as o:
        for line in open(V2, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("new_top1_zero_shot"):
                continue
            n += 1
            fams[r["family"]] += 1
            splits[r["split"]] += 1
            if r["n_legal"] >= 2:
                choice += 1
            o.write(json.dumps({
                "decision_id": r["id"], "episode": r["episode"], "step": r["step"], "seat": r["seat"],
                "root_obs_pointer": r["root_obs_pointer"], "legal_actions": r["legal_actions"],
                "n_legal": r["n_legal"], "pilot_action_label": r["pilot_action"],
                "family": r["family"], "family_detail": r["family_detail"], "context": r["context"],
            }) + "\n")

    manifest = {
        "eval_set": "starmie_v2_zero_shot_newtop1 (Yushin Ito, leaderboard #1, exact deck) -- HELD OUT; the teacher trained on V1 and these 65 episodes are not in V1",
        "file": "data/starmie_corpus/starmie_v2_zero_shot_newtop1.jsonl",
        "n_decisions": n, "decisions_with_real_choice_n_legal_ge_2": choice,
        "by_family": dict(fams), "split_tags_ignore_for_zeroshot_use_all": dict(splits),
        "label_field": "pilot_action_label (indices into legal_actions)",
        "feature_resolution": "resolve root_obs_pointer -> steps[step][seat].observation in pokemon-ai-agent/data/external/replays",
        "forbidden_as_features": ["episode/decision_id", "pilot identity", "outcome", "split"],
        "command_for_model_a": CMD,
        "note": "Model B cannot run this: the frozen Starmie teacher + its feature pipeline live in Model A's .codex worktree (off-limits). Eval set is prepared and ready.",
    }
    MAN.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"zero-shot eval set: {n} decisions | {choice} with real choice (n_legal>=2)")
    print(f"by family: {dict(fams)}")
    print(f"wrote {OUT}\nwrote {MAN}")


if __name__ == "__main__":
    main()
