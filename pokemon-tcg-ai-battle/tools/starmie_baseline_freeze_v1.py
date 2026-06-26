"""STARMIE TACTICAL-LEAF V1 -- Section 0: FREEZE AND VERIFY THE BASELINE.

Records the accepted Starmie baseline (commit, deck hash, heuristic profile, search config, eval weights,
opponent prior, submission/corpus snapshots) and verifies the accepted correctness fixes are present. Fails
loudly (nonzero exit + "verdict":"MISMATCH") if any check fails.

Read-only except it writes data/generated/starmie_tactical_leaf_v1/baseline_manifest.json.

  python tools/starmie_baseline_freeze_v1.py
"""
from __future__ import annotations
import hashlib, inspect, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent"
OUT = ROOT / "data" / "generated" / "starmie_tactical_leaf_v1"
OUT.mkdir(parents=True, exist_ok=True)


def _sha256(p: Path, limit: int | None = None) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT)] + list(args), text=True).strip()
    except Exception as e:
        return f"<git error: {e}>"


def _src_has(fn, *markers) -> bool:
    """True if every marker string appears in fn's source (pragmatic code-marker verification)."""
    try:
        s = inspect.getsource(fn)
    except Exception:
        return False
    return all(m in s for m in markers)


def main():
    sys.path.insert(0, str(AGENT))
    import deck_policy_v3 as DP  # noqa
    import search_v3 as S
    import eval as EV
    import starmie_heuristics as SH
    import main as M

    checks = {}   # name -> (ok: bool, detail)

    # ---- deck ----
    deck = list(SH.STARMIE_DECK)
    deck_hash = hashlib.sha1(json.dumps(deck).encode()).hexdigest()
    deck_main = list(M.STARMIE_DECK)
    checks["deck_matches_main_agent"] = (deck == deck_main, f"heuristics deck == main.STARMIE_DECK: {deck == deck_main}")

    # ---- correctness fixes (accepted baseline must contain these) ----
    checks["opp_meta_prior_not_self"] = (
        bool(getattr(SH, "USE_META_OPP_PRIOR", False)) and bool(getattr(SH, "_OPP_DECKS", None)),
        f"USE_META_OPP_PRIOR={getattr(SH,'USE_META_OPP_PRIOR',None)}, n_opp_decks={len(getattr(SH,'_OPP_DECKS',[]) or [])}",
    )
    checks["boss_not_retarget_existing_ko"] = (
        _src_has(SH._high_value_play, "ko_avail", "not ko_avail"),
        "Boss/Wally in _high_value_play gated on `not ko_avail`",
    )
    checks["wally_not_strip_ko_energy"] = (
        _src_has(SH._high_value_play, "WALLYS", "not ko_avail") and _src_has(SH._wally_useful, "MEGA_STARMIE"),
        "Wally gated on `not ko_avail` + _wally_useful KO-range proxy",
    )
    # Ignition = 3 units on an Evolution: behavioural check via _energy_units on a mock Mega.
    try:
        mock_mega = {"id": SH.MEGA_STARMIE, "energyCards": [{"id": SH.IGNITION}]}
        ign_units = SH._energy_units(mock_mega)
        mock_basic = {"id": SH.STARYU, "energyCards": [{"id": SH.IGNITION}]}
        ign_basic = SH._energy_units(mock_basic)
        checks["ignition_three_units_on_evolution"] = (
            ign_units == 3 and ign_basic == 1, f"Ignition on Mega={ign_units} (want 3), on Staryu={ign_basic} (want 1)",
        )
    except Exception as e:
        checks["ignition_three_units_on_evolution"] = (False, f"error: {e!r}")
    checks["tutor_respects_source_zone"] = (
        _src_has(SH._sel_card, "_AREA_ZONE", "area"), "_sel_card resolves via option.area / _AREA_ZONE",
    )
    checks["no_suicide_only_self_removing"] = (
        _src_has(SH._no_suicide, "shuffle_self") or _src_has(SH._no_suicide, "self_to_deck"),
        "_no_suicide only blocks self-removing abilities",
    )
    checks["develop_before_attack_enabled"] = (
        _src_has(SH._main_action, "_develop_bench", "_best_ko_index") and SH._on("R2") and SH._on("R3") and SH._on("R9"),
        "free development precedes the KO floor in _main_action; R2/R3/R9 enabled by default",
    )

    # ---- search / eval / prior config ----
    eval_weights = {k: getattr(EV, k) for k in dir(EV) if k.startswith("W_")}
    eval_weights.update({k: getattr(EV, k) for k in ("DECKOUT_TEST", "PH_TEST", "WIN", "LOSS") if hasattr(EV, k)})
    search_cfg = {
        "DEFAULT_BUDGET": getattr(S, "DEFAULT_BUDGET", None),
        "N_DETERM": getattr(S, "N_DETERM", None),
        "USE_DYNAMIC_ATTACKS_default": getattr(S, "USE_DYNAMIC_ATTACKS", None),
        "leaf_mode": "deckout", "rollout_mode": "develop", "require_complete_world": True,
        "agent_call": "choose_action -> choose() heuristics, else best_option(leaf_mode='deckout', rollout_mode='develop') + veto(R13), else default",
    }

    opp_meta_path = AGENT / "opponent_meta_v1.json"
    opp_prior = {
        "file": "opponent_meta_v1.json",
        "sha256": _sha256(opp_meta_path),
        "use_meta_opp_prior": bool(getattr(SH, "USE_META_OPP_PRIOR", False)),
        "n_opp_decks": len(getattr(SH, "_OPP_DECKS", []) or []),
    }

    # ---- artifact snapshots ----
    sub2 = ROOT / "submissions" / "sub_starmie2.tar.gz"
    corpus = ROOT / "data" / "starmie_corpus" / "starmie_specialist_corpus_v1.jsonl"
    corpus_manifest = ROOT / "data" / "starmie_audit" / "corpus_manifest.json"
    corpus_rows = None
    if corpus.exists():
        try:
            with open(corpus, "rb") as f:
                corpus_rows = sum(1 for _ in f)
        except Exception:
            corpus_rows = None

    # sub_starmie2 source commit: the bundled starmie_heuristics matches which commit?
    sub2_heur = ROOT / "submissions" / "sub_starmie2" / "starmie_heuristics.py"
    sub2_heur_sha = _sha256(sub2_heur)

    head = _git("rev-parse", "HEAD")
    head_short = _git("rev-parse", "--short", "HEAD")
    fixes_commit_in_head = _git("merge-base", "--is-ancestor", "359f9ae", "HEAD")  # "" on success
    head_includes_fixes = (subprocess.call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "359f9ae", "HEAD"]) == 0)
    checks["head_includes_accepted_fix_commits"] = (head_includes_fixes, "359f9ae (heuristic fixes) is an ancestor of HEAD")

    all_ok = all(ok for ok, _ in checks.values())

    manifest = {
        "verdict": "BASELINE_FROZEN" if all_ok else "MISMATCH",
        "generated_by": "tools/starmie_baseline_freeze_v1.py",
        "git": {
            "head_commit": head, "head_short": head_short, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head_includes_359f9ae_fixes": head_includes_fixes,
            "note": "ATTACK_FLOOR candidate (uncommitted) is default-OFF; baseline behaviour = HEAD default config.",
        },
        "deck": {"ids": deck, "n_cards": len(deck), "sha1": deck_hash},
        "heuristic_profile": {
            "rules": [r.__name__ for r in SH.RULES],
            "disabled_default": sorted(SH.DISABLED),
            "attack_floor_default": bool(getattr(SH, "ATTACK_FLOOR", False)),
            "veto_enabled_R13": SH._on("R13"),
        },
        "search_config": search_cfg,
        "eval_weights": eval_weights,
        "opponent_prior": opp_prior,
        "submission": {
            "sub_starmie2_tar_sha256": _sha256(sub2),
            "sub_starmie2_bytes": sub2.stat().st_size if sub2.exists() else None,
            "sub_starmie2_heuristics_sha256": sub2_heur_sha,
            "source_commit_note": "sub_starmie2 rebuilt from main 359f9ae (per session log)",
        },
        "replay_corpus_snapshot": {
            "specialist_corpus": "data/starmie_corpus/starmie_specialist_corpus_v1.jsonl (gitignored)",
            "rows": corpus_rows,
            "sha256_first_1MB": _sha256(corpus, limit=1 << 20),
            "corpus_manifest": "data/starmie_audit/corpus_manifest.json" if corpus_manifest.exists() else None,
        },
        "file_hashes": {
            "agent/eval.py": _sha256(AGENT / "eval.py"),
            "agent/search_v3.py": _sha256(AGENT / "search_v3.py"),
            "agent/deck_policy_v3.py": _sha256(AGENT / "deck_policy_v3.py"),
            "agent/starmie_heuristics.py": _sha256(AGENT / "starmie_heuristics.py"),
            "agent/main.py": _sha256(AGENT / "main.py"),
        },
        "correctness_checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
    }
    out = OUT / "baseline_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"verdict: {manifest['verdict']}  (HEAD {head_short})")
    for k, (ok, d) in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {k}: {d}")
    print(f"\nwrote {out}")
    if not all_ok:
        print("\nBASELINE MISMATCH -- fix before proceeding.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
