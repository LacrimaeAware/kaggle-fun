# KO sequencing + the measurement-noise finding (2026-06-21)

Two results from this session. The second one (KO sequencing) only became findable because of the first.

## 1. Whole-game self-play A/B is too noisy to tune heuristics

Definitive probe: run production `agent_search` against an **exact copy of itself** through the harness.

| probe | n | win rate | 95% CI |
|---|---|---|---|
| `selfbase` (production vs itself) | 120 | 0.567 | [0.477, 0.652] |
| `control` (V3 inherited baseline vs production) | 120 | 0.483 | [0.396, 0.572] |

Both are *genuinely* 0.5 (identical or proven-equivalent code), yet they landed 0.567 and 0.483. **±0.10 is the noise floor even at n=120.** The harness is fair — it is just high-variance.

Consequence: **every prior whole-game "win" was inside that noise and is retracted.**
- PH-fix 15-5 = 0.75 (n=20)
- N=32 sampling 25-15 = 0.625 (n=40)
- all V3 ablation cells (n=60–80): control 0.58–0.64, phfix 0.39–0.48, v3 0.42–0.45, ph_vis 0.45, ph_vis_final 0.54

None were distinguishable from 0.5. To separate 0.50 from 0.55 cleanly needs ~400 games per cell. The piloting edge is sub-percent per decision — far below that. **Whole-game A/B is structurally the wrong instrument for per-decision heuristic tuning.** The only directional signal that survived pooling: forcing every KO (`phfix`) = 0.43 over 140 games — mildly negative, consistent with the KO-sequencing principle below.

## 2. The applicable-state paired-world test (the right instrument)

For a heuristic H: find the exact states where H applies; for each, have the **engine** score the competing choices on the **same paired hidden worlds**, evaluate at the start of my next turn, and average the paired leaf difference. The determinization noise cancels (paired), so it measures the *decision* directly with a fraction of a whole game's variance. Tool: `tools/ko_sequencing_state_test_v1.py`.

## 3. KO sequencing — first signal that clears the noise

**Principle (provably weakly dominant).** On your turn the opponent cannot respond; they only act after you end it. So a KO can never be lost by *waiting* — only one of your *own* actions can make the attack non-lethal. Therefore "develop every non-endangering action, attack **last**" weakly dominates "attack now": same KO either way, plus a banked turn of development. **Forcing a KO immediately is never strictly better.**

**"Endanger" is concrete per attack:**
- Static KO (Land Crush 90, Super Psy Bolt 30): endangering = retreat/evolve the attacker, or lose its energy. Everything else is safe.
- Powerful Hand (20×hand): the hand *is* the damage, so endangering = dropping the hand below `ceil(target_hp/20)`. Drawing is safe and good; spending hand cards is the risk.

**Result** (50 KO-available states from self-play, 16 paired worlds each):

```
bank-KO minus attack-now = +26.9 eval units  (SE 10.6, ~2.5 sigma, CI excludes 0)
states banking clearly better: 21/50    attack-now clearly better: 7/50
```

**Caveats (do not overclaim):**
- Eval units, **not win rate** (+27 ≈ one body's worth of board). It says the position is better by the evaluator's lights; the evaluator is a proxy.
- Used V3's **conservative** non-endangering set (`deck_policy_v3.safe_pre_attack_indices`, benched evolve/attach only) → this is a **lower bound** on the full principle.
- **7/50 favor attacking now** (developing can expose a fresh Pokémon to a counter-KO) → integrate as **search-chosen per state**, never a blanket/forced rule.

## Next

1. Broaden the non-endangering set toward the full principle (allow any play that does not endanger), re-measure on applicable states — does the effect grow?
2. Integrate KO-sequencing as a **continuation policy** (search compares attack-now vs bank-the-KO per state); never a forced move.
3. Then resume one heuristic at a time via the same applicable-state test: Poffin → Boss → Hilda/Dawn → Run Away Draw. Whole-game A/B only as a final, large-n confirmation of a surviving stack.
