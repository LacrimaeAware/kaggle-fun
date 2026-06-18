# Cross-repo method transfer (raw landing notes)

Methods from the user's other repos that genuinely apply here, each with an honest caveat.
The recurring lesson from those repos (structured-transform-discovery, stable-grn-inference,
mechanistic-model-inference, quant-methods-vault) is that structured/clever methods rarely
beat a tuned standard baseline on accuracy; their durable value is diagnosis and
boundary-mapping. That constraint is applied below, not ignored.

| From | Method | Application to PTCG | Honest caveat |
| --- | --- | --- | --- |
| stable-grn-inference (exp 28, separability phase diagram) | map when a signal is recoverable from noise | "when is a deck/strategy edge recoverable from noisy match outcomes" at the per-matchup game counts we can log | tells you whether to bother, does not raise win rate; the answer may be "below the floor, play the global power read" (H9) |
| structured-transform-discovery | dominant-mode deflation, low-rank structure | deck matchup matrix as a covariance/low-rank object; deflate the global power-level mode, inspect the residual rock-paper-scissors | residual edges are often below the noise floor at realistic n; do not over-read a denoised matrix (H9) |
| structured-transform-discovery (controls/nulls) | the standard baseline is the yardstick | before crediting any latent-archetype model, run logistic regression / gradient boosting on hand-crafted deck features | the baseline usually wins; budget the embedding model against it, not against nothing (H10) |
| mechanistic-model-inference | intervention vs observation | force a deck into the self-play field, measure the directed win-rate response ("A suppresses B") that symmetric co-occurrence cannot show | decidable in sim is not verified real-meta truth; adaptation mutes the signal (H12) |
| quant-methods-vault (04_finance_translation) | game on a payoff matrix; maximin/mixed strategy | metagame deck selection as a small explicit game, solved for a mixed strategy over your pool vs the field | this one can buy win rate cheaply (avoids being hard-countered), and it is near-zero cost; the one place a Nash mindset earns its keep (H14) |

## Structural confound to respect

Deck strength vs pilot skill is not identifiable from ladder win/loss alone (good players
pick good decks). Break it with fixed-policy self-play (the same agent across decks), or
report the joint deck-pilot quantity and flag the non-identifiable split (H11).

## Summary

The genuinely useful transfer is the game-theoretic deck-selection layer (cheap, buys
win-rate) and the diagnostic discipline (separability floor, control baselines, intervention
vs observation, the confound) that keeps us from over-reading noisy match data. The
matrix-denoising and latent-archetype angles are for understanding and deck choice, not a
hidden win-rate source: three convergent negatives across the prior repos say so.
