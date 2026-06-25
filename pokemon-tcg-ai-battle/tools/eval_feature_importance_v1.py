"""Which features predict winning? Self-play the agent, record the 47-feature vector at every decision
and whether that player went on to win, then report which features separate winning states from losing
ones. This tells us which terms eval.py should have, instead of guessing weights.

Uses logistic regression (interpretable coefficients) + univariate correlation. Read-only on the new deck.
  python tools/eval_feature_importance_v1.py --games 250
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW_DECK = Path(r"C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/decks/current_deck.csv")
sys.path.insert(0, str(ROOT / "agent"))
import features as FT          # noqa: E402
import deck_policy_v3 as DP3   # noqa: E402
import main as M               # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle_environments.envs.cabt.cabt as cabt   # noqa: F401
    from kaggle_environments import make

DECK = [int(x) for x in NEW_DECK.read_text(encoding="utf-8").split() if x.strip()]
M.DECK = DECK
KEYS = list(FT.FEATURE_KEYS)


def phaware(obs):
    if obs.get("select") is None:
        return list(DECK)
    try:
        ko = DP3.best_ko_attack(obs)
        if ko is not None:
            return [ko[0]]
    except Exception:
        pass
    return M.agent(obs)


def generate(games: int):
    rows_f, rows_y = [], []
    for g in range(games):
        recorded = []

        def logger(obs):
            try:
                if obs.get("select") is not None:
                    seat = (obs.get("current") or {}).get("yourIndex", 0)
                    recorded.append((FT.vectorize(FT.encode_state(obs)), seat))
            except Exception:
                pass
            return phaware(obs)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = make("cabt")
                env.run([logger, logger])
            last = env.steps[-1]
            r0, r1 = last[0].get("reward"), last[1].get("reward")
        except Exception:
            continue
        if r0 is None or r1 is None or r0 == r1:
            continue
        winner = 0 if r0 > r1 else 1
        for fv, seat in recorded:
            if fv is not None and len(fv) == len(KEYS):
                rows_f.append(fv)
                rows_y.append(1 if seat == winner else 0)
        if (g + 1) % 50 == 0:
            print(f"  {g + 1}/{games} games, {len(rows_y)} state-rows", flush=True)
    return np.asarray(rows_f, dtype=float), np.asarray(rows_y, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=250)
    args = ap.parse_args()

    print(f"self-play {args.games} games to collect (features, won) rows...", flush=True)
    X, y = generate(args.games)
    print(f"\n{len(y)} rows, win rate {y.mean():.3f}, {X.shape[1]} features\n", flush=True)

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    # univariate: correlation of each feature with the win label
    corr = np.array([np.corrcoef(Xs[:, j], y)[0, 1] if sd[j] > 1e-6 else 0.0 for j in range(X.shape[1])])

    # multivariate: logistic regression coefficients (standardized features -> comparable magnitudes)
    coef = None
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
        coef = clf.coef_[0]
        auc = None
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, clf.predict_proba(Xs)[:, 1])
        except Exception:
            pass
        print(f"logistic AUC (in-sample): {auc}\n" if auc is not None else "")
    except Exception as e:
        print(f"(sklearn unavailable: {e}; reporting correlation only)\n")

    order = np.argsort(-(np.abs(coef) if coef is not None else np.abs(corr)))
    print("feature importance for WINNING (sorted; coef = standardized logistic, corr = univariate):")
    print(f"  {'feature':26s} {'coef':>8s} {'corr':>8s}   (eval.py already uses: prize, hp, body, energy)")
    EVAL_HAS = {"prize_lead", "my_prizes_left", "opp_prizes_left", "my_active_hp", "opp_active_hp", "active_n_energy"}
    for j in order[:25]:
        c = coef[j] if coef is not None else float("nan")
        tag = " <- in eval-ish" if KEYS[j] in EVAL_HAS else ""
        print(f"  {KEYS[j]:26s} {c:+8.3f} {corr[j]:+8.3f}{tag}")


if __name__ == "__main__":
    main()
