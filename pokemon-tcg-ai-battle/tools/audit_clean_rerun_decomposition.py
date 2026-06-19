"""Clean-rerun re-audit: is B's "strict live metadata AP 0.985" a useful risk signal or a tautology?

high_regret is DEFINED as value far below the best option (regret > 5000 under N=32). `live_gap` = best_live -
this option's live value is the N=8 estimate of exactly that quantity, and N=8 ~ N=32 on easy (large-margin)
options. So predicting high_regret across ALL options from live metadata mostly re-expresses the label in free
units. This decomposes it: all-options vs non-selected vs the only non-tautological slice (the SELECTED option
being secretly high-regret), and shows live_gap ALONE (untrained) nearly reproduces the trained probe.

    python tools/audit_clean_rerun_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AU

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "manifests"
B = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/kaggle-fun/.claude/worktrees/robust-learner-v2/"
         "pokemon-tcg-ai-battle/docs/workstreams")


def load_split():
    for fn in ("continuous_terrain_representation_v1_clean_rerun.json",
               "continuous_terrain_representation_v1_eval.json"):
        p = B / fn
        if p.is_file():
            sp = json.load(open(p, encoding="utf-8")).get("split")
            if sp:
                return set(sp["train"]), set(sp["test"])
    raise SystemExit("no split found")


def rows(rec):
    lv = [float(o.get("mean_live_value") or 0.0) for o in rec["options"]]
    sl = sorted(lv, reverse=True)
    lm = float(np.mean(lv)) if lv else 0.0
    order = sorted(range(len(lv)), key=lambda i: -lv[i])
    rank = {i: r for r, i in enumerate(order)}
    sel = rec.get("search_selected_option")
    ent = float(rec.get("live_action_entropy", 0) or 0)
    mod = float(rec.get("modal_action_stability", 0) or 0)
    crit = float((rec.get("criticality") or {}).get("score", 0) or 0)
    n = len(lv)
    out = []
    for k, o in enumerate(rec["options"]):
        out.append(dict(
            group=rec["group_id"], is_sel=int(o["index"] == sel),
            live_value=lv[k], live_gap=(sl[0] - lv[k]) if sl else 0.0, live_mean=lm,
            live_margin=(sl[0] - sl[1]) if n > 1 else 0.0,
            live_var=float(o.get("live_value_variance") or 0), rank=rank[k] / max(1, n - 1),
            ent=ent, mod=mod, crit=crit,
            y=1.0 if float(o.get("high_regret_prob", 0) or 0) >= 0.5 else 0.0))
    return out


def fit(tr, te, cols):
    Xtr = np.array([[r[c] for c in cols] for r in tr], float)
    Xte = np.array([[r[c] for c in cols] for r in te], float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])
    if len(set(yte)) < 2:
        return None, None, int(yte.sum()), len(yte)
    c = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    s = c.predict_proba(Xte)[:, 1]
    return round(AP(yte, s), 3), round(AU(yte, s), 3), int(yte.sum()), len(yte)


def main():
    recs = [json.loads(l) for l in open(MAN / "continuous_terrain_v1.jsonl", encoding="utf-8")]
    tr_g, te_g = load_split()
    R = [r for rec in recs if not rec.get("eval_only") for r in rows(rec)]
    tr = [r for r in R if r["group"] in tr_g]
    te = [r for r in R if r["group"] in te_g]
    strict = ["live_value", "live_gap", "live_mean", "live_margin", "live_var", "rank", "ent", "mod", "crit"]
    # corpus-wide "sneaky" rate: high_regret options the live search thinks are playable (live_gap < 5000)
    hr = [r for r in R if r["y"]]
    sneaky = [r for r in hr if r["live_gap"] < 5000]
    print(f"corpus: {len(R)} options, {len(hr)} high_regret; sneaky (live_gap<5000) = {len(sneaky)} "
          f"({round(100*len(sneaky)/max(1,len(hr)))}%) -- the rest are trivially flagged by the live search")
    print(f"\n{'slice / features':56s} {'AP':>6s} {'AUROC':>6s} {'pos/n':>9s}")
    print("ALL OPTIONS (B's headline task):")
    print(f"  {'strict-live (replicate B R1)':54s} {fmt(fit(tr, te, strict))}")
    print(f"  {'live_gap ALONE (untrained-equivalent)':54s} {fmt(fit(tr, te, ['live_gap']))}")
    ten = [r for r in te if not r["is_sel"]]
    trn = [r for r in tr if not r["is_sel"]]
    print(f"  {'strict-live, NON-SELECTED only (trivial mass)':54s} {fmt(fit(trn, ten, strict))}")
    print("SELECTED OPTION ONLY (the only non-tautological task):")
    tes = [r for r in te if r["is_sel"]]
    trs = [r for r in tr if r["is_sel"]]
    print(f"  {'decision-live feats (margin,var,ent,mod,crit)':54s} {fmt(fit(trs, tes, ['live_margin','live_var','ent','mod','crit']))}")
    print(f"  -> selected-catastrophe positives in held-out test: {sum(r['y'] for r in tes):.0f} / {len(tes)} "
          f"(corpus-wide: {sum(1 for r in R if r['is_sel'] and r['y']):.0f} across "
          f"{len(set(r['group'] for r in R if r['is_sel'] and r['y']))} games)")


def fmt(t):
    ap, au, p, n = t
    return f"{str(ap):>6s} {str(au):>6s} {p:>4d}/{n}"


if __name__ == "__main__":
    main()
