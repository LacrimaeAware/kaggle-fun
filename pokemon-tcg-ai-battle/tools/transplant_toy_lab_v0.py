"""FEATURE-DELTA TRANSPLANT TOY LAB V0. A synthetic concept test: does axis-conditioned feature-delta similarity
beat one global similarity score for deciding whether an action analogy transfers? Controlled ground truth, no
real data, no gameplay, no model training of the Starmie proposer.

Design (honest): the ground-truth value depends on a SMALL, FAMILY-DEPENDENT decisive-axis subset (mirroring the
real domain's per-family projections), plus confounding noise axes. A single global weighting cannot be optimal
across families with different decisive axes; per-family axis-conditioning can. We then measure which method best
recovers the held-out value + avoids bad transplants.

  PYTHONIOENCODING=utf-8 python tools/transplant_toy_lab_v0.py
"""
from __future__ import annotations
import collections
import html
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent.parent / "data" / "generated" / "transplant_toy_lab_v0"
RNG = np.random.default_rng(7)

# ---- axes (continuous/ordinal); the last three are pure confounders ----
AXES = ["attacker_readiness", "energy_shortfall", "target_role", "effect_nullification", "turn_progress",
        "safe_development_remaining", "prize_liability", "opponent_threat", "hidden_risk_proxy",
        "noise_a", "noise_b", "noise_c"]
AX = {a: i for i, a in enumerate(AXES)}
FAMILIES = {
    "ATTACH": ["attach_main", "attach_engine", "attach_redundant"],
    "SELECT_CARD": ["fetch_missing_evolution", "fetch_redundant", "fetch_immediate_attacker"],
    "ATTACK": ["attack_KO", "attack_chip"],
    "PLAY": ["heal", "gust", "discard_energy"],
}
# TRUE decisive axes per (family, role) -- the only axes the value depends on (besides effect_nullification on ATTACK)
DECISIVE = {
    "attach_main": ["energy_shortfall", "target_role"],
    "attach_engine": ["attacker_readiness", "target_role"],
    "attach_redundant": ["attacker_readiness"],
    "fetch_missing_evolution": ["safe_development_remaining"],
    "fetch_redundant": [],
    "fetch_immediate_attacker": ["attacker_readiness", "opponent_threat"],
    "attack_KO": ["safe_development_remaining", "prize_liability", "effect_nullification"],
    "attack_chip": ["safe_development_remaining", "turn_progress", "effect_nullification"],
    "heal": ["opponent_threat"],
    "gust": ["prize_liability"],
    "discard_energy": ["opponent_threat"],
}


def true_value(role, v):
    """Transparent ground-truth value in {-1,0,+1}. Reads ONLY the role's decisive axes."""
    g = lambda a: v[AX[a]]
    if role == "attach_main":
        return 1 if (g("energy_shortfall") <= 1 and g("target_role") == 0) else (-1 if g("target_role") != 0 else 0)
    if role == "attach_engine":
        return 1 if (g("attacker_readiness") == 0 and g("target_role") == 1) else (-1 if g("target_role") != 1 else 0)
    if role == "attach_redundant":
        return -1 if g("attacker_readiness") >= 2 else 0
    if role == "fetch_missing_evolution":
        return 1 if g("safe_development_remaining") >= 1 else -1
    if role == "fetch_redundant":
        return -1
    if role == "fetch_immediate_attacker":
        return 1 if (g("attacker_readiness") < 2 and g("opponent_threat") >= 2) else 0
    if role == "attack_KO":
        if g("effect_nullification") == 1:
            return -1
        return 1 if (g("safe_development_remaining") == 0 or g("prize_liability") >= 3) else 0
    if role == "attack_chip":
        if g("effect_nullification") == 1:
            return -1
        return -1 if (g("safe_development_remaining") >= 1 and g("turn_progress") < 0.5) else 0
    if role == "heal":
        return 1 if g("opponent_threat") >= 2 else 0
    if role == "gust":
        return 1 if g("prize_liability") >= 2 else 0
    if role == "discard_energy":
        return 1 if g("opponent_threat") >= 2 else 0
    return 0


def gen(n):
    rows = []
    roles = [(f, r) for f, rs in FAMILIES.items() for r in rs]
    for i in range(n):
        fam, role = roles[RNG.integers(len(roles))]
        v = np.array([
            RNG.integers(0, 4),                # attacker_readiness 0-3
            RNG.integers(0, 4),                # energy_shortfall 0-3
            RNG.integers(0, 3),                # target_role 0=main,1=engine,2=other
            RNG.integers(0, 2),                # effect_nullification 0/1
            round(float(RNG.random()), 3),     # turn_progress 0-1
            RNG.integers(0, 4),                # safe_development_remaining 0-3
            RNG.integers(0, 4),                # prize_liability 0-3
            RNG.integers(0, 4),                # opponent_threat 0-3
            round(float(RNG.random()), 3),     # hidden_risk_proxy (looks relevant, IGNORED by truth)
            round(float(RNG.random()), 3),     # noise_a
            round(float(RNG.random()), 3),     # noise_b
            round(float(RNG.random()), 3),     # noise_c
        ], dtype=float)
        tv = true_value(role, v)
        observed = tv + RNG.normal(0, 0.35)    # noisy observed consequence
        rows.append({"id": f"x{i}", "family": fam, "role": role,
                     "semantic_action_key": f"{fam}:{role}", "vec": v.tolist(),
                     "true_value": int(tv), "observed_outcome": round(float(observed), 3),
                     "decisive_axes": DECISIVE[role]})
    return rows


# ---- normalization for distances ----
def _matrix(rows):
    M = np.array([r["vec"] for r in rows], dtype=float)
    mu, sd = M.mean(0), M.std(0) + 1e-9
    return (M - mu) / sd, mu, sd


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mem = gen(1500)
    qry = gen(500)
    (OUT / "toy_replay_memory.jsonl").write_text(
        "\n".join(json.dumps(r) for r in mem), encoding="utf-8")
    # ground-truth rules
    (OUT / "toy_ground_truth_rules.json").write_text(json.dumps({
        "axes": AXES, "families": FAMILIES, "decisive_axes_per_role": DECISIVE,
        "confounders": ["hidden_risk_proxy", "noise_a", "noise_b", "noise_c"],
        "design": "value depends ONLY on each role's decisive axes; confounders vary randomly and never affect value; "
                  "decisive axes DIFFER across families -> a single global weighting cannot be optimal for all.",
        "rules_human_readable": {
            "attach_main": "+1 if energy_shortfall<=1 and target_role=main; -1 if target_role!=main",
            "attach_engine": "+1 if attacker_readiness=0 and target_role=engine; -1 if target_role!=engine",
            "attack_KO": "-1 if effect_nullification; else +1 if safe_dev=0 or prize_liability>=3",
            "attack_chip": "-1 if effect_nullification; else -1 if safe_dev>=1 and turn_progress early",
            "heal": "+1 if opponent_threat>=2", "gust": "+1 if prize_liability>=2", "fetch_redundant": "-1 always",
        },
    }, indent=2), encoding="utf-8")

    Mmem, mu, sd = _matrix(mem)
    Mqry = (np.array([r["vec"] for r in qry], dtype=float) - mu) / sd
    mem_obs = np.array([r["observed_outcome"] for r in mem])
    mem_tv = np.array([r["true_value"] for r in mem])
    mem_fam = np.array([r["family"] for r in mem])
    mem_role = np.array([r["role"] for r in mem])

    # M0 global weights: learned single weight per axis (|corr| with observed value) -- strongest single weighting
    w_global = np.abs([np.corrcoef(Mmem[:, j], mem_obs)[0, 1] if Mmem[:, j].std() > 0 else 0 for j in range(len(AXES))])
    w_global = np.nan_to_num(w_global)

    # per-family axis weights from |corr| within family (structure-derived) -> M2
    fam_w = {}
    for fam in FAMILIES:
        idx = mem_fam == fam
        ww = np.abs([np.corrcoef(Mmem[idx, j], mem_obs[idx])[0, 1] if Mmem[idx, j].std() > 0 else 0
                     for j in range(len(AXES))])
        fam_w[fam] = np.nan_to_num(ww)

    # M1 hand mask per family (a plausible human guess; deliberately imperfect)
    HAND = {"ATTACH": ["energy_shortfall", "target_role", "attacker_readiness"],
            "SELECT_CARD": ["safe_development_remaining", "attacker_readiness"],
            "ATTACK": ["safe_development_remaining", "prize_liability", "turn_progress"],  # MISSES effect_nullification on purpose
            "PLAY": ["opponent_threat", "prize_liability"]}

    # M4 learned per-family mask: logistic on per-axis |delta| of memory pairs -> same-value-sign?
    fam_clf = {}
    for fam in FAMILIES:
        idx = np.where(mem_fam == fam)[0]
        if len(idx) < 40:
            continue
        a = RNG.choice(idx, 1200); b = RNG.choice(idx, 1200)
        dd = np.abs(Mmem[a] - Mmem[b])
        same = (mem_tv[a] == mem_tv[b]).astype(int)
        try:
            clf = LogisticRegression(max_iter=400).fit(dd, same)
            fam_clf[fam] = np.abs(clf.coef_[0])
        except Exception:
            fam_clf[fam] = fam_w[fam]

    def knn(qvec, fam, weights, same_family, k=7):
        idx = np.where(mem_fam == fam)[0] if same_family else np.arange(len(mem))
        d = np.sqrt(((Mmem[idx] - qvec) ** 2 * (weights ** 2)).sum(1))
        order = idx[np.argsort(d)[:k]]
        return order

    def evaluate(name, weights_fn, same_family, abstain=False):
        sq_err = []
        bad = 0
        abst = 0
        n = 0
        scored = 0
        for qi, r in enumerate(qry):
            fam = r["family"]
            w = weights_fn(fam)
            nb = knn(Mqry[qi], fam, w, same_family)
            if len(nb) == 0:
                continue
            n += 1
            # support / contradiction for abstain
            nb_tv = mem_tv[nb]
            support = len(nb)
            contradiction = float(np.mean(nb_tv != np.sign(np.mean(nb_tv)))) if support else 1.0
            if abstain and (support < 5 or contradiction > 0.45):
                abst += 1
                continue
            scored += 1
            est = float(np.mean(mem_obs[nb]))
            sq_err.append((est - r["true_value"]) ** 2)
            # bad transplant: nearest neighbor true-value sign disagrees with query true-value sign
            nn = nb[0]
            if np.sign(mem_tv[nn]) != np.sign(r["true_value"]) and r["true_value"] != 0:
                bad += 1
        return {"name": name, "value_mse": round(float(np.mean(sq_err)), 4) if sq_err else None,
                "bad_transplant_rate_pct": round(100 * bad / max(1, scored), 1),
                "abstention_rate_pct": round(100 * abst / max(1, n), 1), "scored": scored, "queries": n}

    uniform = np.ones(len(AXES))
    results = [
        evaluate("M0_GLOBAL_SCALAR", lambda f: w_global, same_family=False),
        evaluate("M0u_GLOBAL_UNIFORM", lambda f: uniform, same_family=False),
        # CONTROL: same-family but UNIFORM weights -> isolates family-filtering from axis-conditioning
        evaluate("M0b_SAMEFAM_UNIFORM", lambda f: uniform, same_family=True),
        evaluate("M1_FAMILY_HAND_MASK", lambda f: np.array([1.0 if a in HAND[f] else 0.0 for a in AXES]), same_family=True),
        evaluate("M2_AXIS_CONDITIONED_DELTA", lambda f: fam_w[f], same_family=True),
        evaluate("M4_LEARNED_AXIS_MASK", lambda f: fam_clf.get(f, fam_w[f]), same_family=True),
        evaluate("M5_SUPPORT_ABSTAIN", lambda f: fam_clf.get(f, fam_w[f]), same_family=True, abstain=True),
    ]

    # M3 leave-one-axis decisive-axis detection: per role, which axis-removal most degrades same-value retrieval
    detect = {}
    for fam in FAMILIES:
        idx = np.where(mem_fam == fam)[0]
        base = fam_clf.get(fam, fam_w[fam])
        ranked = [AXES[j] for j in np.argsort(-base)]
        roles = set(mem_role[idx])
        for role in roles:
            true_dec = set(DECISIVE[role]) or set()
            topk = set(ranked[:max(1, len(true_dec))])
            hit = len(topk & true_dec) / max(1, len(true_dec)) if true_dec else (1.0 if not true_dec else 0.0)
            detect[role] = {"true_decisive": sorted(true_dec), "detected_top": sorted(topk),
                            "detection_recall": round(hit, 2)}
    det_recall = round(float(np.mean([d["detection_recall"] for d in detect.values() if d["true_decisive"]])), 2)

    comparison = {"memory_rows": len(mem), "query_rows": len(qry), "methods": results,
                  "decisive_axis_detection_recall_mean": det_recall, "per_role_detection": detect,
                  "global_axis_weights": {AXES[j]: round(float(w_global[j]), 3) for j in range(len(AXES))}}
    (OUT / "similarity_method_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # ---- feature-removal report (Section 5): pairs where one axis flips the analogy ----
    fr = []
    for _ in range(400):
        i, j = int(RNG.integers(len(mem))), int(RNG.integers(len(mem)))
        if mem[i]["role"] != mem[j]["role"]:
            continue
        xi, xj = Mmem[i], Mmem[j]
        full = float(np.sqrt(((xi - xj) ** 2).sum()))
        without = {AXES[a]: round(float(np.sqrt(((xi - xj) ** 2).sum() - (xi[a] - xj[a]) ** 2)), 3) for a in range(len(AXES))}
        # decisive = axis whose removal most reduces distance AND that is in the role's true decisive set
        drop = {a: full - without[a] for a in without}
        dec = sorted(drop, key=lambda a: -drop[a])[:2]
        true_dec = DECISIVE[mem[i]["role"]]
        same_val = mem[i]["true_value"] == mem[j]["true_value"]
        if (not same_val) and any(d in true_dec for d in dec) and len(fr) < 60:
            fr.append({"pair_id": f"{mem[i]['id']}_{mem[j]['id']}", "role": mem[i]["role"],
                       "full_similarity": round(full, 3), "similarity_without_axis": without,
                       "neighbor_decisive_axes_detected": dec, "true_decisive_axes": true_dec,
                       "true_values": [mem[i]["true_value"], mem[j]["true_value"]],
                       "interpretation": "globally close but opposite value; the decisive axis (in the role's true "
                                         "decisive set) is what separates them -- global distance under-weights it."})
    (OUT / "feature_removal_similarity_report.json").write_text(
        json.dumps({"examples": fr, "count": len(fr)}, indent=2), encoding="utf-8")

    # ---- review pack ----
    m = {r["name"]: r for r in results}
    review = []
    # find concrete query examples where global picks a bad NN but axis-conditioned does not
    for qi, r in enumerate(qry[:400]):
        fam = r["family"]
        nb_g = knn(Mqry[qi], fam, w_global, same_family=False)
        nb_a = knn(Mqry[qi], fam, fam_clf.get(fam, fam_w[fam]), same_family=True)
        if len(nb_g) and len(nb_a) and r["true_value"] != 0:
            g_bad = np.sign(mem_tv[nb_g[0]]) != np.sign(r["true_value"])
            a_ok = np.sign(mem_tv[nb_a[0]]) == np.sign(r["true_value"])
            if g_bad and a_ok and len(review) < 24:
                review.append({"category": "global_falsely_accepts_axis_fixes", "query": r["semantic_action_key"],
                               "query_true_value": r["true_value"], "decisive_axes": r["decisive_axes"],
                               "global_NN_value": int(mem_tv[nb_g[0]]), "axis_NN_value": int(mem_tv[nb_a[0]])})
    (OUT / "review_examples.jsonl").write_text("\n".join(json.dumps(r) for r in review), encoding="utf-8")
    css = "body{font:13px system-ui,sans-serif;margin:18px;background:#0f1117;color:#dde}.c{border:1px solid #2a2f3a;border-radius:7px;padding:8px 11px;margin:6px 0;background:#161a22}.bad{color:#ff7a7a}.ok{color:#7ee787}.tag{display:inline-block;padding:0 6px;border-radius:4px;background:#22303f;margin-right:4px;font-size:11px}"
    rows_html = "".join(
        f"<div class='c'><span class='tag'>{html.escape(e['query'])}</span> true={e['query_true_value']} "
        f"decisive={html.escape(str(e['decisive_axes']))} | <span class='bad'>global NN val={e['global_NN_value']}</span> "
        f"-> <span class='ok'>axis NN val={e['axis_NN_value']}</span></div>" for e in review)
    summ = (f"<p>M0 global MSE {m['M0_GLOBAL_SCALAR']['value_mse']} bad {m['M0_GLOBAL_SCALAR']['bad_transplant_rate_pct']}% "
            f"| M2 axis MSE {m['M2_AXIS_CONDITIONED_DELTA']['value_mse']} bad {m['M2_AXIS_CONDITIONED_DELTA']['bad_transplant_rate_pct']}% "
            f"| M4 learned MSE {m['M4_LEARNED_AXIS_MASK']['value_mse']} bad {m['M4_LEARNED_AXIS_MASK']['bad_transplant_rate_pct']}% "
            f"| M5 abstain {m['M5_SUPPORT_ABSTAIN']['abstention_rate_pct']}% bad {m['M5_SUPPORT_ABSTAIN']['bad_transplant_rate_pct']}%</p>")
    (OUT / "review_examples.html").write_text(
        f"<html><head><meta charset='utf-8'><style>{css}</style></head><body><h1>Transplant toy lab review</h1>"
        f"{summ}<h2>global falsely accepts, axis-conditioned fixes ({len(review)})</h2>{rows_html}</body></html>",
        encoding="utf-8")

    # ---- verdict: judged on the ISOLATED axis-conditioning gain (vs same-family-uniform control), not the
    #      confounded global-vs-axis gap (which also includes family-filtering) ----
    ctrl = m["M0b_SAMEFAM_UNIFORM"]            # same-family + uniform weights (family filter, no axis weighting)
    best = min((m["M2_AXIS_CONDITIONED_DELTA"], m["M4_LEARNED_AXIS_MASK"]), key=lambda x: x["value_mse"])
    iso_mse = (ctrl["value_mse"] - best["value_mse"]) / ctrl["value_mse"] if ctrl["value_mse"] else 0
    iso_bad = (ctrl["bad_transplant_rate_pct"] - best["bad_transplant_rate_pct"]) / max(1e-9, ctrl["bad_transplant_rate_pct"])
    # also the family-filter contribution (global -> same-family-uniform)
    g = m["M0_GLOBAL_SCALAR"]
    famfilter_mse = (g["value_mse"] - ctrl["value_mse"]) / g["value_mse"] if g["value_mse"] else 0
    if iso_mse >= 0.20 and iso_bad >= 0.15 and det_recall >= 0.6:
        verdict = "A_AXIS_DELTA_TRANSPLANT_TOY_VALIDATED"
    elif iso_mse >= 0.08 or iso_bad >= 0.08:
        verdict = "B_AXIS_DELTA_DIRECTIONAL_ONLY"
    else:
        verdict = "C_GLOBAL_SIMILARITY_SUFFICIENT_IN_TOY"
    comparison["verdict"] = verdict
    comparison["isolated_axis_conditioning_gain_vs_samefam_uniform"] = {
        "mse_reduction_pct": round(100 * iso_mse, 1), "bad_transplant_reduction_pct": round(100 * iso_bad, 1)}
    comparison["family_filter_gain_global_to_samefam_uniform"] = {"mse_reduction_pct": round(100 * famfilter_mse, 1)}
    comparison["note"] = ("verdict is judged on the ISOLATED axis-conditioning gain (M2/M4 vs the same-family-uniform "
                          "control M0b), which separates axis-weighting from family-filtering. Both contribute.")
    (OUT / "similarity_method_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"verdict {verdict} | det_recall {det_recall} | ISOLATED axis gain: mse -{round(100*iso_mse,1)}% bad -{round(100*iso_bad,1)}% "
          f"| family-filter gain: mse -{round(100*famfilter_mse,1)}%")
    for r in results:
        print(f"  {r['name']:26s} mse={r['value_mse']} bad={r['bad_transplant_rate_pct']}% abstain={r['abstention_rate_pct']}% scored={r['scored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
