"""H024-v2 Phase 4: the action-CONDITIONED sibling ranker (with forward-model consequence deltas).

Per legal option of a winner's decision: card-id EMBEDDING + decoded effects + action descriptor +
root-state features + FORWARD-MODEL one-step DELTAS (prizes/KO/dmg/draw/board) -> shared MLP -> one
logit; listwise softmax over the decision's options; trained to predict the option the WINNER chose
(non-circular target). Reports WITHIN-DECISION top-1/pairwise, STRATIFIED into all / non-option-0
(did it predict the winner's deviation from option 0) / high-criticality (consequence spread), each
vs the chose-option-0 baseline. Ablations: no-deltas / no-effects / no-embedding show what carries it.

Aggregate top-1 vs option-0 is NOT the headline (option ordering is a strong prior). The headline is
the non-option-0 and high-criticality strata: can the model find the better move when option 0 is not
automatically right?

    python tools/train_action_ranker.py [--epochs 40 --lr 0.01 --hidden 128]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
ROWS = [json.loads(l) for l in open(ROOT / "data" / "replay_db" / "action_imit.jsonl", encoding="utf-8")]
OTYPES = [0, 3, 5, 6, 7, 8, 9, 10, 13, 14]
EFFECT_KEYS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "switch_gust",
               "recover_discard", "disrupt", "discard_cost", "status", "has_ability"]
DELTA_KEYS = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn",
              "energy_attached", "board_dev", "deck_used", "discard_gain", "ends_turn",
              "wins_now", "loses_now"]
CARD_IDS = sorted({r["card_id"] for r in ROWS})
ID2IX = {c: i for i, c in enumerate(CARD_IDS)}
HIDDEN, LR = 128, 1e-2


def dense_vec(r, use_eff=True, use_root=True, use_delta=True):
    d = [1.0 if r["otype"] == t else 0.0 for t in OTYPES]
    d += [r["c_pokemon"], r["c_trainer"], r["c_energy"], r["c_basic"], r["c_evo"], r["c_ex"],
          r["c_hp"] / 300.0, r["c_bestdmg"] / 300.0, r["a_dmg"] / 300.0, r["a_cost"] / 4.0,
          r["t_inplay"], r["t_isopp"]]
    d += [r["e_" + k] for k in EFFECT_KEYS] if use_eff else [0.0] * len(EFFECT_KEYS)
    delta = [r.get("d_" + k, 0.0) for k in DELTA_KEYS]
    delta = [delta[0] / 3, delta[1] / 3, delta[2], delta[3] / 100, delta[4] / 5, delta[5] / 3,
             delta[6] / 3, delta[7] / 5, delta[8] / 5, delta[9], delta[10], delta[11]]  # rough scale
    d += delta if use_delta else [0.0] * len(delta)
    d += r["root"] if use_root else [0.0] * len(r["root"])
    return d


def consequence(r):
    return abs(r.get("d_prizes_taken", 0)) + 2 * r.get("d_opp_ko", 0) + r.get("d_dmg_dealt", 0) / 100 + 0.2 * r.get("d_cards_drawn", 0)


def build(use_eff=True, use_root=True, use_delta=True):
    groups = defaultdict(list)
    for r in ROWS:
        groups[r["gid"]].append(r)
    G = []
    for gid, opts in groups.items():
        if len(opts) < 2:
            continue
        chosen = next((j for j, o in enumerate(opts) if o["chosen"] == 1), None)
        if chosen is None:
            continue
        cidx = [ID2IX[o["card_id"]] for o in opts]
        dense = [dense_vec(o, use_eff, use_root, use_delta) for o in opts]
        dev = opts[0].get("dev", 1 if chosen != 0 else 0)
        cons = [consequence(o) for o in opts]
        crit = max(cons) - min(cons)
        G.append((cidx, dense, chosen, dev, crit))
    return G


class Ranker(nn.Module):
    def __init__(self, n_cards, dense_dim, emb=16, use_emb=True):
        super().__init__()
        self.use_emb = use_emb
        self.emb = nn.Embedding(n_cards, emb)
        inp = (emb if use_emb else 0) + dense_dim
        self.net = nn.Sequential(nn.Linear(inp, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

    def forward(self, cidx, dense):
        x = torch.cat([self.emb(cidx), dense], -1) if self.use_emb else dense
        return self.net(x).squeeze(-1)


def metrics(m, test, mean, std, subset):
    top1 = ph = pt = n = 0
    with torch.no_grad():
        for k in subset:
            cidx, dense, ch, dev, crit = test[k]
            dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
            p = m(torch.tensor(cidx), dn).numpy()
            n += 1
            top1 += int(np.argmax(p) == ch)
            for i in range(len(p)):
                if i == ch:
                    continue
                pt += 1
                ph += int(p[ch] > p[i])
    return (top1 / n if n else 0), (ph / pt if pt else 0), n


def run(name, use_emb, use_eff, use_root, use_delta, tr_ids, te_ids, full, epochs):
    torch.manual_seed(0)
    G = build(use_eff, use_root, use_delta)
    tr = [G[i] for i in tr_ids]
    te = {i: G[i] for i in te_ids}
    ad = np.array([d for g in tr for d in g[1]], dtype=np.float32)
    mean = torch.tensor(ad.mean(0)); std = torch.tensor(ad.std(0) + 1e-6)
    m = Ranker(len(CARD_IDS), len(tr[0][1][0]), use_emb=use_emb)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    order = list(range(len(tr)))
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(order)
        for s in range(0, len(order), 64):
            opt.zero_grad(); loss = 0.0
            for gi in order[s:s + 64]:
                cidx, dense, ch, _, _ = tr[gi]
                dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
                loss = loss + nn.functional.cross_entropy(m(torch.tensor(cidx), dn).unsqueeze(0), torch.tensor([ch]))
            (loss / len(order[s:s + 64])).backward(); opt.step()
    allk = list(te.keys())
    devk = [k for k in allk if te[k][3] == 1]
    crit_vals = sorted((te[k][4] for k in allk))
    hi_cut = crit_vals[int(2 / 3 * len(crit_vals))] if crit_vals else 0
    hik = [k for k in allk if te[k][4] >= hi_cut]
    a = metrics(m, te, mean, std, allk)
    dv = metrics(m, te, mean, std, devk)
    hi = metrics(m, te, mean, std, hik)
    print(f"  {name:<22} ALL top1 {a[0]:.3f} | NON-opt0 top1 {dv[0]:.3f} (n={dv[2]}) | HIGH-crit top1 {hi[0]:.3f} (n={hi[2]})")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    global HIDDEN, LR
    HIDDEN, LR = args.hidden, args.lr

    base = build(True, True, True)
    rng = np.random.default_rng(0)
    idx = np.arange(len(base)); rng.shuffle(idx)
    cut = int(0.8 * len(base))
    tr_ids, te_ids = idx[:cut].tolist(), idx[cut:].tolist()

    # stratified baselines
    teg = {i: base[i] for i in te_ids}
    devk = [i for i in te_ids if teg[i][3] == 1]
    crit_vals = sorted((teg[i][4] for i in te_ids)); hi_cut = crit_vals[int(2 / 3 * len(crit_vals))]
    hik = [i for i in te_ids if teg[i][4] >= hi_cut]
    def opt0(sub): return np.mean([1.0 if base[i][2] == 0 else 0.0 for i in sub]) if sub else 0
    def rnd(sub): return np.mean([1.0 / len(base[i][0]) for i in sub]) if sub else 0
    print(f"{len(base)} decisions, 80/20 split | non-opt0 test n={len(devk)} | high-crit test n={len(hik)}")
    print(f"BASELINE chose-option-0:  ALL {opt0(te_ids):.3f} | NON-opt0 {opt0(devk):.3f} | HIGH-crit {opt0(hik):.3f}")
    print(f"BASELINE random:          ALL {rnd(te_ids):.3f} | NON-opt0 {rnd(devk):.3f} | HIGH-crit {rnd(hik):.3f}\n")

    for name, ue, uf, ur, ud in [("FULL (+deltas)", True, True, True, True),
                                 ("no deltas", True, True, True, False),
                                 ("no effects", True, False, True, True),
                                 ("no embedding", False, True, True, True)]:
        run(name, ue, uf, ur, ud, tr_ids, te_ids, base, args.epochs)
    print("\nRead: headline = NON-opt0 and HIGH-crit top-1 (can it find the better move when option 0 is")
    print("not automatically right). FULL vs no-deltas shows whether the forward-model consequence signal")
    print("carries it. Beating option-0 on NON-opt0/HIGH-crit is the real offline win; then arena win-rate.")


if __name__ == "__main__":
    main()
