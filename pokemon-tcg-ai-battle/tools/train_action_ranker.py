"""H024-v2 Phase 4-5: the action-CONDITIONED sibling ranker, trained two ways.

Per legal option of a winner's decision: card-id EMBEDDING + decoded effects + action descriptor +
root-state features + FORWARD-MODEL one-step DELTAS (prizes/KO/dmg/draw/board) -> shared MLP -> one
logit; listwise softmax over the decision's options.

Two targets (--target):
  imit    : predict the option the WINNER chose (the auxiliary expert prior; non-circular wrt our eval).
  distill : predict the option the frozen SEARCH TEACHER prefers (max multi-turn leaf value; the
            `bsrch` field from build_action_dataset.py --values). This DISTILLS the slow search into
            a net that is instant at match time -- the whole reason a learned policy is worth having
            (we cannot run 0.6s/decision search inside the match budget; a net is ~0 cost).
  both    : imitation CE + lam * distillation CE (expert prior + search teacher together).

Headline metrics (within decision, canonical by eq-class):
  imitation top-1  : argmax matches the winner's move, stratified all / non-opt0 / high-crit.
  DISTILL top-1    : argmax matches the SEARCH teacher's pick -- does the cheap net reproduce the
                     expensive search? This is the distillation fidelity that justifies the net.
  reference: SEARCH-vs-winner agreement and option-0 baselines are printed so the strata are anchored.

Ablations: no-deltas / no-effects / no-embedding show which inputs carry the fidelity.

    python tools/train_action_ranker.py --target imit                       # action_imit.jsonl
    python tools/train_action_ranker.py --data action_adv.jsonl --target distill
    python tools/train_action_ranker.py --data action_adv.jsonl --target both --lam 0.5
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
OTYPES = [0, 3, 5, 6, 7, 8, 9, 10, 13, 14]
EFFECT_KEYS = ["draw", "search", "search_to_bench", "energy_accel", "heal", "switch_gust",
               "recover_discard", "disrupt", "discard_cost", "status", "has_ability"]
DELTA_KEYS = ["prizes_taken", "opp_prizes_taken", "opp_ko", "dmg_dealt", "cards_drawn",
              "energy_attached", "board_dev", "deck_used", "discard_gain", "ends_turn",
              "wins_now", "loses_now"]
HIDDEN, LR = 128, 1e-2
ROWS, CARD_IDS, ID2IX = [], [], {}


def load(path):
    global ROWS, CARD_IDS, ID2IX
    fp = path if Path(path).is_absolute() else ROOT / "data" / "replay_db" / path
    ROWS = [json.loads(l) for l in open(fp, encoding="utf-8")]
    CARD_IDS = sorted({r["card_id"] for r in ROWS})
    ID2IX = {c: i for i, c in enumerate(CARD_IDS)}


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
        eq = [o.get("eq", j) for j, o in enumerate(opts)]   # equivalence class per option (canonical top-1)
        bsi = next((j for j, o in enumerate(opts) if o.get("bsrch") == 1), None)  # search teacher's pick
        G.append((cidx, dense, chosen, dev, crit, eq, bsi))
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


def metrics(m, test, mean, std, subset, label="chosen"):
    """top-1 / pairwise vs `label` target ('chosen'=winner index 2, or 'bsi'=search-best index 6)."""
    top1 = ph = pt = n = 0
    tcol = 2 if label == "chosen" else 6
    with torch.no_grad():
        for k in subset:
            g = test[k]
            tgt = g[tcol]
            if tgt is None:
                continue
            cidx, dense, eq = g[0], g[1], g[5]
            dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
            p = m(torch.tensor(cidx), dn).numpy()
            n += 1
            top1 += int(eq[int(np.argmax(p))] == eq[tgt])    # canonical: any option equivalent to target
            for i in range(len(p)):
                if eq[i] == eq[tgt]:
                    continue
                pt += 1
                ph += int(p[tgt] > p[i])
    return (top1 / n if n else 0), (ph / pt if pt else 0), n


def run(name, use_emb, use_eff, use_root, use_delta, tr_ids, te_ids, epochs, target, lam):
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
            opt.zero_grad(); loss = 0.0; nb = 0
            for gi in order[s:s + 64]:
                cidx, dense, ch, dev, crit, eq, bsi = tr[gi]
                dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
                logit = m(torch.tensor(cidx), dn).unsqueeze(0)
                li = 0.0
                if target in ("imit", "both"):
                    li = li + nn.functional.cross_entropy(logit, torch.tensor([ch]))
                if target in ("distill", "both") and bsi is not None:
                    li = li + lam * nn.functional.cross_entropy(logit, torch.tensor([bsi]))
                if isinstance(li, float):
                    continue
                loss = loss + li; nb += 1
            if nb:
                (loss / nb).backward(); opt.step()
    allk = list(te.keys())
    devk = [k for k in allk if te[k][3] == 1]
    crit_vals = sorted((te[k][4] for k in allk))
    hi_cut = crit_vals[int(2 / 3 * len(crit_vals))] if crit_vals else 0
    hik = [k for k in allk if te[k][4] >= hi_cut]
    im = metrics(m, te, mean, std, allk, "chosen")
    di = metrics(m, te, mean, std, allk, "bsi")
    di_dev = metrics(m, te, mean, std, devk, "bsi")
    di_hi = metrics(m, te, mean, std, hik, "bsi")
    print(f"  {name:<22} IMIT top1 {im[0]:.3f} | DISTILL top1 {di[0]:.3f} (n={di[2]})"
          f" | distill NON-opt0 {di_dev[0]:.3f} (n={di_dev[2]}) | distill HIGH-crit {di_hi[0]:.3f} (n={di_hi[2]})")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="action_imit.jsonl", help="dataset file under data/replay_db (action_adv.jsonl for distill)")
    ap.add_argument("--target", choices=["imit", "distill", "both"], default="imit")
    ap.add_argument("--lam", type=float, default=0.5, help="distill weight when --target both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    global HIDDEN, LR
    HIDDEN, LR = args.hidden, args.lr
    load(args.data)

    base = build(True, True, True)
    rng = np.random.default_rng(0)
    idx = np.arange(len(base)); rng.shuffle(idx)
    cut = int(0.8 * len(base))
    tr_ids, te_ids = idx[:cut].tolist(), idx[cut:].tolist()

    # stratified baselines + teacher reference
    teg = {i: base[i] for i in te_ids}
    devk = [i for i in te_ids if teg[i][3] == 1]
    crit_vals = sorted((teg[i][4] for i in te_ids)); hi_cut = crit_vals[int(2 / 3 * len(crit_vals))]
    hik = [i for i in te_ids if teg[i][4] >= hi_cut]
    def opt0(sub): return np.mean([1.0 if base[i][5][0] == base[i][5][base[i][2]] else 0.0 for i in sub]) if sub else 0
    def rnd(sub): return np.mean([1.0 / len(base[i][0]) for i in sub]) if sub else 0
    has_bsi = [i for i in te_ids if base[i][6] is not None]
    # how often the SEARCH teacher's pick equals the winner's move (eq-class) -- agreement, anchors distill
    def agree(sub): return np.mean([1.0 if base[i][5][base[i][6]] == base[i][5][base[i][2]] else 0.0
                                    for i in sub if base[i][6] is not None]) if sub else 0
    def opt0_bsi(sub): return np.mean([1.0 if base[i][5][0] == base[i][5][base[i][6]] else 0.0
                                       for i in sub if base[i][6] is not None]) if sub else 0
    print(f"{len(base)} decisions, 80/20 split | non-opt0 test n={len(devk)} | high-crit test n={len(hik)} "
          f"| with search value n={len(has_bsi)}")
    print(f"BASELINE chose-option-0 (vs winner):  ALL {opt0(te_ids):.3f} | NON-opt0 {opt0(devk):.3f} | HIGH-crit {opt0(hik):.3f}")
    print(f"BASELINE random (vs winner):          ALL {rnd(te_ids):.3f} | NON-opt0 {rnd(devk):.3f} | HIGH-crit {rnd(hik):.3f}")
    if has_bsi:
        print(f"SEARCH teacher vs winner agreement:   ALL {agree(te_ids):.3f}  (option-0 vs search {opt0_bsi(te_ids):.3f})")
    print(f"target={args.target} lam={args.lam}\n")

    for name, ue, uf, ur, ud in [("FULL (+deltas)", True, True, True, True),
                                 ("no deltas", True, True, True, False),
                                 ("no effects", True, False, True, True),
                                 ("no embedding", False, True, True, True)]:
        run(name, ue, uf, ur, ud, tr_ids, te_ids, args.epochs, args.target, args.lam)
    print("\nRead: DISTILL top-1 = does the cheap net reproduce the expensive search's pick. High distill")
    print("fidelity (esp. on NON-opt0/HIGH-crit, where option-0 is not automatically right) means the net")
    print("can REPLACE search at match time. FULL vs no-deltas shows whether the consequence signal carries it.")


if __name__ == "__main__":
    main()
