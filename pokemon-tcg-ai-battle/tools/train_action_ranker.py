"""H024-v2 Phase 4: the action-CONDITIONED sibling ranker (the central object).

Per legal option of a winner's decision: learned card-id EMBEDDING + decoded card effects + action
descriptor + root-state features -> shared MLP -> one logit; listwise softmax over the decision's
options; trained to predict the option the WINNER actually chose (non-circular target). Grouped by
decision. Reports WITHIN-DECISION top-1 / pairwise vs the chose-option-0 baseline, AND the required
component ablations (no-embedding / no-effects / no-root) so we can see what actually carries it.

    python tools/train_action_ranker.py [--epochs 25]
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
CARD_IDS = sorted({r["card_id"] for r in ROWS})
ID2IX = {c: i for i, c in enumerate(CARD_IDS)}


def dense_vec(r, use_eff=True, use_root=True):
    d = [1.0 if r["otype"] == t else 0.0 for t in OTYPES]
    d += [r["c_pokemon"], r["c_trainer"], r["c_energy"], r["c_basic"], r["c_evo"], r["c_ex"],
          r["c_hp"] / 300.0, r["c_bestdmg"] / 300.0, r["a_dmg"] / 300.0, r["a_cost"] / 4.0,
          r["t_inplay"], r["t_isopp"]]
    eff = [r["e_" + k] for k in EFFECT_KEYS]
    d += eff if use_eff else [0.0] * len(eff)
    root = r["root"]
    d += root if use_root else [0.0] * len(root)
    return d


def build(use_eff=True, use_root=True):
    groups = defaultdict(list)
    for r in ROWS:
        groups[r["gid"]].append(r)
    G = []
    for gid, opts in groups.items():
        if len(opts) < 2:
            continue
        cidx = [ID2IX[o["card_id"]] for o in opts]
        dense = [dense_vec(o, use_eff, use_root) for o in opts]
        chosen = next((j for j, o in enumerate(opts) if o["chosen"] == 1), None)
        if chosen is None:
            continue
        G.append((cidx, dense, chosen))
    return G


class Ranker(nn.Module):
    def __init__(self, n_cards, dense_dim, emb=16, use_emb=True):
        super().__init__()
        self.use_emb = use_emb
        self.emb = nn.Embedding(n_cards, emb)
        inp = (emb if use_emb else 0) + dense_dim
        self.net = nn.Sequential(nn.Linear(inp, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, cidx, dense):
        x = torch.cat([self.emb(cidx), dense], -1) if self.use_emb else dense
        return self.net(x).squeeze(-1)


def run(name, use_emb, use_eff, use_root, train, test, mean, std, epochs):
    torch.manual_seed(0)
    dense_dim = len(train[0][1][0])
    m = Ranker(len(CARD_IDS), dense_dim, use_emb=use_emb)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    order = list(range(len(train)))
    print(f"  [{name}] training {epochs} epochs over {len(train)} decisions...", flush=True)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(order)
        ep_loss = 0.0
        for s in range(0, len(order), 64):
            opt.zero_grad()
            loss = 0.0
            batch = order[s:s + 64]
            for gi in batch:
                cidx, dense, ch = train[gi]
                ci = torch.tensor(cidx)
                dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
                logits = m(ci, dn)
                loss = loss + nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([ch]))
            (loss / len(batch)).backward()
            opt.step()
            ep_loss += float(loss)
        if ep == 0 or (ep + 1) % 4 == 0 or ep == epochs - 1:
            print(f"    [{name}] epoch {ep+1}/{epochs} train-loss {ep_loss/len(train):.4f}", flush=True)
    # within-decision metrics on test
    top1 = ph = pt = n = 0
    with torch.no_grad():
        for cidx, dense, ch in test:
            dn = (torch.tensor(dense, dtype=torch.float32) - mean) / std
            p = m(torch.tensor(cidx), dn).numpy()
            n += 1
            top1 += int(np.argmax(p) == ch)
            for i in range(len(p)):
                if i == ch:
                    continue
                pt += 1
                ph += int(p[ch] > p[i])
    print(f"  {name:<26} top-1 {top1/n:.3f} | pairwise {ph/pt:.3f} (n={n})")
    return top1 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    full = build(True, True)
    rng = np.random.default_rng(0)
    idx = np.arange(len(full)); rng.shuffle(idx)
    cut = int(0.8 * len(full))
    tr_ids, te_ids = set(idx[:cut].tolist()), set(idx[cut:].tolist())

    # standardization from training dense rows of the FULL feature set
    alld = np.array([d for gi in tr_ids for d in full[gi][1]], dtype=np.float32)
    mean = torch.tensor(alld.mean(0)); std = torch.tensor(alld.std(0) + 1e-6)

    print(f"{len(full)} decisions ({sum(len(g[0]) for g in full)} options), 80/20 split")
    rand = np.mean([1.0 / len(full[gi][0]) for gi in te_ids])
    opt0 = np.mean([1.0 if full[gi][2] == 0 else 0.0 for gi in te_ids])
    print(f"BASELINE within-decision top-1: random {rand:.3f} | chose-option-0 {opt0:.3f}  <- bar to beat\n")

    variants = [("FULL (emb+effects+action+root)", True, True, True),
                ("no embedding", False, True, True),
                ("no effects", True, False, True),
                ("no root (action-only)", True, True, False)]
    for name, ue, uf, ur in variants:
        G = build(uf, ur)
        tr = [G[i] for i in tr_ids]; te = [G[i] for i in te_ids]
        ddim = len(G[0][1][0])
        ad = np.array([d for g in tr for d in g[1]], dtype=np.float32)
        mn = torch.tensor(ad.mean(0)); sd = torch.tensor(ad.std(0) + 1e-6)
        run(name, ue, uf, ur, tr, te, mn, sd, args.epochs)
    print("\nRead: FULL must beat chose-option-0 to matter. Compare FULL vs no-embedding / no-effects to")
    print("see whether the card embedding and decoded effects actually carry the within-decision signal.")


if __name__ == "__main__":
    main()
