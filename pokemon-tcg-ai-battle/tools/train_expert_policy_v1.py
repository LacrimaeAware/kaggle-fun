"""Model B / B2: train a deck-specialist behavior policy on the expert dataset.

Grouped sibling-action model: for each decision, embed every legal option (type, acting-card,
target) + the root state, score each option, softmax over the group, cross-entropy on the expert's
pick. Option-order is PERMUTED every epoch so the model cannot win by memorizing option index.
Trains on single-pick (maxCount==1) decisions; held-out split is by GAME (no sibling leakage).

  python tools/train_expert_policy_v1.py --role our_deck --epochs 12
Reports B4 metrics: top-1, top-3 recall, MRR, option-0 baseline, per-action-type. Saves the model.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "expert_policy" / "dataset_v1.jsonl"
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

OPT_NAME = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD/tutor", 7: "PLAY", 8: "ATTACH",
            9: "EVOLVE", 10: "ABILITY", 13: "ATTACK", 14: "END", 12: "RETREAT"}


def load_rows(role):
    rows = []
    for ln in open(DATA, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        if r.get("k") != 1 or not r.get("sel") or len(r["sel"]) != 1:
            continue
        if not (0 <= r["sel"][0] < len(r["opts"])) or len(r["opts"]) < 2:
            continue
        if role == "our_deck" and r.get("tier") not in (1, 2):
            continue
        rows.append(r)
    return rows


def build_vocab(rows):
    cards, types = {0: 0}, {None: 0}
    for r in rows:
        for o in r["opts"]:
            for c in (o[1], o[2]):
                if c is not None and c not in cards:
                    cards[c] = len(cards)
            if o[0] not in types:
                types[o[0]] = len(types)
    return cards, types


class PolicyNet(nn.Module):
    def __init__(self, n_types, n_cards, n_sf):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, 8)
        self.card_emb = nn.Embedding(n_cards, 32)
        self.sf_mlp = nn.Sequential(nn.Linear(n_sf, 64), nn.ReLU(), nn.Linear(64, 32))
        self.score = nn.Sequential(nn.Linear(8 + 32 + 32 + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, types, cards, targets, sf, mask):
        te = self.type_emb(types)
        ce = self.card_emb(cards)
        tge = self.card_emb(targets)
        se = self.sf_mlp(sf).unsqueeze(1).expand(-1, types.size(1), -1)
        logits = self.score(torch.cat([te, ce, tge, se], -1)).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)


def encode(rows, cards, types, n_sf):
    """-> list of (type_ids, card_ids, target_ids, sf, sel_idx, sel_opt_type)."""
    out = []
    for r in rows:
        ti = [types.get(o[0], 0) for o in r["opts"]]
        ci = [cards.get(o[1], 0) for o in r["opts"]]
        gi = [cards.get(o[2], 0) for o in r["opts"]]
        sf = (r.get("sf") or [])[:n_sf]
        sf = sf + [0.0] * (n_sf - len(sf))
        out.append((ti, ci, gi, sf, r["sel"][0], r["opts"][r["sel"][0]][0]))
    return out


def batches(data, bs, permute):
    idx = list(range(len(data)))
    if permute:
        random.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [data[i] for i in idx[s:s + bs]]
        maxo = max(len(d[0]) for d in chunk)
        B = len(chunk)
        T = torch.zeros(B, maxo, dtype=torch.long)
        C = torch.zeros(B, maxo, dtype=torch.long)
        G = torch.zeros(B, maxo, dtype=torch.long)
        M = torch.zeros(B, maxo, dtype=torch.bool)
        SF = torch.zeros(B, len(chunk[0][3]))
        Y = torch.zeros(B, dtype=torch.long)
        for b, (ti, ci, gi, sf, sel, _) in enumerate(chunk):
            order = list(range(len(ti)))
            if permute:
                random.shuffle(order)
            for j, o in enumerate(order):
                T[b, j], C[b, j], G[b, j], M[b, j] = ti[o], ci[o], gi[o], True
            SF[b] = torch.tensor(sf)
            Y[b] = order.index(sel)
        yield T, C, G, SF, M, Y, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="our_deck", choices=["our_deck", "generic_opponent"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    rows = load_rows(args.role)
    eps = sorted({r["ep"] for r in rows})
    val_eps = set(eps[::7])     # ~14% of games held out
    tr = [r for r in rows if r["ep"] not in val_eps]
    va = [r for r in rows if r["ep"] in val_eps]
    n_sf = max(len(r.get("sf") or []) for r in rows)
    cards, types = build_vocab(tr)
    Dtr, Dva = encode(tr, cards, types, n_sf), encode(va, cards, types, n_sf)
    print(f"role={args.role}  decisions: train {len(Dtr)} / val {len(Dva)}  "
          f"games {len(eps)} (val {len(val_eps)})  vocab cards {len(cards)} types {len(types)} sf {n_sf}")

    net = PolicyNet(len(types) + 1, len(cards) + 1, n_sf)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for T, C, G, SF, M, Y, _ in batches(Dtr, args.bs, permute=True):
            opt.zero_grad()
            loss = lossf(net(T, C, G, SF, M), Y)
            loss.backward(); opt.step()
            tot += loss.item() * len(Y)
        if ep % 3 == 2 or ep == args.epochs - 1:
            print(f"  epoch {ep+1}: train loss {tot/len(Dtr):.3f}")

    # ---- B4 metrics on held-out games (original option order, no permutation) ----
    net.eval()
    top1 = top3 = mrr = n = opt0 = 0
    by_type = {}
    with torch.no_grad():
        for T, C, G, SF, M, Y, chunk in batches(Dva, 512, permute=False):
            logits = net(T, C, G, SF, M)
            ranks = logits.argsort(dim=1, descending=True)
            for b in range(len(Y)):
                sel = Y[b].item()
                order = (ranks[b] == sel).nonzero(as_tuple=True)[0].item()
                top1 += int(order == 0); top3 += int(order < 3); mrr += 1.0 / (order + 1)
                opt0 += int(sel == 0)
                t = chunk[b][5]
                d = by_type.setdefault(t, [0, 0, 0])
                d[0] += 1; d[1] += int(order == 0); d[2] += int(order < 3)
                n += 1
    print(f"\n=== B4 held-out metrics (role={args.role}, n={n}) ===")
    print(f"  top-1 accuracy : {top1/n:.3f}")
    print(f"  top-3 recall   : {top3/n:.3f}")
    print(f"  MRR            : {mrr/n:.3f}")
    print(f"  option-0 base  : {opt0/n:.3f}   (always-pick-index-0 accuracy)")
    print("  per selected-option-type (n, top1, top3):")
    for t, (c, t1, t3) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f"    {str(OPT_NAME.get(t, t)):12s} n={c:5d}  top1={t1/c:.3f}  top3={t3/c:.3f}")

    art = ROOT / "agent" / f"expert_policy_v1_{args.role}.pt"
    torch.save({"state": net.state_dict(), "cards": cards, "types": types, "n_sf": n_sf,
                "metrics": {"top1": top1/n, "top3": top3/n, "mrr": mrr/n, "opt0": opt0/n, "n": n}}, art)
    print(f"\nsaved {art}")


if __name__ == "__main__":
    main()
