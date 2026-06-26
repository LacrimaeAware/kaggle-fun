"""STARMIE SELECTOR DIAGNOSTIC PACK V1 -- human review pack (section 4).

Renders curated example categories from changed_decision_classes.jsonl with full per-decision context
(board / hand / legal options / tactical state / baseline-vs-selector pick), re-resolving the observation from
replays. matchup/game_result are not available (smoke saved aggregates only) and are shown as such.

  PYTHONIOENCODING=utf-8 python tools/selector_diagnostic_html_v1.py
"""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
import deck_policy_v3 as DP            # noqa: E402
import learned_proposer_adapter as AD  # noqa: E402
OUT = ROOT / "data" / "generated" / "starmie_selector_live_smoke_v1"
REPLAYS = Path("C:/Users/EcceNihilum/Desktop/GithubRepos/pokemon-ai-agent/data/external/replays")
NAME = {1031: "MegaStarmie", 1030: "Staryu", 666: "Cinderace", 17: "Ignition", 3: "Water", 1159: "HeroCape",
        1229: "Wally", 1086: "?1086"}
_EPC: dict = {}


def _obs(did):
    ep, s, seat = (int(x) for x in did.split(":"))
    if ep not in _EPC:
        if len(_EPC) > 16:
            _EPC.clear()
        try:
            _EPC[ep] = json.load(open(REPLAYS / f"{ep}.json", encoding="utf-8"))
        except Exception:
            _EPC[ep] = None
    try:
        return _EPC[ep]["steps"][s][seat].get("observation")
    except Exception:
        return None


def _ent_str(e):
    cid = DP._cid(e)
    nm = NAME.get(cid, str(cid))
    hp = DP._get(e, "hp", "?")
    en = len(DP._items(DP._get(e, "energies", [])))
    return f"{nm}(hp{hp},e{en})"


def _board(obs):
    cur = DP._current(obs)
    me = DP._perspective(cur)
    out = {}
    for role, pidx in (("our", me), ("opp", 1 - me)):
        p = DP._player(cur, pidx)
        act = DP._active(p)
        out[role] = {"active": _ent_str(act) if act else "-",
                     "bench": [_ent_str(b) for b in DP._bench(p)],
                     "hand": [NAME.get(DP._cid(c), str(DP._cid(c))) for c in DP._items(DP._get(p, "hand", []))] if role == "our" else f"{DP._get(p,'handCount','?')} cards",
                     "prizes": len(DP._items(DP._get(p, "prize", [])))}
    return out


def _context(row):
    obs = _obs(row["decision_id"])
    if obs is None:
        return None
    keys = AD.option_index_to_key(obs)
    return {
        "decision_id": row["decision_id"], "mode": row["mode"],
        "transition_class": row["transition_class"], "proposer_rank": row["proposer_rank_of_pick"],
        "proposer_prob": row["proposer_prob_of_pick"], "terminal": row["terminal_override"],
        "premature_terminal": row["premature_terminal_override"],
        "tactical": {"guaranteed_ko": row["guaranteed_ko_available"], "gamewin": row["game_winning_attack_available"],
                     "safe_dev_available": row["safe_development_available"], "prize_diff": row["prize_diff"],
                     "deckout_pressure": row["deckout_pressure"]},
        "baseline_action": row["baseline_key"], "selector_action": row["selector_key"],
        "legal_options": list(keys.values()),
        "board": _board(obs),
        "matchup": "UNAVAILABLE (smoke saved aggregates)", "game_result": "UNAVAILABLE",
    }


def _pick(rows, pred, n):
    out = []
    for r in rows:
        if len(out) >= n:
            break
        if pred(r):
            c = _context(r)
            if c:
                out.append(c)
    return out


CATS = [
    ("top3 PREMATURE terminal overrides (rank>=2) -- the regression signature",
     lambda r: r["mode"] == "top3_selector" and r["premature_terminal_override"] and (r["proposer_rank_of_pick"] or 9) >= 2, 20),
    ("top3 DEVELOP_TO_END overrides", lambda r: r["mode"] == "top3_selector" and r["transition_class"] == "DEVELOP_TO_END", 12),
    ("top1_gate rank-1 terminal overrides (kept; neutral set for contrast)",
     lambda r: r["mode"] == "top1_gate" and r["terminal_override"], 12),
    ("nonterminal ATTACH target changes (potentially-useful learned signal)",
     lambda r: r["mode"] == "top3_selector" and r["transition_class"] == "ATTACH_TARGET_CHANGE", 12),
    ("nonterminal SELECT_CARD / PLAY changes",
     lambda r: r["mode"] == "top3_selector" and r["transition_class"] in ("SELECT_CARD_CHANGE", "PLAY_CHANGE"), 12),
    ("safety hard-veto blocked overrides", lambda r: bool(r.get("hard_veto_on_pick")), 8),
]


def main() -> int:
    rows = [json.loads(l) for l in open(OUT / "changed_decision_classes.jsonl", encoding="utf-8")]
    sections = [(title, _pick(rows, pred, n)) for title, pred, n in CATS]
    # jsonl
    with open(OUT / "selector_smoke_review.jsonl", "w", encoding="utf-8") as f:
        for title, exs in sections:
            for e in exs:
                f.write(json.dumps({**e, "category": title}, default=str) + "\n")
    # html
    css = ("body{font:13px/1.5 system-ui,Segoe UI,sans-serif;margin:18px;background:#0f1117;color:#dde}"
           "h1{font-size:19px}h2{font-size:15px;margin-top:26px;border-bottom:1px solid #333;padding-bottom:4px}"
           ".c{border:1px solid #2a2f3a;border-radius:8px;padding:10px 12px;margin:8px 0;background:#161a22}"
           ".hd{font-weight:600}.bad{color:#ff7a7a}.ok{color:#7ee787}.mut{color:#8b95a5}.k{color:#9fc5ff}"
           "table{border-collapse:collapse;margin:6px 0}td{padding:1px 10px 1px 0;vertical-align:top}"
           ".tag{display:inline-block;padding:0 6px;border-radius:4px;background:#22303f;margin-right:4px;font-size:11px}")
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
             "<h1>Starmie selector live-smoke review pack</h1>",
             "<p class='mut'>Mechanism diagnostic on real Starmie decisions. matchup/game_result UNAVAILABLE "
             "(smoke saved aggregate win/loss only; games not re-run). Examples re-resolved from replays.</p>"]
    for title, exs in sections:
        parts.append(f"<h2>{html.escape(title)} ({len(exs)})</h2>")
        for e in exs:
            t = e["tactical"]
            flags = (f"<span class='tag {'bad' if e['premature_terminal'] else ''}'>"
                     f"{'PREMATURE-TERMINAL' if e['premature_terminal'] else ('terminal' if e['terminal'] else 'nonterminal')}</span>"
                     f"<span class='tag'>rank {e['proposer_rank']}</span>"
                     f"<span class='tag'>p={round(e['proposer_prob'],3) if isinstance(e['proposer_prob'],(int,float)) else e['proposer_prob']}</span>"
                     f"<span class='tag'>{'KO' if t['guaranteed_ko'] else 'no-KO'}</span>"
                     f"<span class='tag'>{'safe-dev' if t['safe_dev_available'] else 'no-safe-dev'}</span>"
                     f"<span class='tag'>prizeΔ {t['prize_diff']}</span>")
            b = e["board"]
            parts.append(
                "<div class='c'>"
                f"<div class='hd'>{e['decision_id']} <span class='mut'>[{e['mode']}] {e['transition_class']}</span></div>"
                f"<div>{flags}</div>"
                f"<table><tr><td class='mut'>baseline</td><td class='k'>{html.escape(str(e['baseline_action']))}</td></tr>"
                f"<tr><td class='mut'>selector</td><td class='{'bad' if e['premature_terminal'] else 'ok'}'>{html.escape(str(e['selector_action']))}</td></tr>"
                f"<tr><td class='mut'>our</td><td>act {html.escape(b['our']['active'])} | bench {html.escape(', '.join(b['our']['bench']) or '-')} | hand {html.escape(', '.join(map(str,b['our']['hand'])) or '-')} | prizes {b['our']['prizes']}</td></tr>"
                f"<tr><td class='mut'>opp</td><td>act {html.escape(b['opp']['active'])} | bench {html.escape(', '.join(b['opp']['bench']) or '-')} | prizes {b['opp']['prizes']}</td></tr>"
                f"<tr><td class='mut'>options</td><td class='mut'>{html.escape(' | '.join(map(str,e['legal_options'])))}</td></tr></table>"
                "</div>")
    parts.append("</body></html>")
    (OUT / "selector_smoke_review.html").write_text("\n".join(parts), encoding="utf-8")
    n = sum(len(exs) for _, exs in sections)
    print(f"wrote selector_smoke_review.html + .jsonl ({n} examples across {len(sections)} categories)")
    for title, exs in sections:
        print(f"  {len(exs):3d}  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
