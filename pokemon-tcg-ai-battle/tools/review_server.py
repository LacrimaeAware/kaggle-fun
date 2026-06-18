"""Human-in-the-loop card classification tool.

A local web app. A background worker asks your LM Studio model (Llama 3.1 8B) to tag each of
the 1267 cards with functional TCG roles; you act as oracle in the browser, reviewing ONE
card at a time (large), most-uncertain and class-balanced first, confirming or editing the
tags. You do NOT have to review all 1267. Classes are grouped, editable, and have hover help.

  python tools/review_server.py        # then open http://localhost:8771

Store: registry/card_review.json (resumes if you restart). Stdlib only; calls LM Studio's
OpenAI-compatible API at http://localhost:1234. The confirmed labels become the functional
features for the heuristics, the embeddings, and the RL agent. Thresholds in the class
descriptions are GUIDES (from the card-pool distribution), not hard rules.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "external" / "official" / "cards_full.json"
ROLES = ROOT / "registry" / "card_functional_classification.json"
IMG_DIR = ROOT / "data" / "external" / "official" / "card_images"
STORE = ROOT / "registry" / "card_review.json"
PORT = 8771
LM_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL = "meta-llama-3.1-8b-instruct"

# Functional roles, grounded in competitive Pokemon TCG terms. Multi-label, overlap is fine.
# Numbers are GUIDES from the card-pool distribution (HP median 100 / p90 230; best-attack
# median 50 / p75 110 / p90 160), not hard cutoffs. Editable in the UI; refined as we go.
TAXONOMY = {
    # Attack & win
    "main_attacker": "The deck's main damage source: a strong attack (guide: best attack ~120+ is the top ~25% of the whole pool, which includes many weak support Pokemon, so this is a guide not a cutoff). Not a low/no-damage Pokemon.",
    "tech_attacker": "Secondary or situational attacker: lower/utility damage, or 1-2 copies for a specific matchup.",
    "attack_copy": "Uses, copies, or enables attacks from another Pokemon/source.",
    "damage_mod": "Non-attacker damage support: boosts damage, reduces damage, or places damage counters.",
    "snipe_spread": "An attack that hits the opponent's Benched Pokemon (snipe one, or spread to several).",
    "win_condition": "Wins or combos toward winning beyond raw damage: extra prizes on KO, deck-out/mill, or an alternate win.",
    "gust": "A card text that switches the opponent's Active Pokémon (including forced benched->active exchange).",
    "switch_pivot": "A card text that swaps your own Active Pokémon with one of your Benched Pokémon.",
    # Energy
    "basic_energy": "Basic Energy card.",
    "special_energy": "Special Energy card (carries an extra effect).",
    "energy_accel": "Attaches MORE than the one manual Energy per turn (ramp / pseudo-energy), e.g. Dark Patch.",
    # Card flow
    "draw": "Net card advantage: draw more cards than you spend, e.g. Professor's Research, Iono.",
    "tutor": "Search the deck for a specific card (ball/search), e.g. Ultra Ball, Nest Ball.",
    "cycle": "Draw plus discard / dig and refill without net card gain.",
    "consistency": "Smooths setup / makes turns reliable. Overlaps draw and tutor on purpose; a meta-tag.",
    # Disruption
    "hand_disruption": "Attacks the opponent's HAND (shuffle away or shrink it), e.g. Iono, Judge.",
    "board_disruption": "Removes, shuffles, bounces, or otherwise disrupts opponent Pokemon already in play without being simple damage.",
    "energy_disruption": "Removes or denies the opponent's Energy.",
    "tool_disruption": "Removes or disables Pokemon Tools attached in play.",
    "stadium_disruption": "Removes or disrupts Stadium cards in play.",
    "special_condition": "Applies or modifies Special Conditions: Burned, Poisoned, Confused, Asleep, or Paralyzed.",
    "ability_disable": "Shuts off the opponent's Abilities, OR prevents the EFFECTS of attacks (not the damage), e.g. Mist Energy, Rock Fighting Energy.",
    "stall_lock": "Broad lock or stall: Item lock, Ability lock, deck-out stall.",
    # Defense & recovery
    "tanky": "Durable Pokemon that can take hits, often a tanky attacker. Guide: HP above the ~100 median (130+). Not a hard cutoff.",
    "wall": "Built to stall or block: very high HP (top ~10%, ~230+) with little or no offense; a defensive identity.",
    "protection": "Reduces or prevents DAMAGE to your Pokemon (damage reduction or immunity). Distinct from preventing effects.",
    "heal": "Heal / remove damage from your own Pokemon.",
    "retrieval": "Return Pokemon (and basic Energy) from your discard to hand or deck, e.g. Super Rod, Night Stretcher.",
    # Setup & engines
    "bench_setup": "Puts EXTRA Pokemon into play onto your Bench beyond the normal flow. If it searches the deck, also tag tutor.",
    "ability_engine": "A passive or activated Ability that generates ONGOING value (a draw/energy/damage engine). Not for one-shot or purely defensive abilities.",
    # Triggers & mechanics
    "on_ko_trigger": "Has an effect that fires when a Pokemon is Knocked Out.",
    "coin_flip": "The outcome depends on a coin flip.",
    "mill": "Discards cards from the opponent's deck (a deck-out plan).",
    # Card type / structural
    "basic_mon": "A Basic Pokemon.",
    "evolution": "A Stage 1 or Stage 2 (evolved) Pokemon.",
    "ex_mon": "Pokemon ex rule-box card, including Mega Pokemon ex.",
    "mega_mon": "Mega Pokemon ex card.",
    "tera_mon": "Tera Pokemon rule-box card with bench-damage protection.",
    "stadium": "Stadium card.",
    "tool": "Pokemon Tool card.",
    "ace_spec": "ACE SPEC card (max 1 per deck) - a deckbuilding restriction, orthogonal to function.",
}

GROUPS = {
    "Attack & win": ["main_attacker", "tech_attacker", "attack_copy", "damage_mod", "snipe_spread", "win_condition"],
    "Switching": ["gust", "switch_pivot"],
    "Energy": ["basic_energy", "special_energy", "energy_accel"],
    "Card flow": ["draw", "tutor", "cycle", "consistency"],
    "Disruption": ["hand_disruption", "board_disruption", "energy_disruption", "tool_disruption", "stadium_disruption", "special_condition", "ability_disable", "stall_lock"],
    "Defense & recovery": ["tanky", "wall", "protection", "heal", "retrieval"],
    "Setup & engines": ["bench_setup", "ability_engine"],
    "Triggers & mechanics": ["on_ko_trigger", "coin_flip", "mill"],
    "Card type": ["basic_mon", "evolution", "ex_mon", "mega_mon", "tera_mon", "stadium", "tool", "ace_spec"],
}

LOCK = threading.Lock()
STATE: dict[str, dict] = {}
TAX: dict[str, str] = dict(TAXONOMY)


def load_store() -> None:
    global TAX
    full = json.loads(FULL.read_text(encoding="utf-8"))
    roles_raw = json.loads(ROLES.read_text(encoding="utf-8")) if ROLES.exists() else {}
    roles = roles_raw.get("cards", roles_raw)
    saved = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    TAX = dict(TAXONOMY)                          # refined classes from code are authoritative
    for k, v in (saved.get("taxonomy") or {}).items():
        if k not in TAX and v == "(user-added)":  # keep only classes the user added in the UI
            TAX[k] = v
    cards = saved.get("cards", {})
    for cid, c in full.items():
        text = " ".join(s.get("t", "") for s in c.get("skills", []))
        atk = "; ".join(
            f"{a.get('n', '')} ({a.get('d', 0)} dmg, {a.get('c', 0)}E)"
            + (f": {a.get('t', '')}" if a.get("t") else "")
            for a in c.get("atk", [])
        )
        prev = cards.get(cid, {})
        rule = roles.get(cid) or {}
        rule_tags = rule.get("tags", [])
        rule_why = rule.get("why", "")
        STATE[cid] = {
            "id": cid, "n": c.get("n"), "cat": rule.get("cat", "?"),
            "rule_tags": rule_tags,
            "text": text, "atk": atk, "hp": c.get("hp", 0),
            "llm_tags": rule_tags or prev.get("llm_tags", []), "conf": 0.95 if rule_tags else prev.get("conf"),
            "why": rule_why or prev.get("why", ""), "human_tags": prev.get("human_tags"),
            "status": prev.get("status", "pending"),
        }


def save_store() -> None:
    with LOCK:
        STORE.write_text(json.dumps({
            "taxonomy": TAX,
            "cards": {cid: {k: s[k] for k in ("llm_tags", "conf", "why", "human_tags", "status")}
                      for cid, s in STATE.items() if s["status"] != "pending"},
        }, ensure_ascii=False, indent=0), encoding="utf-8")


def exemplars(limit: int = 8) -> str:
    """The Oracle loop: feed the human's confirmed labels back as few-shot examples, so the
    user's corrections steer future classifications (their fixes fix future fixes). Cards
    where the human OVERRODE the model are most informative, so prefer those, then diversity."""
    with LOCK:
        conf = [s for s in STATE.values() if s["status"] == "confirmed" and s.get("human_tags") is not None]
    if not conf:
        return ""
    disagree = [s for s in conf if set(s["human_tags"]) != set(s["llm_tags"] or [])]
    seen, pick = set(), []
    for s in disagree + conf:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        pick.append(s)
        if len(pick) >= limit:
            break
    lines = [f'- "{s["n"]}" ({s["cat"]}): {(s["text"] or s["atk"] or "")[:70]} => {", ".join(s["human_tags"]) or "none"}'
             for s in pick]
    return "\nThe human expert CONFIRMED these labels. Match this judgment and style:\n" + "\n".join(lines)


def classify_card(s: dict) -> dict | None:
    tags = ", ".join(f"{k} ({v})" for k, v in TAX.items())
    sys_p = ("You classify Pokemon TCG cards by FUNCTION. Choose all that apply (1-4) ONLY from this list:\n"
             + tags + exemplars() + "\nReturn strict JSON only: "
             '{"tags":["..."],"confidence":0.0-1.0,"why":"<=8 words"}. No prose.')
    usr = f"Card: {s['n']} (type {s['cat']}, HP {s['hp']}). Ability/effect: {s['text'] or 'none'}. Attacks: {s['atk'] or 'none'}. Rule-guess: {', '.join(s['rule_tags']) or 'none'}."
    body = json.dumps({"model": LM_MODEL, "temperature": 0, "max_tokens": 130,
                       "messages": [{"role": "system", "content": sys_p},
                                    {"role": "user", "content": usr}]}).encode()
    try:
        req = urllib.request.Request(LM_URL, body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        out = r["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", out, re.S)
        d = json.loads(m.group()) if m else {}
        valid = [t for t in d.get("tags", []) if t in TAX]
        return {"llm_tags": valid, "conf": float(d.get("confidence", 0.5)), "why": str(d.get("why", ""))[:80]}
    except Exception:
        return None


def worker() -> None:
    import random
    while True:
        with LOCK:
            pend = [cid for cid, s in STATE.items() if s["status"] == "pending"]
        if not pend:
            time.sleep(3)
            continue
        random.shuffle(pend)
        cid = pend[0]
        res = classify_card(STATE[cid])
        with LOCK:
            if res is not None:
                STATE[cid].update(res)
                STATE[cid]["status"] = "proposed"
            else:
                time.sleep(5)
        if res is not None and sum(1 for s in STATE.values() if s["status"] != "pending") % 10 == 0:
            save_store()


def queue(n: int = 24) -> list[dict]:
    with LOCK:
        prop = [s for s in STATE.values() if s["status"] == "proposed"]
    prop.sort(key=lambda s: (s["conf"] if s["conf"] is not None else 0.5))
    out, seen_primary = [], Counter()
    while prop and len(out) < n:
        prop.sort(key=lambda s: (seen_primary[(s["llm_tags"] or ["?"])[0]],
                                 s["conf"] if s["conf"] is not None else 0.5))
        s = prop.pop(0)
        seen_primary[(s["llm_tags"] or ["?"])[0]] += 1
        out.append(s)
    return out


def view(s: dict) -> dict:
    return {"id": s["id"], "n": s["n"], "cat": s["cat"], "hp": s["hp"],
            "text": s["text"], "atk": s["atk"], "rule_tags": s["rule_tags"],
            "llm_tags": s["llm_tags"], "conf": s["conf"], "why": s["why"],
            "human_tags": s["human_tags"], "status": s["status"]}


def stats() -> dict:
    with LOCK:
        c = Counter(s["status"] for s in STATE.values())
        bytag: Counter = Counter()
        for s in STATE.values():
            for t in (s["human_tags"] if s["status"] == "confirmed" else s["llm_tags"]) or []:
                bytag[t] += 1
    return {"pending": c["pending"], "proposed": c["proposed"], "confirmed": c["confirmed"],
            "total": len(STATE), "by_tag": dict(bytag.most_common())}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._send(200, UI, "text/html; charset=utf-8")
        elif p == "/api/state":
            self._send(200, {"stats": stats(), "taxonomy": TAX, "groups": GROUPS})
        elif p == "/api/queue":
            self._send(200, [view(s) for s in queue()])
        elif p == "/api/done":
            with LOCK:
                done = [view(s) for s in STATE.values() if s["status"] == "confirmed"]
            self._send(200, done[-200:][::-1])
        elif p.startswith("/img/"):
            f = IMG_DIR / p.split("/img/")[1]
            self._send(200, f.read_bytes(), "image/jpeg") if f.exists() else self._send(404, b"", "image/jpeg")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(ln) or b"{}")
        if self.path == "/api/answer":
            cid = str(data.get("id"))
            with LOCK:
                if cid in STATE:
                    STATE[cid]["human_tags"] = [t for t in data.get("tags", []) if t in TAX]
                    STATE[cid]["status"] = "confirmed"
            save_store()
            self._send(200, {"ok": True})
        elif self.path == "/api/taxonomy":
            with LOCK:
                if data.get("add"):
                    TAX[data["add"].strip().replace(" ", "_")] = data.get("desc", "(user-added)")
                if data.get("remove") and data["remove"] in TAX:
                    del TAX[data["remove"]]
            save_store()
            self._send(200, {"taxonomy": TAX})
        else:
            self._send(404, {"error": "not found"})


UI = r"""<!doctype html><html><head><meta charset="utf-8"><title>Card review</title>
<style>
:root{--bg:#0f141b;--panel:#19212c;--ink:#eef2f7;--soft:#93a1b3;--line:rgba(255,255,255,.09);--accent:#5e7fff;--good:#5fd38c;--gold:#caa24a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header b{font-size:1.05rem}.muted{color:var(--soft)}
.tab{padding:6px 13px;border-radius:8px;cursor:pointer;font-weight:600}.tab.on{background:var(--accent);color:#06122e}
.stat{background:#11181f;border:1px solid var(--line);border-radius:8px;padding:5px 11px;font-size:.84rem}
.barwrap{height:7px;background:#11181f;border-radius:4px;overflow:hidden;flex:1;min-width:120px;max-width:300px}.barwrap i{display:block;height:100%;background:var(--good)}
#addbox{margin-left:auto;display:flex;gap:6px}#addbox input{background:#11181f;border:1px solid var(--line);color:var(--ink);border-radius:7px;padding:6px 10px}
.stage{max-width:960px;margin:24px auto;padding:0 18px}
.cardbig{display:flex;gap:24px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px}
.cardbig img{width:300px;height:418px;object-fit:contain;border-radius:10px;background:#0a0e13;flex:none}
.info{min-width:0;flex:1}.info h2{margin:0 0 2px;font-size:1.5rem}.info .sub{color:var(--soft);font-size:.88rem}
.llm{margin:12px 0;padding:8px 12px;background:#11181f;border-radius:9px;border:1px solid var(--line);font-size:.9rem}
.effect{font-size:.95rem;margin-top:10px}
.chips{margin:16px 0 6px}.grp{margin-bottom:9px}.grphead{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:var(--soft);margin:6px 0 4px}
.grprow{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:7px 13px;border-radius:999px;border:1px solid var(--line);background:#11181f;cursor:pointer;font-size:.85rem;user-select:none}
.chip.on{background:var(--accent);border-color:var(--accent);color:#06122e;font-weight:700}.chip.llm{border-color:var(--gold)}
.help{min-height:2.6em;color:var(--soft);background:#11181f;border:1px solid var(--line);border-radius:9px;padding:9px 13px;font-size:.9rem}
.actions{display:flex;gap:11px;align-items:center;margin-top:14px}
.btn{padding:11px 18px;border-radius:9px;font-weight:700;cursor:pointer;border:none;font-size:.95rem}
.confirm{background:var(--good);color:#04210f}.skip,.nav{background:#11181f;color:var(--ink);border:1px solid var(--line)}
.count{margin-left:auto;color:var(--soft)}
.empty{text-align:center;color:var(--soft);padding:80px 0;font-size:1.05rem}
.kbd{font-size:.78rem;color:var(--soft);margin-top:10px}
/* done grid */
.dgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:14px}
.dcell{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px;cursor:pointer}
.dcell img{width:100%;border-radius:6px;aspect-ratio:451/630;object-fit:cover;background:#0a0e13}
.dcell .nm{font-size:.74rem;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dcell .tg{font-size:.68rem;color:var(--soft);margin-top:2px;height:2.4em;overflow:hidden}
</style></head><body>
<header>
  <b>Card review</b>
  <span class="tab on" id="tab-review" onclick="setTab('review')">To review</span>
  <span class="tab" id="tab-done" onclick="setTab('done')">Done</span>
  <span class="tab" id="toggle-view" onclick="toggleView()" style="display:none">Grid</span>
  <span class="stat" id="s-counts">...</span>
  <div class="barwrap"><i id="prog" style="width:0%"></i></div>
  <div id="addbox"><input id="newclass" placeholder="add a class"><button class="btn confirm" onclick="addClass()">+ class</button></div>
</header>
<div class="stage" id="stage"></div>
<script>
let TAX={}, GROUPS={}, tab='review', batch=[], idx=0, sel=new Set(), doneGrid=false;
async function poll(){
  const st=await(await fetch('/api/state')).json(); TAX=st.taxonomy; GROUPS=st.groups||{}; const s=st.stats;
  document.getElementById('s-counts').textContent=`${s.confirmed} done / ${s.proposed} ready / ${s.pending} pending`;
  document.getElementById('prog').style.width=(100*s.confirmed/Math.max(1,s.total))+'%';
  if(!batch.length) loadBatch();
}
async function loadBatch(){ batch=await(await fetch(tab==='review'?'/api/queue':'/api/done')).json(); idx=0; show(); }
function renderChips(c){
  const wrap=document.getElementById('chips'); if(!wrap)return; wrap.innerHTML='';
  const help=document.getElementById('help');
  const grouped=new Set(Object.values(GROUPS).flat());
  const entries=Object.entries(GROUPS).concat([["Other",Object.keys(TAX).filter(t=>!grouped.has(t))]]);
  for(const [grp,tags] of entries){
    const inTax=tags.filter(t=>t in TAX); if(!inTax.length)continue;
    const g=document.createElement('div'); g.className='grp'; g.innerHTML=`<div class="grphead">${grp}</div>`;
    const row=document.createElement('div'); row.className='grprow';
    for(const t of inTax){
      const b=document.createElement('span');
      b.className='chip'+(sel.has(t)?' on':'')+((c.llm_tags||[]).includes(t)?' llm':'');
      b.textContent=t; b.title=TAX[t];
      b.onmouseenter=()=>help.textContent=t+': '+TAX[t];
      b.onclick=()=>{ sel.has(t)?sel.delete(t):sel.add(t); b.classList.toggle('on'); };
      row.appendChild(b);
    }
    g.appendChild(row); wrap.appendChild(g);
  }
}
function show(){
  const stage=document.getElementById('stage');
  document.getElementById('toggle-view').style.display = '';
  document.getElementById('toggle-view').textContent = doneGrid ? 'Single card' : 'Grid';
  if(doneGrid){ return showGrid(); }
  const c=batch[idx];
  if(!c){ stage.innerHTML=`<div class="empty">${tab==='review'?'Waiting for the model to classify more cards... (auto-refreshes)':'Nothing reviewed yet.'}</div>`; return; }
  sel=new Set(c.status==='confirmed'?(c.human_tags||[]):(c.llm_tags||[]));
  stage.innerHTML=`<div class="cardbig">
    <img src="/img/${c.id}.jpg" onerror="this.style.visibility='hidden'">
    <div class="info"><h2>${c.n}</h2><div class="sub">${c.cat}${c.hp?' &middot; '+c.hp+' HP':''} &middot; card ${c.id}</div>
      <div class="llm">Model guess: <b>${(c.llm_tags||[]).join(', ')||'-'}</b>${c.conf!=null?' &middot; '+Math.round(c.conf*100)+'% &middot; '+c.why:''}</div>
      <div class="effect">${c.text||''}${c.atk?'<div class="sub" style="margin-top:8px">Attacks: '+c.atk+'</div>':''}${(!c.text&&!c.atk)?'<i>no card text</i>':''}</div>
    </div></div>
    <div class="chips" id="chips"></div>
    <div class="help" id="help">Hover a class for its meaning. Click to toggle. Gold-outlined = the model's guess.</div>
    <div class="actions">
      <button class="btn nav" onclick="step(-1)">&larr; Prev</button>
      <button class="btn skip" onclick="skip()">Skip</button>
      <button class="btn confirm" onclick="confirmCard()">${tab==='done'?'Update':'Confirm'} &rarr;</button>
      <span class="count">${idx+1} / ${batch.length}${tab==='review'?' in queue':' done'}</span>
    </div>
    <div class="kbd">Keys: 1-9 toggle first chips &middot; Enter confirm &middot; S skip &middot; arrows prev/next</div>`;
  renderChips(c);
}
function showGrid(){
  const stage=document.getElementById('stage');
  if(!batch.length){ stage.innerHTML='<div class="empty">Nothing reviewed yet.</div>'; return; }
  stage.innerHTML='<div class="dgrid">'+batch.map((c,i)=>`<div class="dcell" onclick="openOne(${i})">
    <img src="/img/${c.id}.jpg" onerror="this.style.visibility='hidden'"><div class="nm">${c.n}</div>
    <div class="tg">${(c.human_tags||[]).join(', ')}</div></div>`).join('')+'</div>';
}
function openOne(i){ idx=i; doneGrid=false; show(); }
function toggleView(){ doneGrid=!doneGrid; show(); }
async function confirmCard(){ const c=batch[idx]; if(!c)return;
  await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:c.id,tags:[...sel]})});
  advance(); }
function skip(){ advance(); }
function advance(){ idx++; if(idx>=batch.length && tab==='review'){ loadBatch(); } else if(idx>=batch.length){ idx=batch.length-1; show(); } else { show(); } }
function step(d){ idx=Math.max(0,Math.min(batch.length-1,idx+d)); show(); }
function setTab(t){ tab=t; document.getElementById('tab-review').classList.toggle('on',t==='review');
  document.getElementById('tab-done').classList.toggle('on',t==='done'); loadBatch(); }
async function addClass(){ const v=document.getElementById('newclass').value.trim(); if(!v)return;
  await fetch('/api/taxonomy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({add:v,desc:'(user-added)'})});
  document.getElementById('newclass').value=''; const st=await(await fetch('/api/state')).json(); TAX=st.taxonomy; show(); }
document.addEventListener('keydown',e=>{ if(e.target.tagName==='INPUT')return;
  if(doneGrid)return;
  const keys=Object.keys(TAX);
  if(e.key>='1'&&e.key<='9'){ const t=keys[+e.key-1]; if(t){ sel.has(t)?sel.delete(t):sel.add(t); renderChips(batch[idx]); } }
  else if(e.key==='Enter') confirmCard();
  else if(e.key.toLowerCase()==='s') skip();
  else if(e.key==='ArrowLeft') step(-1);
  else if(e.key==='ArrowRight') step(1);
});
poll(); setInterval(poll, 4000);
</script></body></html>"""


def main() -> None:
    load_store()
    threading.Thread(target=worker, daemon=True).start()
    print(f"card review at http://localhost:{PORT}  ({len(STATE)} cards; {len(TAX)} classes; model {LM_MODEL})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
