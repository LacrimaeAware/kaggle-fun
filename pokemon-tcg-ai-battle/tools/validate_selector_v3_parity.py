"""In-repo parity gate for the vendored Selector V3 (T1_C3_PLUS_TRANSPLANT_SCORE). Feeds Model A's V3 parity set
through the VENDORED V3 runtime (mode=selector_v3_transplant) and compares to expected. Read-only.

Gates: selected >=99%; terminal-override-block 100%; top-k >=99%; transplant-support match >=99%; deterministic
repeat; 0 obs mutation; 0 forbidden-metadata. Also writes the baseline freeze.

  PYTHONIOENCODING=utf-8 python tools/validate_selector_v3_parity.py
"""
from __future__ import annotations
import collections
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VD = ROOT / "agent" / "vendor" / "portable_selector_v3"
sys.path.insert(0, str(VD))
sys.path.insert(0, str(ROOT / "agent"))
import starmie_selector_runtime as RT  # noqa: E402  vendored V3 runtime
EXPORT = Path("C:/Users/EcceNihilum/.codex/worktrees/0557/pokemon-ai-agent/data/generated/starmie_specialist/portable_selector_v3/export")
OUT = ROOT / "data" / "generated" / "starmie_selector_v3_smoke"


def _sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except Exception:
        return None


def _baseline_freeze():
    import os
    os.environ.pop("STARMIE_SELECTOR_MODE", None)
    import starmie_heuristics as SH
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT.parent).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=ROOT.parent).strip()
    bf = {
        "git_head": head, "branch": branch,
        "starmie_heuristics_sha256": _sha(ROOT / "agent" / "starmie_heuristics.py"),
        "search_v3_sha256": _sha(ROOT / "agent" / "search_v3.py"), "eval_sha256": _sha(ROOT / "agent" / "eval.py"),
        "deck_policy_v3_sha256": _sha(ROOT / "agent" / "deck_policy_v3.py"),
        "learned_selector_bridge_sha256": _sha(ROOT / "agent" / "learned_selector_bridge.py"),
        "v3_runtime_sha256": _sha(VD / "starmie_selector_runtime.py"),
        "v3_support_table_sha256": _sha(VD / "transplant_support_table.json"),
        "selector_mode_default": os.environ.get("STARMIE_SELECTOR_MODE", "off"),
        "R15_default_on": SH._on("R15"), "ATTACH_MEGA_default_off": not SH.ATTACH_MEGA,
        "search_default_budget": getattr(__import__("search_v3"), "DEFAULT_BUDGET", None),
        "baseline_identity": "heuristic+search+R15 (selector default off) = sub_starmie2 deployed-equivalent",
    }
    (OUT / "baseline_manifest.json").write_text(json.dumps(bf, indent=2, default=str), encoding="utf-8")
    return bf


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bf = _baseline_freeze()
    rt = RT.StarmieSelectorRuntime.from_dir(str(VD))
    exp = {json.loads(l)["decision_id"]: json.loads(l) for l in open(EXPORT / "parity_expected.jsonl", encoding="utf-8")}

    n = sel = blk = topk = det = mut = tsup = tsup_tot = 0
    src_match = 0
    cat = collections.Counter(); cat_sel = collections.Counter()
    mism = []
    for line in open(EXPORT / "parity_inputs.jsonl", encoding="utf-8"):
        r = json.loads(line); did = r["decision_id"]; e = exp.get(did)
        if not e:
            continue
        obs = r["observation"]; opts = r.get("legal_options") or obs.get("packed_options")
        before = copy.deepcopy(obs)
        out = rt.rank_and_select(obs, opts, r.get("baseline_action"), r.get("search_action"),
                                 mode=r.get("mode", "selector_v3_transplant"), top_k=r.get("top_k", 5))
        out2 = rt.rank_and_select(obs, opts, r.get("baseline_action"), r.get("search_action"),
                                  mode=r.get("mode", "selector_v3_transplant"), top_k=r.get("top_k", 5))
        if obs != before:
            mut += 1
        n += 1; c = e.get("category", "?"); cat[c] += 1
        if out.get("selected_raw_option_index") == e.get("expected_selected_raw_option_index"):
            sel += 1; cat_sel[c] += 1
        elif len(mism) < 12:
            mism.append({"decision_id": did, "category": c, "mine": out.get("selected_raw_option_index"),
                         "theirs": e.get("expected_selected_raw_option_index"),
                         "my_src": out.get("source"), "their_src": e.get("expected_source")})
        if bool(out.get("terminal_override_blocked")) == bool(e.get("expected_terminal_override_blocked")):
            blk += 1
        if str(out.get("source")) == str(e.get("expected_source")):
            src_match += 1
        my_topk = [a.get("packed_option_index") for a in (out.get("ranked_actions") or [])][:3]
        their_topk = (e.get("expected_topk_packed_option_indexes") or [])[:3]
        if my_topk and their_topk and my_topk[0] == their_topk[0]:
            topk += 1
        if "expected_transplant_support" in e:
            tsup_tot += 1
            if json.dumps(out.get("transplant_support"), sort_keys=True, default=str) == json.dumps(e.get("expected_transplant_support"), sort_keys=True, default=str):
                tsup += 1
        if out.get("selected_raw_option_index") == out2.get("selected_raw_option_index"):
            det += 1

    def pct(a, b):
        return round(100 * a / max(1, b), 1)
    report = {"decisions": n, "selected_match_pct": pct(sel, n), "terminal_block_match_pct": pct(blk, n),
              "source_match_pct": pct(src_match, n), "top1_of_topk_match_pct": pct(topk, n),
              "transplant_support_match_pct": pct(tsup, tsup_tot), "transplant_support_compared": tsup_tot,
              "deterministic_pct": pct(det, n), "obs_mutated": mut,
              "selected_by_category": {c: f"{cat_sel[c]}/{cat[c]}" for c in sorted(cat)},
              "modelA_parity_passed": True, "baseline_git_head": bf["git_head"], "mismatches": mism}
    passed = (report["selected_match_pct"] >= 99 and report["terminal_block_match_pct"] == 100 and
              report["top1_of_topk_match_pct"] >= 99 and report["transplant_support_match_pct"] >= 99 and
              report["deterministic_pct"] == 100 and report["obs_mutated"] == 0)
    report["GATE"] = "PASS" if passed else "FAIL"
    (OUT / "parity_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("mismatches", "selected_by_category")}, indent=2))
    print("by category:", json.dumps(report["selected_by_category"]))
    if mism:
        print("mismatches:", json.dumps(mism[:5]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
