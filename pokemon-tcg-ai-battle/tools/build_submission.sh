#!/usr/bin/env bash
# Build a self-contained cabt submission tarball for a chosen agent variant.
# Usage: tools/build_submission.sh <variant>   where variant in {search, combine, search_v, agent}
# Output: submissions/sub_<variant>/ and submissions/sub_<variant>.tar.gz (main.py at the root).
#
# The submission's main.py exposes a single callable `agent` (the Kaggle loader runs the last
# module-level callable) delegating to agent_impl.agent_<variant>. It bundles the runtime modules,
# the value/feature data, and the cg forward-model engine so search runs on Kaggle. The agent
# degrades to the heuristic if cg or the time budget fails, so it never forfeits.
set -euo pipefail
V="${1:?variant required: search|combine|search_v|agent}"
cd "$(dirname "$0")/.."                 # repo: pokemon-tcg-ai-battle/
FN="agent_$V"; [ "$V" = "agent" ] && FN="agent"
D="submissions/sub_$V"
rm -rf "$D"; mkdir -p "$D"
cp agent/main.py "$D/agent_impl.py"
cp agent/search.py agent/eval.py agent/value_model.py agent/features.py "$D/"
cp agent/card_features.json agent/value_weights.json agent/card_stats.json agent/attack_stats.json "$D/"
cp -r data/external/official/sample_submission/cg "$D/cg"
rm -rf "$D/cg/__pycache__"
cat > "$D/main.py" <<PY
"""Submission entry point. The Kaggle loader runs the last module-level callable; the only one
here is \`agent\`, delegating to agent_impl.$FN. Crash-safe (never raises)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_impl


def agent(obs):
    try:
        return agent_impl.$FN(obs)
    except Exception:
        sel = obs.get("select")
        if sel is None:
            return list(agent_impl.DECK)
        opts = (sel or {}).get("option") or []
        k = (sel or {}).get("minCount") or 1
        return list(range(min(max(k, 1), len(opts)))) if opts else []
PY
tar -czf "submissions/sub_$V.tar.gz" -C "$D" .
echo "built submissions/sub_$V.tar.gz ($(du -h "submissions/sub_$V.tar.gz" | cut -f1))"
