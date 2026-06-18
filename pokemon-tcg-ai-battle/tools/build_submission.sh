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
here is \`agent\`, delegating to agent_impl.$FN. Crash-safe (never raises).

NOTE: Kaggle exec()s this file WITHOUT __file__ defined, so do NOT reference __file__ or
os.path.abspath(__file__) at module scope (that caused an ERROR-status validation failure on
2026-06-17). The agent dir is importable on Kaggle (per the official sample); we also add the
known agent path explicitly. Imported modules (agent_impl, features, ...) DO have __file__, so
their own path handling is fine."""
import sys
if "/kaggle_simulations/agent" not in sys.path:
    sys.path.insert(0, "/kaggle_simulations/agent")
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
# Verify the way Kaggle loads it (exec without __file__). Needs a python with kaggle_environments
# (the repo venv). If this python lacks it, SKIP with a note instead of a misleading failure --
# run `PYTHON=<repo-venv-python> tools/build_submission.sh <variant>` to verify.
PYTHON="${PYTHON:-python}"
if "$PYTHON" -c "import kaggle_environments" 2>/dev/null; then
    "$PYTHON" tools/verify_submission.py "$D" || { echo "VERIFY FAILED -- do not submit $D"; exit 1; }
else
    echo "NOTE: '$PYTHON' lacks kaggle_environments; skipped verify. Re-run with PYTHON=<repo venv python>."
fi
