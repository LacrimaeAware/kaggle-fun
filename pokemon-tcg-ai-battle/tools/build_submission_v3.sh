#!/usr/bin/env bash
# Build a self-contained cabt submission for a v3 agent (search_v3 + deck_policy_v3 + deck-out eval).
# Usage: tools/build_submission_v3.sh <variant>   where variant in {planner, phaware}
#   planner = PH KO floor + search (deck-out leaf, develop-first rollout)  -> agent_impl.agent_planner
#   phaware = PH KO floor + heuristic (no search)                          -> agent_impl.agent_phaware
# Output: submissions/sub_<variant>/ and submissions/sub_<variant>.tar.gz (single `agent` in main.py).
#
# Differs from build_submission.sh: bundles search_v3.py + deck_policy_v3.py + card_effects.json
# (the v3 file set), not the legacy search.py / value_model. Degrades to the heuristic if cg/time fails.
set -euo pipefail
V="${1:?variant required: planner|phaware|starmie}"
case "$V" in planner) FN="agent_planner";; phaware) FN="agent_phaware";; starmie) FN="agent_starmie";; *) echo "bad variant $V"; exit 2;; esac
cd "$(dirname "$0")/.."                 # repo: pokemon-tcg-ai-battle/
D="submissions/sub_$V"
rm -rf "$D"; mkdir -p "$D"
cp agent/main.py "$D/agent_impl.py"
cp agent/features.py agent/eval.py agent/search_v3.py agent/deck_policy_v3.py "$D/"
cp agent/card_features.json agent/card_stats.json agent/attack_stats.json agent/card_effects.json "$D/"
cp -r data/external/official/sample_submission/cg "$D/cg"
rm -rf "$D/cg/__pycache__"
cat > "$D/main.py" <<PY
"""Submission entry point. The Kaggle loader runs the LAST module-level callable; the only one here is
\`agent\`, delegating to agent_impl.$FN. Crash-safe (never raises).

NOTE: Kaggle exec()s this file WITHOUT __file__ defined, so do NOT reference __file__ at module scope
(that caused an ERROR-status validation failure on 2026-06-17). Imported modules DO have __file__, so
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
PYTHON="${PYTHON:-python}"
if "$PYTHON" -c "import kaggle_environments" 2>/dev/null; then
    "$PYTHON" tools/verify_submission.py "$D" || { echo "VERIFY FAILED -- do not submit $D"; exit 1; }
else
    echo "NOTE: '$PYTHON' lacks kaggle_environments; skipped verify. Re-run with PYTHON=<repo venv python>."
fi
