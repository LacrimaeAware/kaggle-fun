from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SDK_PATH = ROOT / "data" / "competition"


def ensure_sdk_path() -> None:
    if (SDK_PATH / "aicomp_sdk").is_dir() and str(SDK_PATH) not in sys.path:
        sys.path.insert(0, str(SDK_PATH))


def load_attack_class(path: Path) -> type[Any]:
    spec = importlib.util.spec_from_file_location("local_attack_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load attack module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    attack_cls = getattr(module, "AttackAlgorithm", None)
    if attack_cls is None:
        raise RuntimeError(f"{path} does not define AttackAlgorithm")
    return attack_cls


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local red-team smoke evaluation.")
    parser.add_argument("--attack", default=str(ROOT / "attack.py"))
    parser.add_argument("--budget-s", type=float, default=10.0)
    parser.add_argument("--agent", default="deterministic")
    parser.add_argument("--env", default="sandbox")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "results" / "local_eval"))
    parser.add_argument("--fixtures-dir", default=str(SDK_PATH / "aicomp_sdk" / "fixtures"))
    args = parser.parse_args()

    ensure_sdk_path()

    from aicomp_sdk.agents import AgentSelection
    from aicomp_sdk.core.env.api import EnvSelection
    from aicomp_sdk.evaluation.reports import ReportProfile, build_evaluation_report
    from aicomp_sdk.evaluation.runner import evaluate_redteam

    attack_cls = load_attack_class(Path(args.attack).resolve())
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fixtures_dir = Path(args.fixtures_dir).resolve()
    if not fixtures_dir.is_dir():
        fixtures_dir = None

    execution = evaluate_redteam(
        attack_cls,
        budget_s=args.budget_s,
        agent_selection=AgentSelection(args.agent),
        env_selection=EnvSelection(args.env),
        fixtures_dir=fixtures_dir,
    )
    report = build_evaluation_report(execution, profile=ReportProfile.EVALUATE)

    (artifacts_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (artifacts_dir / "score.txt").write_text(str(report["final_score"]), encoding="utf-8")

    attack = report.get("attack", {})
    print(f"score={report['final_score']}")
    print(f"raw={attack.get('score_raw')}")
    print(f"findings={attack.get('findings_count')}")
    print(f"unique_cells={attack.get('unique_cells')}")
    print(f"artifacts={artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
