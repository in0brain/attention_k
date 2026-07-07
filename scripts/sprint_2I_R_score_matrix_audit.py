"""Sprint 2I-R: build score-matrix decomposition and root-cause audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.score_matrix_audit import BACKEND, run_score_matrix_audit  # noqa: E402


DEFAULT_2H_INSTANCE = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_instance_signal_500"
DEFAULT_2H_ENRICHED = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_feature_enrichment_500"
DEFAULT_2H_ORDINAL = PROJECT_ROOT / "outputs" / "logs" / "sprint_2H_ordinal_calibration_500"
DEFAULT_2I_ATTENTION = PROJECT_ROOT / "outputs" / "logs" / "sprint_2I_attention_features_500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current 2H/2I score components without rerunning upstream pipeline."
    )
    parser.add_argument(
        "--risk-strength",
        type=Path,
        default=DEFAULT_2H_INSTANCE / "risk_strength_dataset.jsonl",
    )
    parser.add_argument(
        "--pre-recovery-features",
        type=Path,
        default=DEFAULT_2H_ENRICHED / "pre_recovery_feature_dataset.jsonl",
    )
    parser.add_argument(
        "--ordinal-predictions",
        type=Path,
        default=DEFAULT_2H_ORDINAL / "ordinal_predictions.jsonl",
    )
    parser.add_argument(
        "--ordinal-report",
        type=Path,
        default=DEFAULT_2H_ORDINAL / "ordinal_calibration_report.json",
    )
    parser.add_argument(
        "--attention-features",
        type=Path,
        default=DEFAULT_2I_ATTENTION / "attention_feature_dataset.jsonl",
    )
    parser.add_argument(
        "--attention-report",
        type=Path,
        default=DEFAULT_2I_ATTENTION / "attention_ordinal_calibration_report.json",
    )
    parser.add_argument(
        "--enriched-predictions",
        type=Path,
        default=DEFAULT_2H_ENRICHED / "fragility_probe_enriched_predictions.jsonl",
    )
    parser.add_argument(
        "--hidden-predictions",
        type=Path,
        default=DEFAULT_2H_INSTANCE / "fragility_probe_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "sprint_2I_R_score_matrix_audit_500",
    )
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_score_matrix_audit(
        risk_strength_path=args.risk_strength,
        pre_recovery_feature_path=args.pre_recovery_features,
        ordinal_predictions_path=args.ordinal_predictions,
        ordinal_report_path=args.ordinal_report,
        attention_feature_path=args.attention_features,
        attention_report_path=args.attention_report,
        enriched_predictions_path=args.enriched_predictions,
        hidden_predictions_path=args.hidden_predictions,
        output_dir=args.output_dir,
        backend=args.backend,
        overwrite=args.overwrite,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
