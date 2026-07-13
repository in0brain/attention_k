from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recover_attention.data_io import ensure_dir, write_json, write_jsonl, write_text  # noqa: E402
from recover_attention.h1_data import (  # noqa: E402
    audit_h1_samples,
    build_h1_samples,
    estimate_id_space_density,
)
from recover_attention.h1_identifier import build_ontology_index  # noqa: E402
from recover_attention.schemas import validate_h1_sample_record  # noqa: E402


def f5_design_text() -> str:
    return """# H1 F5 Kill-Baseline Design

This is a design artifact only. Sprint 4D-0 does not call a model, generate
completions, compute F5, train a probe, or run steering.

## Why MCQ F5 Does Not Transfer Directly

H1 is free generation over exact identifier strings, not a finite option-letter
space. MCQ label margin and label entropy cannot be reused as-is because there
is no closed candidate set at every answer step.

## Candidate H1 F5 Baselines

Mention-level:
- mean and minimum logprob over the emitted identifier span;
- rank of the first identifier token among top-k next-token candidates;
- exact-string parse confidence, counted separately from ontology existence.

Sequence-level:
- completion perplexity;
- length-normalized completion logprob;
- ratio of identifier tokens to total completion tokens as a verbosity control.

Sampling-level:
- K-sample exact identifier agreement for the same requested slot;
- self-consistency over exact strings, not semantic clusters;
- semantic-entropy-style clustering is mostly degenerate here because the label
  is an exact id string, so exact-match and normalized-id match are primary.

Verbalized confidence:
- extract explicit confidence phrases with a fixed regex/protocol;
- keep verbalized confidence separate from ontology labels and never use gold ids
  as inference features.

## Kill Gate

F5 is expected to be weaker on H1 than on calibrated MCQ, which is one reason H1
is the next vehicle. The discipline is unchanged: any future internal feature
family F1-F4 must report grouped-CV and CI incremental value over H1-F5.

## Sprint 4D-1 Preregistered Smoke Gate

4D-1 should measure only:
- id emission rate: fraction of completions containing a legal-format id;
- fabrication base rate: mention-level and completion-level.

Preregistered feasibility gate:
- Route A emission rate >= 0.7;
- fabrication base rate in [0.05, 0.60].

If the gate fails, return to prompt/data design. Do not proceed directly to
feature engineering.
"""


def review_gate_text(audit: dict, density: dict) -> str:
    route_counts = audit["route_counts"]
    family_counts = audit["family_counts"]
    return f"""# Review Gate: Sprint 4D-0 H1 Data Design

1. Three ontology snapshots have dated manifest/id counts/sha256: see
   `ontology_snapshot_manifest.json`.
2. RESERVED/revoked/deprecated rules: existence-positive, status counted
   separately; fabrication means absent from the complete snapshot index.
3. ID-space density conclusion: CVE is weaker in dense low-number ranges.
   Strategy: ATT&CK+CWE are primary; CVE is auxiliary/high-number-sensitive.
   Labeling strategy: `{density['labeling_strategy']['cve_policy']}`.
4. Extraction negative tests cover version numbers, year ranges, ports, hash
   fragments, T-shirt/T5 ordinary text, and non-id acronyms; targeted tests pass.
5. Echo exclusion is normalized-id matching against the prompt. Question
   gold-id residual assertion is full-dataset and passed.
6. Route counts: {route_counts}; family counts: {family_counts}.
7. group_id leakage units: recall = ontology entry; open_gen = topic entity.
   group leakage count = {audit['group_leakage_count']}.
8. `h1_f5_design.md` covers mention/sequence/sampling/verbalized-confidence
   baselines and explains why H1-F5 is expected to be weaker than MCQ-F5.
9. 4D-1 emission/fabrication preregistered gates are written in
   `h1_f5_design.md`.
10. causal LM called / completions generated / probe trained: no / no / no.
11. Gold id is eval-only; H1 leakage tests cover feature-record shape: yes.
12. No claim is made about hallucination reduction, accuracy improvement,
    detection performance, or H1 emission viability.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology-dir", type=Path, default=Path("data/raw/ontology"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/h1/h1_samples.jsonl"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/logs/sprint_4D_0_h1_data_design"))
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    ensure_dir(args.report_dir)
    index = build_ontology_index(args.ontology_dir)
    records = build_h1_samples(index, seed=args.seed)
    for record in records:
        validate_h1_sample_record(record)
    audit = audit_h1_samples(records)
    density = estimate_id_space_density(index)
    write_jsonl(records, args.output_path)
    write_json(density, args.report_dir / "id_space_density_report.json")
    write_json(audit, args.report_dir / "h1_dataset_audit_report.json")
    write_text(f5_design_text(), args.report_dir / "h1_f5_design.md")
    write_text(review_gate_text(audit, density), args.report_dir / "review_gate_h1_data_design.md")
    print("wrote H1 samples:", len(records), "to", args.output_path)
    print("audit:", args.report_dir / "h1_dataset_audit_report.json")


if __name__ == "__main__":
    main()
