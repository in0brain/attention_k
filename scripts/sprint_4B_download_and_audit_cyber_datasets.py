"""Download and audit candidate cyber MCQ datasets for Sprint 4B.

This script only downloads raw/source files and inspects their format. It does
not call a model, generate completions, convert to the Sprint 4B canonical
schema, run F5 baselines, train probes, or run site-transfer checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


USER_AGENT = "recover-attention-sprint-4b-dataset-audit"


CYBERMETRIC_FILES = [
    (
        "CyberMetric-500-v1.json",
        "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json",
    ),
    (
        "CyberMetric-2000-v1.json",
        "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-2000-v1.json",
    ),
    (
        "CyberMetric-10000-v1.json",
        "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-10000-v1.json",
    ),
    (
        "README.md",
        "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/README.md",
    ),
]


SECQA_FILES = [
    (
        "README.md",
        "https://huggingface.co/datasets/zefang-liu/secqa/raw/main/README.md",
    ),
    (
        "data/secqa_v1_dev.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v1_dev.csv",
    ),
    (
        "data/secqa_v1_val.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v1_val.csv",
    ),
    (
        "data/secqa_v1_test.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v1_test.csv",
    ),
    (
        "data/secqa_v2_dev.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v2_dev.csv",
    ),
    (
        "data/secqa_v2_val.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v2_val.csv",
    ),
    (
        "data/secqa_v2_test.csv",
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v2_test.csv",
    ),
]


CS_EVAL_FILES = [
    (
        "LICENSE",
        "https://raw.githubusercontent.com/CS-EVAL/CS-Eval/main/LICENSE",
    ),
    (
        "README.md",
        "https://raw.githubusercontent.com/CS-EVAL/CS-Eval/main/README.md",
    ),
    (
        "README_zh.md",
        "https://raw.githubusercontent.com/CS-EVAL/CS-Eval/main/README_zh.md",
    ),
    (
        "dataset_example.md",
        "https://raw.githubusercontent.com/CS-EVAL/CS-Eval/main/dataset_example.md",
    ),
    (
        "submission_example.json",
        "https://raw.githubusercontent.com/CS-EVAL/CS-Eval/main/submission_example.json",
    ),
]


DATASET_OUTPUTS = {
    "cybermetric": Path("data/raw/cyber/cybermetric"),
    "secqa": Path("data/raw/cyber/secqa"),
    "cs_eval": Path("data/raw/cyber/cs_eval"),
}


QUESTION_KEYS = ("question", "prompt", "instruction", "input", "query")
ANSWER_KEYS = (
    "answer",
    "answers",
    "correct_answer",
    "correct",
    "gold",
    "gold_answer",
    "label",
    "target",
    "output",
    "solution",
)
SEMANTIC_KEYS = (
    "category",
    "domain",
    "topic",
    "subject",
    "subdomain",
    "knowledge_point",
    "label_text",
    "label_id",
)
OPTION_CONTAINER_KEYS = ("choices", "options", "answers", "candidate_answers", "candidates")
OPTION_FIELD_PATTERNS = (
    re.compile(r"^[A-Da-d]$"),
    re.compile(r"^option[_ -]?[A-Da-d0-9]$"),
    re.compile(r"^choice[_ -]?[A-Da-d0-9]$"),
    re.compile(r"^answer[_ -]?[A-Da-d0-9]$"),
)


@dataclass
class DownloadResult:
    relative_path: str
    source_url: str
    status: str
    bytes: int = 0
    error: str | None = None


@dataclass
class DatasetAudit:
    dataset_name: str
    download_status: str = "skipped"
    source: str = ""
    license_note: str = ""
    usage_note: str = ""
    raw_paths: list[str] = field(default_factory=list)
    num_records: int = 0
    is_mcq: bool = False
    num_options_distribution: dict[str, int] = field(default_factory=dict)
    has_gold_answer: bool = False
    has_semantic_label_or_category: bool = False
    field_names: list[str] = field(default_factory=list)
    notes: str = ""
    fit_for_sprint_4b: str = "unknown"
    recommended_role: str = ""
    downloads: list[DownloadResult] = field(default_factory=list)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "download_status": self.download_status,
            "raw_paths": self.raw_paths,
            "num_records": self.num_records,
            "is_mcq": self.is_mcq,
            "num_options_distribution": self.num_options_distribution,
            "has_gold_answer": self.has_gold_answer,
            "field_names": self.field_names,
            "notes": self.notes,
        }


def request_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def download_file(url: str, dest: Path, overwrite: bool) -> DownloadResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rel = dest.as_posix()
    if dest.exists() and not overwrite:
        return DownloadResult(rel, url, "exists", bytes=dest.stat().st_size)
    try:
        data = request_url(url)
        dest.write_bytes(data)
        return DownloadResult(rel, url, "success", bytes=len(data))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return DownloadResult(rel, url, "failed", error=str(exc))


def load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "questions", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return [data]
    return []


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def field_names(records: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for record in records[:200]:
        fields.update(str(key) for key in record.keys())
    return sorted(fields)


def value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def count_options(record: dict[str, Any]) -> int:
    lower_record = {str(key).lower(): value for key, value in record.items()}
    for key in OPTION_CONTAINER_KEYS:
        value = lower_record.get(key.lower())
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        if isinstance(value, str) and value.strip():
            # Common text formats use newline- or delimiter-separated options.
            if "\n" in value:
                lines = [line for line in value.splitlines() if line.strip()]
                if len(lines) >= 2:
                    return len(lines)
            delimiter_count = max(value.count("|"), value.count(";"))
            if delimiter_count:
                return delimiter_count + 1

    count = 0
    for key, value in record.items():
        key_text = str(key).strip()
        if any(pattern.match(key_text) for pattern in OPTION_FIELD_PATTERNS):
            if value_is_present(value):
                count += 1
    return count


def has_gold(records: list[dict[str, Any]]) -> bool:
    for record in records[:200]:
        lower_record = {str(key).lower(): value for key, value in record.items()}
        for key in ANSWER_KEYS:
            value = lower_record.get(key.lower())
            # CyberMetric uses `answers` as the option dictionary, not the gold.
            if key == "answers" and isinstance(value, (dict, list)):
                continue
            if value_is_present(value):
                return True
    return False


def has_semantic(records: list[dict[str, Any]]) -> bool:
    for record in records[:200]:
        lower_record = {str(key).lower(): value for key, value in record.items()}
        for key in SEMANTIC_KEYS:
            if value_is_present(lower_record.get(key.lower())):
                return True
    return False


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    option_counts = Counter(str(count_options(record)) for record in records)
    option_counts.pop("0", None)
    num_records = len(records)
    mcq_like = bool(option_counts) and sum(option_counts.values()) / max(1, num_records) >= 0.8
    return {
        "num_records": num_records,
        "field_names": field_names(records),
        "num_options_distribution": dict(sorted(option_counts.items())),
        "is_mcq": mcq_like,
        "has_gold_answer": has_gold(records),
        "has_semantic_label_or_category": has_semantic(records),
    }


def compact_preview(dataset_name: str, source_path: Path, records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for index, record in enumerate(records[:limit]):
        keys = field_names([record])
        option_count = count_options(record)
        item = {
            "dataset_name": dataset_name,
            "source_path": source_path.as_posix(),
            "record_index": index,
            "field_names": keys,
            "num_options_detected": option_count,
            "has_gold_answer": has_gold([record]),
            "has_semantic_label_or_category": has_semantic([record]),
            "record_preview": {key: record.get(key) for key in keys[:12]},
        }
        preview.append(item)
    return preview


def inspect_json_files(dataset_name: str, files: list[Path], preview_limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    for path in files:
        try:
            records = load_json_records(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            per_file.append({"path": path.as_posix(), "error": str(exc)})
            continue
        summary = summarize_records(records)
        summary["path"] = path.as_posix()
        per_file.append(summary)
        all_records.extend(records)
        previews.extend(compact_preview(dataset_name, path, records, preview_limit))
    combined = summarize_records(all_records)
    combined["per_file"] = per_file
    return combined, previews


def inspect_csv_files(dataset_name: str, files: list[Path], preview_limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []
    for path in files:
        try:
            records = load_csv_records(path)
        except (csv.Error, UnicodeDecodeError, OSError) as exc:
            per_file.append({"path": path.as_posix(), "error": str(exc)})
            continue
        summary = summarize_records(records)
        summary["path"] = path.as_posix()
        per_file.append(summary)
        all_records.extend(records)
        previews.extend(compact_preview(dataset_name, path, records, preview_limit))
    combined = summarize_records(all_records)
    combined["per_file"] = per_file
    return combined, previews


def download_dataset_files(files: list[tuple[str, str]], out_dir: Path, overwrite: bool) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for rel_path, url in files:
        results.append(download_file(url, out_dir / rel_path, overwrite))
    return results


def successful_paths(results: list[DownloadResult], suffixes: tuple[str, ...]) -> list[Path]:
    paths = []
    for result in results:
        path = Path(result.relative_path)
        if result.status in {"success", "exists"} and path.suffix.lower() in suffixes:
            paths.append(path)
    return paths


def audit_cybermetric(raw_root: Path, overwrite: bool, preview_limit: int) -> tuple[DatasetAudit, list[dict[str, Any]]]:
    out_dir = raw_root / "cybermetric"
    downloads = download_dataset_files(CYBERMETRIC_FILES, out_dir, overwrite)
    json_paths = [path for path in successful_paths(downloads, (".json",)) if path.name.startswith("CyberMetric-")]
    summary, previews = inspect_json_files("cybermetric", json_paths, preview_limit)
    audit = DatasetAudit(
        dataset_name="cybermetric",
        source="https://github.com/cybermetric/CyberMetric",
        license_note="No explicit LICENSE file found in the downloaded repository root during this audit.",
        usage_note="Public CyberMetric benchmark files; verify citation/license expectations before publication use.",
        raw_paths=[result.relative_path for result in downloads if result.status in {"success", "exists"}],
        downloads=downloads,
        download_status="success" if json_paths else "failed",
        num_records=summary["num_records"],
        is_mcq=summary["is_mcq"],
        num_options_distribution=summary["num_options_distribution"],
        has_gold_answer=summary["has_gold_answer"],
        has_semantic_label_or_category=summary["has_semantic_label_or_category"],
        field_names=summary["field_names"],
        notes="Downloaded CyberMetric-500, CyberMetric-2000, CyberMetric-10000, and README when reachable.",
        fit_for_sprint_4b="yes" if summary["is_mcq"] and summary["has_gold_answer"] else "needs_review",
        recommended_role="primary" if summary["is_mcq"] and summary["has_gold_answer"] else "candidate",
    )
    audit.notes += f" Per-file summaries: {json.dumps(summary.get('per_file', []), ensure_ascii=False)}"
    return audit, previews


def audit_secqa(raw_root: Path, overwrite: bool, preview_limit: int) -> tuple[DatasetAudit, list[dict[str, Any]]]:
    out_dir = raw_root / "secqa"
    downloads = download_dataset_files(SECQA_FILES, out_dir, overwrite)
    csv_paths = successful_paths(downloads, (".csv",))
    summary, previews = inspect_csv_files("secqa", csv_paths, preview_limit)
    audit = DatasetAudit(
        dataset_name="secqa",
        source="https://huggingface.co/datasets/zefang-liu/secqa",
        license_note="Hugging Face dataset card reports license cc-by-nc-sa-4.0.",
        usage_note="Multiple-choice computer security QA; v1 and v2 splits are small (<1K total).",
        raw_paths=[result.relative_path for result in downloads if result.status in {"success", "exists"}],
        downloads=downloads,
        download_status="success" if csv_paths else "failed",
        num_records=summary["num_records"],
        is_mcq=summary["is_mcq"],
        num_options_distribution=summary["num_options_distribution"],
        has_gold_answer=summary["has_gold_answer"],
        has_semantic_label_or_category=summary["has_semantic_label_or_category"],
        field_names=summary["field_names"],
        notes="Downloaded secqa_v1/secqa_v2 dev/val/test CSV files and README when reachable.",
        fit_for_sprint_4b="yes_small" if summary["is_mcq"] and summary["has_gold_answer"] else "needs_review",
        recommended_role="fallback_or_held_out",
    )
    audit.notes += f" Per-file summaries: {json.dumps(summary.get('per_file', []), ensure_ascii=False)}"
    return audit, previews


def github_tree_manifest(repo_api_url: str, dest: Path) -> DownloadResult:
    return download_file(repo_api_url, dest, overwrite=True)


def audit_cs_eval(raw_root: Path, overwrite: bool, preview_limit: int) -> tuple[DatasetAudit, list[dict[str, Any]]]:
    out_dir = raw_root / "cs_eval"
    downloads = download_dataset_files(CS_EVAL_FILES, out_dir, overwrite)
    tree_result = github_tree_manifest(
        "https://api.github.com/repos/CS-EVAL/CS-Eval/git/trees/main?recursive=1",
        out_dir / "github_tree_manifest.json",
    )
    downloads.append(tree_result)

    records: list[dict[str, Any]] = []
    sample_json = out_dir / "submission_example.json"
    if sample_json.exists():
        try:
            records = load_json_records(sample_json)
        except json.JSONDecodeError:
            records = []
    summary = summarize_records(records)
    previews = compact_preview("cs_eval", sample_json, records, preview_limit) if records else []
    audit = DatasetAudit(
        dataset_name="cs_eval",
        source="https://github.com/CS-EVAL/CS-Eval",
        license_note="Repository includes an MIT LICENSE file.",
        usage_note="Repository is an evaluation suite; this task only downloaded source metadata/examples and did not convert data.",
        raw_paths=[result.relative_path for result in downloads if result.status in {"success", "exists"}],
        downloads=downloads,
        download_status="success" if any(result.status in {"success", "exists"} for result in downloads) else "failed",
        num_records=summary["num_records"],
        is_mcq=summary["is_mcq"],
        num_options_distribution=summary["num_options_distribution"],
        has_gold_answer=summary["has_gold_answer"],
        has_semantic_label_or_category=summary["has_semantic_label_or_category"],
        field_names=summary["field_names"],
        notes=(
            "Source inspection only. GitHub tree shows docs/examples/resources but no plain "
            "downloaded MCQ dataset file in the root metadata inspected here."
        ),
        fit_for_sprint_4b="not_primary_without_extra_download",
        recommended_role="source_inspection_only",
    )
    return audit, previews


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit_markdown(path: Path, audits: list[DatasetAudit]) -> None:
    lines = [
        "# Sprint 4B Dataset Download Audit",
        "",
        "Boundary: raw data download, source audit, format inspection, and small previews only. "
        "No model calls, completions, probe training, steering, F5 baseline, or site-transfer checks were run.",
        "",
        "## Summary",
        "",
        "| Dataset | Download | Records | MCQ | Option counts | Gold answer | Semantic/category | Fit | Role |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for audit in audits:
        lines.append(
            "| {name} | {status} | {records} | {mcq} | `{opts}` | {gold} | {semantic} | {fit} | {role} |".format(
                name=audit.dataset_name,
                status=audit.download_status,
                records=audit.num_records,
                mcq=str(audit.is_mcq).lower(),
                opts=json.dumps(audit.num_options_distribution, ensure_ascii=False),
                gold=str(audit.has_gold_answer).lower(),
                semantic=str(audit.has_semantic_label_or_category).lower(),
                fit=audit.fit_for_sprint_4b,
                role=audit.recommended_role,
            )
        )

    lines.extend(["", "## Dataset Details", ""])
    for audit in audits:
        lines.extend(
            [
                f"### {audit.dataset_name}",
                "",
                f"1. Download status: `{audit.download_status}`",
                f"2. Source: {audit.source}",
                f"3. License / usage note: {audit.license_note} {audit.usage_note}".strip(),
                f"4. Raw paths: `{json.dumps(audit.raw_paths, ensure_ascii=False)}`",
                f"5. Question / record count: `{audit.num_records}`",
                f"6. Is MCQ: `{str(audit.is_mcq).lower()}`",
                f"7. Options per question distribution: `{json.dumps(audit.num_options_distribution, ensure_ascii=False)}`",
                f"8. Has gold answer: `{str(audit.has_gold_answer).lower()}`",
                f"9. Has semantic label / category: `{str(audit.has_semantic_label_or_category).lower()}`",
                f"10. Field names: `{json.dumps(audit.field_names, ensure_ascii=False)}`",
                f"11. Fit for Sprint 4B: `{audit.fit_for_sprint_4b}`",
                f"12. Notes: {audit.notes}",
                "",
                "Download files:",
                "",
            ]
        )
        for result in audit.downloads:
            line = (
                f"- `{result.relative_path}` from {result.source_url}: "
                f"`{result.status}` ({result.bytes} bytes)"
            )
            if result.error:
                line += f"; error: {result.error}"
            lines.append(line)
        lines.append("")

    primary = next((audit for audit in audits if audit.recommended_role == "primary"), None)
    if primary:
        recommendation = primary.dataset_name
    else:
        viable = [audit.dataset_name for audit in audits if audit.is_mcq and audit.has_gold_answer]
        recommendation = viable[0] if viable else "none"

    lines.extend(
        [
            "## Recommendation",
            "",
            f"Recommended primary dataset: `{recommendation}`.",
            "",
            "Rationale: prefer CyberMetric if its raw JSON remains MCQ-like with gold answers and larger scale; "
            "use SecQA as a smaller fallback or held-out sanity source; keep CS-Eval as source-inspection-only "
            "until a plain downloadable MCQ file is identified.",
            "",
            "Manual download guidance if needed:",
            "",
            "- CyberMetric: place `CyberMetric-500-v1.json` and `CyberMetric-2000-v1.json` under `data/raw/cyber/cybermetric/`.",
            "- SecQA: place `secqa_v1_*` and `secqa_v2_*` CSV files under `data/raw/cyber/secqa/data/`.",
            "- CS-Eval: place any official released MCQ data under `data/raw/cyber/cs_eval/` and rerun this audit.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/cyber"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/logs/sprint_4B_dataset_download_audit"))
    parser.add_argument("--preview-per-file", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits: list[DatasetAudit] = []
    previews: list[dict[str, Any]] = []

    for audit_func in (audit_cybermetric, audit_secqa, audit_cs_eval):
        audit, dataset_previews = audit_func(args.raw_root, args.overwrite, args.preview_per_file)
        audits.append(audit)
        previews.extend(dataset_previews)
        print(
            f"{audit.dataset_name}: status={audit.download_status}, "
            f"records={audit.num_records}, mcq={audit.is_mcq}, gold={audit.has_gold_answer}"
        )

    manifest = {
        "task": "sprint_4B_dataset_download_audit",
        "boundary": "download_and_raw_format_inspection_only",
        "datasets": [audit.manifest_entry() for audit in audits],
        "downloads": {
            audit.dataset_name: [result.__dict__ for result in audit.downloads]
            for audit in audits
        },
    }
    write_json(args.output_dir / "raw_file_manifest.json", manifest)
    write_jsonl(args.output_dir / "sample_records_preview.jsonl", previews)
    write_audit_markdown(args.output_dir / "dataset_source_audit.md", audits)

    print(f"Wrote audit outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
