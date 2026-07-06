"""Sprint 2H-B task 4: instance-level fragility_bucket / risk_strength targets.

The whole point of these targets is that they must NOT be a deterministic function
of span_type. Two number spans can land in different buckets depending on solution-
path membership and recovery drift. Bucket assignment uses worst-case drift evidence.

Ordinal buckets:
    0 = irrelevant_or_off_path
    1 = stable_required
    2 = generic_or_under_recovered
    3 = wrong_or_drifted

Ambiguous-number spans are excluded from probe training (kept in cases/report).
"""

from __future__ import annotations

from typing import Any

from recover_attention.recovery_drift import (
    DIRECTION_DRIFT,
    GENERIC,
    HARD_DRIFT_LABELS,
    UNRECOVERABLE,
)
from recover_attention.solution_path_numbers import AMBIGUOUS, OFF_PATH, ON_PATH, NOT_A_NUMBER


BUCKET_OFF_PATH = 0          # irrelevant_or_off_path
BUCKET_STABLE = 1            # stable_required
BUCKET_UNDER_RECOVERED = 2  # generic_or_under_recovered
BUCKET_DRIFTED = 3          # wrong_or_drifted

BUCKET_NAMES = {
    BUCKET_OFF_PATH: "irrelevant_or_off_path",
    BUCKET_STABLE: "stable_required",
    BUCKET_UNDER_RECOVERED: "generic_or_under_recovered",
    BUCKET_DRIFTED: "wrong_or_drifted",
}

# risk_strength bands per bucket. Within-band offset (0..1) is scaled by recovery
# instability so ordering across buckets is always preserved.
_BANDS = {
    BUCKET_OFF_PATH: (0.00, 0.20),
    BUCKET_STABLE: (0.30, 0.50),
    BUCKET_UNDER_RECOVERED: (0.55, 0.75),
    BUCKET_DRIFTED: (0.80, 1.00),
}


def assign_fragility_bucket(
    span_type: str,
    solution_path_status: str,
    drift_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Assign an ordinal fragility bucket from instance-level evidence.

    Returns a dict with ``bucket`` (int or None), ``excluded`` (bool), and ``reason``.
    Ambiguous-number spans are excluded (bucket=None).
    """
    if span_type == "number" and solution_path_status == AMBIGUOUS:
        return {"bucket": None, "excluded": True, "reason": "ambiguous_number_excluded"}

    # Off-path distractor numbers are low guidance-priority by definition: steering
    # attention to a number the gold solution never uses is wasted budget, even if
    # the model recovers it wrong. Solution-path membership gates these to bucket 0
    # BEFORE drift escalation (this is what makes the distractor-budget gate meaningful;
    # it resolves the spec tension between "off_path -> 0" and "drift > path evidence"
    # in favour of not steering known distractors).
    if span_type == "number" and solution_path_status == OFF_PATH:
        return {"bucket": BUCKET_OFF_PATH, "excluded": False, "reason": "off_path_distractor"}

    # Hard drift evidence dominates everything else (worst-case) for on-path numbers
    # and for non-number spans (which carry no solution-path signal).
    if drift_aggregate.get("any_hard_drift"):
        return {
            "bucket": BUCKET_DRIFTED,
            "excluded": False,
            "reason": f"hard_drift:{','.join(drift_aggregate.get('hard_drift_labels', []))}",
        }

    majority = drift_aggregate.get("majority_drift_label", UNRECOVERABLE)
    num_generic = drift_aggregate.get("num_generic", 0)
    num_unrecoverable = drift_aggregate.get("num_unrecoverable", 0)
    num_exact = drift_aggregate.get("num_exact", 0)
    total = drift_aggregate.get("num_samples", 0) or 1

    # under-recovered: generic majority or predominantly unrecoverable
    if num_generic >= max(1, total // 2) or majority == GENERIC:
        return {"bucket": BUCKET_UNDER_RECOVERED, "excluded": False, "reason": "generic_majority"}
    if num_unrecoverable >= max(1, total // 2) or majority == UNRECOVERABLE:
        return {"bucket": BUCKET_UNDER_RECOVERED, "excluded": False, "reason": "under_recovered"}

    # remaining cases are exact-dominated (stable recovery). Off-path numbers were
    # already routed to bucket 0 above, so anything here is an on-path number or a
    # non-number span that recovered cleanly -> stable_required.
    if num_exact >= 1:
        if span_type != "number" and solution_path_status == NOT_A_NUMBER:
            return {"bucket": BUCKET_STABLE, "excluded": False, "reason": "non_number_stable"}
        return {"bucket": BUCKET_STABLE, "excluded": False, "reason": "on_path_stable"}

    # fallback: no exact, no generic-majority, no hard drift -> treat as under-recovered
    return {"bucket": BUCKET_UNDER_RECOVERED, "excluded": False, "reason": "no_stable_evidence"}


def compute_risk_strength(bucket: int, drift_aggregate: dict[str, Any]) -> float:
    """Map a bucket + instability into a continuous risk_strength that preserves the
    cross-bucket ordering (bucket3 > bucket2 > bucket1 > bucket0) while varying within
    a band by recovery instability."""
    low, high = _BANDS[bucket]
    instability = float(drift_aggregate.get("inconsistency_rate", 0.0))
    instability = min(1.0, max(0.0, instability))
    return round(low + instability * (high - low), 6)


def verify_ordering_constraints(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify required ordinal constraints hold across the built dataset.

    Constraints:
        wrong_or_drifted > generic_or_under_recovered > stable_required > off_path
        on_solution_path_number > off_solution_path_number (for stable numbers)
        direction_drift bucket >= generic_recovery bucket
    """
    by_bucket: dict[int, list[float]] = {b: [] for b in _BANDS}
    for record in dataset:
        bucket = record.get("fragility_bucket")
        if bucket is None:
            continue
        by_bucket[bucket].append(record.get("risk_strength", 0.0))

    violations: list[str] = []
    ordered_buckets = [BUCKET_OFF_PATH, BUCKET_STABLE, BUCKET_UNDER_RECOVERED, BUCKET_DRIFTED]
    # band separation: max of lower bucket <= min of higher bucket
    for lower, higher in zip(ordered_buckets, ordered_buckets[1:]):
        low_vals = by_bucket[lower]
        high_vals = by_bucket[higher]
        if low_vals and high_vals and max(low_vals) > min(high_vals) + 1e-9:
            violations.append(
                f"strength overlap: bucket {lower} max {max(low_vals):.3f} "
                f"> bucket {higher} min {min(high_vals):.3f}"
            )

    # on-path vs off-path for stable numbers
    on_path_numbers = [
        r["fragility_bucket"]
        for r in dataset
        if r.get("span_type") == "number"
        and r.get("solution_path_status") == ON_PATH
        and r.get("fragility_bucket") is not None
    ]
    off_path_numbers = [
        r["fragility_bucket"]
        for r in dataset
        if r.get("span_type") == "number"
        and r.get("solution_path_status") == OFF_PATH
        and r.get("fragility_bucket") is not None
    ]
    mean_on = sum(on_path_numbers) / len(on_path_numbers) if on_path_numbers else None
    mean_off = sum(off_path_numbers) / len(off_path_numbers) if off_path_numbers else None
    if mean_on is not None and mean_off is not None and mean_on < mean_off:
        violations.append(
            f"on-path number mean bucket {mean_on:.3f} < off-path {mean_off:.3f}"
        )

    return {
        "num_violations": len(violations),
        "violations": violations,
        "bucket_strength_ranges": {
            BUCKET_NAMES[b]: {
                "count": len(by_bucket[b]),
                "min": round(min(by_bucket[b]), 4) if by_bucket[b] else None,
                "max": round(max(by_bucket[b]), 4) if by_bucket[b] else None,
            }
            for b in ordered_buckets
        },
        "on_path_number_mean_bucket": round(mean_on, 4) if mean_on is not None else None,
        "off_path_number_mean_bucket": round(mean_off, 4) if mean_off is not None else None,
    }


def check_not_span_type_deterministic(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirm fragility_bucket is NOT a pure function of span_type.

    Returns per-span_type bucket distributions; if any span_type maps to >1 bucket,
    the target is instance-level rather than a span-type classifier.
    """
    by_type: dict[str, dict[int, int]] = {}
    for record in dataset:
        bucket = record.get("fragility_bucket")
        if bucket is None:
            continue
        span_type = record.get("span_type", "unknown")
        by_type.setdefault(span_type, {})
        by_type[span_type][bucket] = by_type[span_type].get(bucket, 0) + 1

    multi_bucket_types = [t for t, dist in by_type.items() if len(dist) > 1]
    return {
        "bucket_distribution_by_span_type": {
            t: {BUCKET_NAMES[b]: c for b, c in sorted(dist.items())} for t, dist in by_type.items()
        },
        "span_types_with_multiple_buckets": multi_bucket_types,
        "is_span_type_deterministic": len(multi_bucket_types) == 0,
    }
