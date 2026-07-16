from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recover_attention import h1_f5_features as f5


def test_token_indices_for_char_span_handles_middle_and_boundary_mentions() -> None:
    offsets = [(0, 3), (3, 4), (4, 8), (9, 14), (14, 18)]
    assert f5.token_indices_for_char_span(offsets, 4, 8) == [2]
    assert f5.token_indices_for_char_span(offsets, 9, 18) == [3, 4]
    assert f5.token_indices_for_char_span(offsets, 3, 4) == [1]


def test_mention_and_sequence_logprob_features() -> None:
    features = f5.mention_logprob_features(
        {2: -0.25, 3: -1.25, 4: -0.50},
        {2: 0.4, 3: 0.8, 4: 0.6},
        {2: 1, 3: 12},
        [2, 3],
    )
    assert features["f5_id_token_count"] == 2
    assert features["f5_id_logprob_mean"] == -0.75
    assert features["f5_id_logprob_min"] == -1.25
    assert features["f5_first_id_token_rank"] == 1
    assert features["f5_id_token_entropy_mean"] == 0.6000000000000001

    sequence = f5.sequence_logprob_features([-0.1, -0.2, -0.3, -0.4], id_token_count=2)
    assert sequence["f5_lengthnorm_logprob"] == -0.25
    assert math.isclose(sequence["f5_completion_perplexity"], math.exp(0.25))
    assert sequence["f5_id_token_ratio"] == 0.5


def test_exact_consistency_and_id_agreement_boundaries() -> None:
    assert f5.exact_consistency([])["f5_self_consistency_exact"] is None
    result = f5.exact_consistency(["T1059", "T1059", "CWE-79"])
    assert result["mode_value"] == "T1059"
    assert result["f5_self_consistency_exact"] == 2 / 3
    assert f5.id_agreement_rate("T1059", ["T1059", "CWE-79", "T1059"]) == 2 / 3
    assert f5.id_agreement_rate("T1059", []) is None


def test_primary_detection_excludes_echo_and_cve() -> None:
    assert f5.included_in_primary_detection({"mention_family": "attack", "label": "fabricated"})
    assert f5.included_in_primary_detection({"mention_family": "cwe", "label": "grounded"})
    assert not f5.included_in_primary_detection({"mention_family": "cve", "label": "fabricated"})
    assert not f5.included_in_primary_detection({"mention_family": "attack", "label": "echoed"})


def test_confidence_and_list_false_positive_helpers() -> None:
    confidence = f5.extract_verbalized_confidence("This is likely CWE-79, but I am not sure.")
    assert confidence["f5_confidence_medium"] == 1
    assert confidence["f5_confidence_low"] == 1
    assert f5.looks_like_structured_identifier_list(
        "1. T1059 - scripting\n2. T1105 - ingress\n3. CWE-79 - xss\n4. CWE-89 - sql"
    )
    assert not f5.looks_like_structured_identifier_list("!!!!!!!!!!!!")
