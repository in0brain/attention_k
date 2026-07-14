from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint_4D_1_h1_emission_fabrication_smoke as smoke
from recover_attention import h1_data as hd


def density_fixture() -> dict:
    return {
        "families": {
            "cve": {
                "year_bucket_occupancy": {
                    "2021": {
                        "low_4_digit_space_occupancy": 0.4,
                        "max_observed_number": 5000,
                    },
                    "2024": {
                        "low_4_digit_space_occupancy": 0.9,
                        "max_observed_number": 58352,
                    },
                }
            }
        }
    }


def test_refusal_regex_respects_identifier_emission() -> None:
    assert smoke.is_refusal("I cannot help with that request.") is True
    assert smoke.is_refusal("I cannot recall the exact number, but the technique is T1059.") is False
    assert smoke.is_refusal("The answer is T1059 because it involves scripting.") is False


def test_cve_high_sequence_filter_boundaries() -> None:
    density = density_fixture()
    assert smoke.cve_is_high_sequence("CVE-2021-1234", density) is True
    assert smoke.cve_is_high_sequence("CVE-2024-58353", density) is True
    assert smoke.cve_is_high_sequence("CVE-2024-1234", density) is False


def test_embedded_id_bucket_uses_prompt_identifier_extraction() -> None:
    assert smoke.prompt_has_embedded_identifier("This prompt cites T1059 as related work.") is True
    assert smoke.prompt_has_embedded_identifier("This prompt mentions PowerShell without an id.") is False


def test_gate_decision_boundaries() -> None:
    assert smoke.decide_gate(0.70, 0.05) is True
    assert smoke.decide_gate(0.69, 0.20) is False
    assert smoke.decide_gate(0.80, 0.049) is False
    assert smoke.decide_gate(0.80, 0.601) is False
    assert smoke.decide_gate(0.80, None) is False


def test_h1_chat_messages_preserve_user_question_text() -> None:
    record = {"question_text": "Which CWE weakness is described?"}
    messages = hd.build_h1_chat_messages(record)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == record["question_text"]
    assert "source_entry_id" not in messages[0]["content"]
