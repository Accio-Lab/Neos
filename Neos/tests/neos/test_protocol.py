
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neos.workflow.protocol import (
    ASSISTANT_FORMAT_PROFILES,
    MAX_ASSISTANT_TURNS,
    NEOS_PROTOCOL_VERSION,
    NEOS_SYSTEM_PROMPT,
    NATIVE_ASSISTANT_FORMAT_PROFILE,
    QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE,
    parse_neos_assistant_message,
    protocol_version_for_turn_limit,
    system_prompt_for_turn_limit,
)


def _call(name: str, arguments: dict, call_id: str = "call_001") -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "Select the next evidence-gathering action.",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def test_native_and_qwen_compat_profiles_are_active() -> None:
    assert ASSISTANT_FORMAT_PROFILES == (
        NATIVE_ASSISTANT_FORMAT_PROFILE,
        QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE,
    )
    assert NATIVE_ASSISTANT_FORMAT_PROFILE == "openai_native"


def test_native_tool_call_is_structured_and_schema_checked() -> None:
    parsed = parse_neos_assistant_message(
        _call("TextSearch", {"query": ["one", "two"]})
    )
    assert parsed.valid, parsed.error
    assert parsed.action_kind == "tool_call"
    assert parsed.tool_call == {
        "id": "call_001",
        "name": "TextSearch",
        "arguments": {"query": ["one", "two"]},
    }

    invalid = parse_neos_assistant_message(
        _call("TextSearch", {"invented": True})
    )
    assert not invalid.valid
    assert "invalid TextSearch arguments" in str(invalid.error)


def test_parallel_unknown_and_unidentified_calls_are_rejected() -> None:
    message = _call("TextSearch", {"query": "one"})
    message["tool_calls"].append(message["tool_calls"][0])
    assert not parse_neos_assistant_message(message).valid
    assert not parse_neos_assistant_message(
        _call("ocr", {"image": "img_1"})
    ).valid
    assert not parse_neos_assistant_message(
        _call("TextSearch", {"query": "one"}, call_id="")
    ).valid


def test_final_answer_is_plain_content_and_serialization_leak_is_rejected() -> None:
    final = parse_neos_assistant_message(
        {
            "role": "assistant",
            "reasoning_content": "The sources agree.",
            "content": "Example",
        }
    )
    assert final.valid
    assert final.action_kind == "answer"
    assert final.answer == "Example"

    textual_tool = parse_neos_assistant_message(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"TextSearch","arguments":'
                '{"query":"one"}}</tool_call>'
            ),
        }
    )
    assert not textual_tool.valid
    assert "reserved serialization token" in str(textual_tool.error)


def test_policy_describes_semantics_without_teaching_serialization_tags() -> None:
    assert "native function call" in NEOS_SYSTEM_PROMPT
    assert "role=tool" in NEOS_SYSTEM_PROMPT
    assert "<think>" not in NEOS_SYSTEM_PROMPT
    assert "<tool_call>" not in NEOS_SYSTEM_PROMPT
    assert "<answer>" not in NEOS_SYSTEM_PROMPT
    assert "You are Neos" not in NEOS_SYSTEM_PROMPT


def test_canonical_turn_limit() -> None:
    assert MAX_ASSISTANT_TURNS == 50
    assert "within 50 assistant turns." in NEOS_SYSTEM_PROMPT
    assert system_prompt_for_turn_limit(50) == NEOS_SYSTEM_PROMPT
    assert protocol_version_for_turn_limit(50) == NEOS_PROTOCOL_VERSION


@pytest.mark.parametrize("turn_limit", [20, 32])
def test_unsupported_turn_limits(turn_limit) -> None:
    with pytest.raises(ValueError):
        system_prompt_for_turn_limit(turn_limit)
    with pytest.raises(ValueError):
        protocol_version_for_turn_limit(turn_limit)


def test_current_launchers_use_official_family_parsers_without_template_override() -> None:
    from types import SimpleNamespace
    from neos.serving.service import commands
    args=SimpleNamespace(kind='policy',gpu_groups='0,1;2,3',model='models/policy',
        name='policy',backend_port=62000,context_length=262144,max_running_requests=16,memory_fraction=.6)
    for _,command in commands(args):
        assert command[command.index('--tool-call-parser')+1]=='qwen3_coder'
        assert command[command.index('--reasoning-parser')+1]=='qwen3'
        assert '--chat-template' not in command


def test_active_runtime_does_not_import_archived_regex_adapter() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "neos/workflow/agent.py",
        "neos/workflow/eval.py",
        "neos/workflow/models.py",
        "neos/workflow/protocol.py",
    ):
        source = (root / relative).read_text()
        assert "from neos.workflow.assistant_format" not in source


def test_rl_launcher_uses_official_qwen35_parser() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/run_rl_backend.sh").read_text()
    assert "multi_turn.format=qwen3_coder" in script
    assert "neos_hermes" not in script
