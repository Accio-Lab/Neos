from __future__ import annotations

import json
from types import SimpleNamespace

from neos.workflow.agent import _rejected_message_for_retry, run_agent
from neos.workflow.eval import _backfill_empty_adapter_transforms
from neos.workflow.native_adapter import (
    adapt_native_assistant_message,
    adapt_qwen35_native_assistant_message,
    build_format_retry_instruction,
)
from neos.workflow.protocol import (
    QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE,
    QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION,
    parse_neos_assistant_message,
)


def _adapt_qwen35(
    message: dict, *, question: str = "Which result is correct?", images=()
):
    strict = parse_neos_assistant_message(
        message, tool_schema_version="native_array"
    )
    assert not strict.valid
    return adapt_qwen35_native_assistant_message(
        message,
        strict,
        question=question,
        available_image_refs=images,
    )


def test_qwen35_adapter_normalizes_malformed_array_string() -> None:
    parsed = _adapt_qwen35(
        {
            "role": "assistant",
            "reasoning_content": "Search both clues.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_qwen35",
                    "type": "function",
                    "function": {
                        "name": "TextSearch",
                        "arguments": '{"query":"[\\"first clue\\", \\"second clue\\")"}',
                    },
                }
            ],
        }
    )
    assert parsed.valid, parsed.error
    assert parsed.tool_call["name"] == "TextSearch"
    assert parsed.tool_call["arguments"]["query"]
    assert parsed.adaptation["profile"] == QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE
    assert parsed.adaptation["adapter_version"] == QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION


def _adapt(message: dict, *, question: str = "Which result is correct?", images=()):
    strict = parse_neos_assistant_message(message)
    assert not strict.valid
    return adapt_native_assistant_message(
        message,
        strict,
        question=question,
        available_image_refs=images,
    )




def test_qwen35_repairs_hermes_json_and_keeps_array_arguments() -> None:
    parsed = _adapt_qwen35(
        {
            "role": "assistant",
            "reasoning_content": "Inspect the source.",
            "content": (
                '<tool_call>\n{"name":"WebVisit","arguments":'
                '{"url":["https://example.com"],"goal":"Find title"}}}'
                "\n</tool_call>"
            ),
        }
    )
    assert parsed.valid, parsed.error
    assert parsed.tool_call == {
        "id": parsed.tool_call["id"],
        "name": "WebVisit",
        "arguments": {
            "url": ["https://example.com"],
            "goal": "Find title",
        },
    }
    assert parsed.adaptation["profile"] == QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE
    assert parsed.adaptation["adapter_version"] == QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION
    assert "repair_hermes_json" in parsed.adaptation["transforms"]


def test_qwen35_collapses_only_identical_repeated_hermes_calls() -> None:
    payload = (
        '{"name":"TextSearch","arguments":{"query":["same query"]}}'
    )
    identical = _adapt_qwen35(
        {
            "role": "assistant",
            "content": (
                f"<tool_call>\n{payload}\n</think>\n\n"
                f"<tool_call>\n{payload}\n</tool_call>"
            ),
        }
    )
    assert identical.valid
    assert identical.tool_call["arguments"] == {"query": ["same query"]}
    assert (
        "collapse_identical_repeated_tool_calls"
        in identical.adaptation["transforms"]
    )

    distinct = _adapt_qwen35(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"TextSearch","arguments":'
                '{"query":["one"]}}</tool_call>'
                '<tool_call>{"name":"TextSearch","arguments":'
                '{"query":["two"]}}</tool_call>'
            ),
        }
    )
    assert not distinct.valid


def test_qwen35_compat_retries_ambiguous_empty_turn_under_native_schema() -> None:
    class Model:
        official_multimodal_message_order = True

        def __init__(self):
            self.calls = 0
            self.last = None

        def chat(self, messages, **kwargs):
            self.calls += 1
            self.last = {"request": {}, "response": {"usage": {}}}
            if self.calls == 1:
                return {"role": "assistant", "content": ""}
            if kwargs["tool_choice"] == "none":
                return {"role": "assistant", "content": "Example"}
            assert kwargs["tool_choice"] == "required"
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_retry",
                        "type": "function",
                        "function": {
                            "name": "TextSearch",
                            "arguments": '{"query":["example"]}',
                        },
                    }
                ],
            }

        def take_last_response_metadata(self):
            return self.last

    class Runtime:
        config = {"tool_schema_version": "native_array"}

        def execute(self, name, arguments, **kwargs):
            from neos.tools.runtime import ToolExecution

            return ToolExecution("evidence", metadata={"runtime_type_validation_success": True})

    
    
    result = run_agent(
        question="Example?",
        image_inputs=[],
        model=Model(),
        runtime=Runtime(),
        max_turns=2,
        assistant_format_profile=QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE,
        tool_schema_version="native_array",
    )
    assert result["format_recovery_model_calls"] >= 1
    assert result["assistant_format_adapter_version"] == QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION


def test_recovers_textual_tool_wrapper_as_native_call() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "reasoning_content": "The source page must be inspected.",
            "content": (
                '<tool_call>{"name":"WebVisit","arguments":'
                '{"url":["https://example.com"],"goal":"Find title"}}'
                "</tool_call>"
            ),
            "tool_calls": [],
        }
    )
    assert parsed.valid
    assert parsed.action_kind == "tool_call"
    assert parsed.tool_call["name"] == "WebVisit"
    assert parsed.message["content"] == ""
    assert parsed.message["tool_calls"][0]["id"].startswith("call_adapter_")
    assert "<tool_call>" not in json.dumps(parsed.message)


def test_repairs_native_argument_aliases_limits_and_missing_goal() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "WebVisit",
                        "arguments": {
                            "urls": [
                                f"https://example.com/{index}" for index in range(8)
                            ]
                        },
                    },
                }
            ],
        }
    )
    assert parsed.valid
    assert parsed.tool_call["name"] == "WebVisit"
    assert len(parsed.tool_call["arguments"]["url"]) == 5
    assert parsed.tool_call["arguments"]["goal"]
    assert parsed.adaptation["transforms"] == [
        "rename_urls_to_url",
        "supply_webvisit_goal",
        "truncate_webvisit_batch",
    ]


def test_repairs_crop_aliases_pixel_ranges_and_box_objects() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_crop",
                    "type": "function",
                    "function": {
                        "name": "CropImage",
                        "arguments": {
                            "image": "img_1",
                            "boxes": [
                                {"left": 0.2, "top": 0.1, "right": 0.8, "bottom": 0.9},
                                [0, 1200, 1000, 1800],
                            ],
                        },
                    },
                }
            ],
        },
        images=("img_1",),
    )
    assert parsed.valid
    assert parsed.tool_call["arguments"]["crop_config"]["img_1"][0] == [
        200,
        100,
        800,
        900,
    ]

    parallel = _adapt(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_parallel_crop",
                    "type": "function",
                    "function": {
                        "name": "CropImage",
                        "arguments": {
                            "image_token": ["img_1", "img_2"],
                            "crop": [[0, 0, 300, 300], [700, 700, 1000, 1000]],
                        },
                    },
                }
            ],
        },
        images=("img_1", "img_2"),
    )
    assert parallel.valid
    assert parallel.tool_call["arguments"]["crop_config"] == {
        "img_1": [0, 0, 300, 300],
        "img_2": [700, 700, 1000, 1000],
    }


def test_recovers_answer_pseudotool_and_orphan_think_close() -> None:
    pseudo = _adapt(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"Answer","arguments":'
                '{"answer":"Diona"}}</tool_call>'
            ),
        }
    )
    assert pseudo.valid and pseudo.answer == "Diona"

    pyfinal = _adapt(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"PyFinalText","arguments":'
                '{"final":"Diona"}}</tool_call>'
            ),
        }
    )
    assert pyfinal.valid and pyfinal.answer == "Diona"

    orphan = _adapt(
        {
            "role": "assistant",
            "reasoning_content": "The sources agree.",
            "content": "United States\n</think>",
        }
    )
    assert orphan.valid and orphan.answer == "United States"
    assert "</think>" not in json.dumps(orphan.message)


def test_extracts_only_explicit_reasoning_answer() -> None:
    explicit = _adapt(
        {
            "role": "assistant",
            "reasoning_content": (
                "Evidence is sufficient. Therefore, the final answer: 1996."
            ),
            "content": "",
        }
    )
    assert explicit.valid and explicit.answer == "1996."

    strict = parse_neos_assistant_message(
        {
            "role": "assistant",
            "reasoning_content": "I should continue investigating.",
            "content": "",
        }
    )
    ambiguous = adapt_native_assistant_message(
        {
            "role": "assistant",
            "reasoning_content": "I should continue investigating.",
            "content": "",
        },
        strict,
        question="Example?",
        available_image_refs=(),
    )
    assert not ambiguous.valid


def test_format_retry_recovers_same_agent_turn() -> None:
    class Model:
        official_multimodal_message_order = True

        def __init__(self):
            self.calls = 0
            self.last = None
            self.requests = []

        def chat(self, messages, **kwargs):
            self.calls += 1
            self.requests.append((messages, kwargs))
            self.last = {
                "request": {"message_count": len(messages)},
                "response": {"usage": {}},
            }
            if self.calls == 1:
                return {
                    "role": "assistant",
                    "reasoning_content": "I should continue but emitted no action.",
                    "content": "",
                }
            assert "previous response could not be accepted" in messages[-1]["content"]
            assert messages[-2]["role"] == "assistant"
            assert "emitted no action" in messages[-2]["reasoning_content"]
            return {"role": "assistant", "content": "Example"}

        def take_last_response_metadata(self):
            return self.last

    model = Model()
    result = run_agent(
        question="Give the example.",
        image_inputs=[],
        model=model,
        runtime=SimpleNamespace(config={}),
        assistant_format_profile=QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE,
    )
    assert result["termination_reason"] == "final_answer"
    assert result["final_answer"] == "Example"
    assert result["model_calls"] == 2
    assert result["format_recovery_model_calls"] == 1
    assert result["format_adapted_turns"] == 1
    assert result["trajectory"][0]["format_recovery_attempts"]
    assert model.requests[0][1]["tool_choice"] == "auto"
    assert model.requests[1][1]["tool_choice"] == "required"


def test_backfills_empty_native_adapter_transform_trace() -> None:
    row = {
        "format_adapter_transform_counts": {},
        "trajectory": [
            {
                "role": "assistant",
                "format_adaptation": {
                    "profile": "qwen35_native_compat",
                    "adapter_version": "neos-qwen35-native-tool-normalization",
                    "transforms": [],
                },
            }
        ],
    }
    assert _backfill_empty_adapter_transforms(row) == 1
    assert row["trajectory"][0]["format_adaptation"]["transforms"] == [
        "normalize_native_tool_call"
    ]
    assert row["format_adapter_transform_counts"] == {
        "normalize_native_tool_call": 1
    }
    assert _backfill_empty_adapter_transforms(row) == 0


def test_does_not_promote_malformed_tool_payload_to_answer() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"TextSearch","arguments":',
        }
    )
    assert not parsed.valid


def test_recovers_first_unclosed_tool_payload_before_repeated_opening() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name":"TextSearch","arguments":'
                '{"query":"one"}}\n<tool_call>\n<|im_start|>'
            ),
        }
    )
    assert parsed.valid
    assert parsed.tool_call["arguments"] == {"query": ["one"]}


def test_multiple_valid_calls_are_retried_instead_of_silently_dropped() -> None:
    parsed = _adapt(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "TextSearch",
                        "arguments": '{"query":"one"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "TextSearch",
                        "arguments": '{"query":"two"}',
                    },
                },
            ],
        }
    )
    assert not parsed.valid


def test_retry_instruction_never_teaches_literal_serialization_tags() -> None:
    instruction = build_format_retry_instruction(
        "parser leaked <tool_call> and </think>", forced_final=False
    )
    assert "<tool_call>" not in instruction
    assert "<think>" not in instruction
    rejected = _rejected_message_for_retry(
        {
            "role": "assistant",
            "reasoning_content": "<think>I should search.</think>",
            "content": '<tool_call>{"bad":true}',
        }
    )
    rendered = json.dumps(rejected)
    assert "<tool_call>" not in rendered
    assert "<think>" not in rendered
