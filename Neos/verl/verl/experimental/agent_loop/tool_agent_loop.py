# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from neos.protocol import (
    DUPLICATE_TOOL_CALL_FINAL as _DUPLICATE_TOOL_CALL_FINAL,
    DUPLICATE_TOOL_CALL_WARNING as _DUPLICATE_TOOL_CALL_WARNING,
    tool_call_key,
    tool_call_overflow_message as _tool_call_overflow_message,
    truncate_tool_response,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)

def _tool_call_key(tool_call: FunctionCall) -> str:
    """Normalize name+arguments so that semantically identical calls hash equal regardless of
    JSON key order. ``arguments`` may be a JSON string or an already-parsed object."""
    return tool_call_key(tool_call.name, tool_call.arguments)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_call_counts: dict[str, int] = {}

        
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        
        
        self.max_duplicate_tool_calls = getattr(self.rollout_config.multi_turn, "max_duplicate_tool_calls", 3)

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        
        
        
        extra_info = kwargs.get("extra_info", {}) or {}
        if str(extra_info.get("protocol_version") or "").startswith(
            "neos-"
        ):
            from neos.workflow.image_codec import prepare_context_images

            images = prepare_context_images(images)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        
        agent_data.extra_fields.setdefault("tool_calls_emitted", 0)
        agent_data.extra_fields.setdefault("tool_calls_rejected", 0)
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        if self.enable_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(agent_data.messages, tools=schemas)
        else:
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
                audios=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        if self.enable_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                agent_data.prompt_ids,
                agent_data.response_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if (agent_data.response_logprobs or output.log_probs) else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if response_logprobs is not None:
                agent_data.response_logprobs = response_logprobs
        else:
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        assistant_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, tools
        )
        if self.enable_continuous_token:
            agent_data.messages.append(self._build_assistant_message(assistant_content, agent_data))

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  
        previous_messages = list(agent_data.messages)

        emitted_this_turn = len(agent_data.tool_calls)
        accepted_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        rejected_this_turn = emitted_this_turn - len(accepted_tool_calls)
        agent_data.extra_fields["tool_calls_emitted"] = (
            agent_data.extra_fields.get("tool_calls_emitted", 0) + emitted_this_turn
        )
        agent_data.extra_fields["tool_calls_rejected"] = (
            agent_data.extra_fields.get("tool_calls_rejected", 0) + rejected_this_turn
        )

        call_counts = agent_data.tool_call_counts
        dup_terminate = False

        dup_guard = self.max_duplicate_tool_calls > 0

        async def _dispatch(tool_call: FunctionCall, dup_count: int):
            if dup_guard and dup_count >= self.max_duplicate_tool_calls:
                return ToolResponse(text=_DUPLICATE_TOOL_CALL_FINAL), None, {}
            if dup_guard and dup_count == self.max_duplicate_tool_calls - 1:
                return ToolResponse(text=_DUPLICATE_TOOL_CALL_WARNING), None, {}
            return await self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)

        tasks = []
        tool_call_names = []
        for tool_call in accepted_tool_calls:
            call_key = _tool_call_key(tool_call)
            dup_count = call_counts.get(call_key, 0)
            call_counts[call_key] = dup_count + 1
            if dup_guard and dup_count >= self.max_duplicate_tool_calls:
                dup_terminate = True
            tasks.append(_dispatch(tool_call, dup_count))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        
        
        for tool_index, (tool_response, tool_reward, _) in enumerate(responses):
            tool_call = agent_data.tool_calls[tool_index]
            
            if tool_response.image or tool_response.video:
                
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    
                    
                    n_images = len(tool_response.image) if isinstance(tool_response.image, list) else 1
                    content.extend({"type": "image"} for _ in range(n_images))
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                
                message = {"role": "tool", "content": tool_response.text or ""}
            if tool_call.tool_call_id is not None:
                message["tool_call_id"] = tool_call.tool_call_id

            add_messages.append(message)

            
            if tool_response.image:
                
                if isinstance(tool_response.image, list):
                    
                    for img in tool_response.image:
                        if img is not None:  
                            new_images_this_turn.append(img)  
                else:
                    
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  

            
            if tool_response.video:
                
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if rejected_this_turn:
            overflow_message = _tool_call_overflow_message(emitted_this_turn, self.max_parallel_calls)
            
            
            
            if add_messages:
                content = add_messages[-1]["content"]
                if isinstance(content, list):
                    content.append({"type": "text", "text": overflow_message})
                else:
                    add_messages[-1]["content"] = f"{content}\n\n{overflow_message}"
            else:
                add_messages.append({"role": "tool", "content": overflow_message})

        agent_data.messages.extend(add_messages)

        if dup_terminate:
            logger.warning(
                f"Terminating trajectory {agent_data.request_id}: identical tool call repeated "
                f"more than {self.max_duplicate_tool_calls} times"
            )
            return AgentState.TERMINATED

        if self.enable_continuous_token and not new_images_this_turn:
            schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                previous_messages,
                agent_data.messages,
                agent_data.prompt_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if agent_data.response_logprobs else None,
                tools=schemas,
            )
            if len(response_mask) >= self.response_length:
                return AgentState.TERMINATED
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if agent_data.response_logprobs:
                agent_data.response_logprobs = response_logprobs or []
            agent_data.user_turns += 1
            return AgentState.GENERATING
        elif self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4":
            
            
            
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            
            
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )
            
            
            
            
            
            response_ids = self.turn_separator + response_ids

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _build_assistant_message(self, content: str, agent_data: AgentData) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if not agent_data.tool_calls:
            return message

        tool_calls = []
        for index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            function_call, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.arguments)
            )
            if has_decode_error:
                raise ValueError(
                    f"Invalid tool call arguments for '{tool_call.name}': expected a JSON object string, "
                    f"got {tool_call.arguments!r}"
                )
            tool_call_message = {
                "type": "function",
                "function": function_call.model_dump(),
            }
            if tool_call.tool_call_id is not None:
                tool_call_message["id"] = tool_call.tool_call_id
            tool_calls.append(tool_call_message)
        message["tool_calls"] = tool_calls
        return message

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]

            if isinstance(tool, FunctionTool):
                
                
                
                
                raw = await tool.call(tool_args)
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
            else:
                
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = await tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            tool_response_text = truncate_tool_response(
                tool_response_text,
                max_length=self.max_tool_response_length,
                side=self.tool_response_truncate_side,
            )

        
        tool_response_kwargs = {"text": tool_response_text}

        
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
