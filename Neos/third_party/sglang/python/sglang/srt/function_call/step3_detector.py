import ast
import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Tool]) -> str:
    """Get the expected type for a function argument from tool schema."""
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties", {})
    if arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


def parse_arguments(value: str) -> tuple[Any, bool]:
    """Parse a string value to appropriate type. Returns (parsed_value, success)."""
    try:
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = ast.literal_eval(value)
        return parsed_value, True
    except:
        return value, False


class Step3Detector(BaseFormatDetector):
    """
    Detector for Step3 model function call format.

    The Step3 format uses special Unicode tokens to delimit function calls
    with steptml XML format for invocations.

    Format Structure:
    ```
    <｜tool_calls_begin｜>
    <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="function_name">
    <steptml:parameter name="param1">value1</steptml:parameter>
    <steptml:parameter name="param2">value2</steptml:parameter>
    </steptml:invoke><｜tool_call_end｜>
    <｜tool_calls_end｜>
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool_calls_begin｜>"
        self.eot_token = "<｜tool_calls_end｜>"
        self.tool_call_begin = "<｜tool_call_begin｜>"
        self.tool_call_end = "<｜tool_call_end｜>"
        self.tool_sep = "<｜tool_sep｜>"

        
        self.invoke_regex = re.compile(
            r'<steptml:invoke name="([^"]+)">(.+?)</steptml:invoke>', re.DOTALL
        )
        self.param_regex = re.compile(
            r'<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>', re.DOTALL
        )

        
        self._in_tool_block: bool = False
        self._tool_block_finished: bool = False
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Step3 format tool call."""
        return self.bot_token in text

    def _parse_steptml_invoke(
        self, text: str, tools: List[Tool] = None
    ) -> tuple[str, dict]:
        """Parse steptml invoke format to extract function name and parameters."""
        invoke_match = self.invoke_regex.search(text)
        if not invoke_match:
            return None, {}

        func_name = invoke_match.group(1)
        params_text = invoke_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            
            if tools:
                arg_type = get_argument_type(func_name, param_name, tools)
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    params[param_name] = parsed_value
                else:
                    params[param_name] = param_value
            else:
                
                parsed_value, _ = parse_arguments(param_value)
                params[param_name] = parsed_value

        return func_name, params

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            pre_text, rest = text.split(self.bot_token, 1)

            
            if self.eot_token not in rest:
                return StreamingParseResult(normal_text=text, calls=[])

            tool_section, post_text = rest.split(self.eot_token, 1)

            
            calls = []
            tool_call_pattern = (
                f"{re.escape(self.tool_call_begin)}(.*?){re.escape(self.tool_call_end)}"
            )

            for match in re.finditer(tool_call_pattern, tool_section, re.DOTALL):
                call_content = match.group(1)

                
                if self.tool_sep not in call_content:
                    continue

                type_part, invoke_part = call_content.split(self.tool_sep, 1)
                if type_part.strip() != "function":
                    continue

                func_name, params = self._parse_steptml_invoke(invoke_part, tools)
                if func_name:
                    
                    action = {"name": func_name, "arguments": params}
                    calls.extend(self.parse_base_json(action, tools))

            
            normal_text = pre_text + post_text

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Step3 format.
        """
        self._buffer += new_text

        
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        
        if self._tool_block_finished:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        
        if not self._in_tool_block:
            if self.bot_token in self._buffer:
                idx = self._buffer.find(self.bot_token)
                normal_text = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self.bot_token) :]
                self._in_tool_block = True
                return StreamingParseResult(normal_text=normal_text)
            else:
                
                partial_len = self._ends_with_partial_token(
                    self._buffer, self.bot_token
                )
                if partial_len:
                    return StreamingParseResult()  
                else:
                    normal_text = self._buffer
                    self._buffer = ""
                    return StreamingParseResult(normal_text=normal_text)

        
        calls: List[ToolCallItem] = []

        
        if self.eot_token in self._buffer:
            idx = self._buffer.find(self.eot_token)

            
            if self._in_tool_call:
                
                before_eot = self._buffer[:idx]
                if self.tool_call_end in before_eot:
                    
                    result = self._parse_partial_tool_call(tools)
                    calls.extend(result.calls)
                else:
                    
                    logger.warning("Tool block ended with incomplete tool call")

            remaining = self._buffer[idx + len(self.eot_token) :]
            self._buffer = ""
            self._tool_block_finished = True

            
            self._reset_streaming_state()

            return StreamingParseResult(normal_text=remaining, calls=calls)

        
        if not self._in_tool_call:
            if self.tool_call_begin in self._buffer:
                idx = self._buffer.find(self.tool_call_begin)
                
                self._buffer = self._buffer[idx + len(self.tool_call_begin) :]
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                
            else:
                
                return StreamingParseResult()

        
        if self._in_tool_call:
            return self._parse_partial_tool_call(tools)

        return StreamingParseResult()

    def _parse_partial_tool_call(self, tools: List[Tool]) -> StreamingParseResult:
        """Parse partial tool call for streaming scenarios."""
        calls = []

        
        if self.tool_sep not in self._buffer:
            return StreamingParseResult(calls=calls)  

        type_part, invoke_part = self._buffer.split(self.tool_sep, 1)
        if type_part.strip() != "function":
            
            self._reset_streaming_state()
            return StreamingParseResult(calls=calls)

        
        if not self._function_name_sent:
            name_match = re.search(r'<steptml:invoke name="([^"]+)">', invoke_part)
            if name_match:
                func_name = name_match.group(1)

                
                if func_name in self._tool_indices:
                    self._current_function_name = func_name
                    self._function_name_sent = True

                    
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0

                    
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }

                    
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                else:
                    
                    logger.warning(f"Invalid function name: {func_name}")
                    self._reset_streaming_state()
                    return StreamingParseResult(calls=calls)
            else:
                
                return StreamingParseResult(calls=calls)

        
        if self._function_name_sent:
            
            new_params = {}
            for param_match in self.param_regex.finditer(invoke_part):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                
                arg_type = get_argument_type(
                    self._current_function_name, param_name, tools
                )
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    new_params[param_name] = parsed_value
                else:
                    new_params[param_name] = param_value

            
            if new_params != self._current_parameters:
                
                if not self._current_parameters:
                    
                    params_content = json.dumps(new_params, ensure_ascii=False)
                    if len(params_content) > 2:  
                        
                        diff = params_content[:-1]
                    else:
                        diff = "{"
                else:
                    
                    old_json = json.dumps(self._current_parameters, ensure_ascii=False)
                    new_json = json.dumps(new_params, ensure_ascii=False)

                    
                    old_without_brace = old_json[:-1]
                    new_without_brace = new_json[:-1]

                    
                    if new_without_brace.startswith(old_without_brace):
                        diff = new_without_brace[len(old_without_brace) :]
                    else:
                        
                        diff = ""

                if diff:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff

                
                self._current_parameters = new_params
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params

            
            if self.tool_call_end in self._buffer:
                
                if self.streamed_args_for_tool[self.current_tool_id]:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters="}",
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"

                
                end_idx = self._buffer.find(self.tool_call_end)
                
                self._buffer = self._buffer[end_idx + len(self.tool_call_end) :]

                
                self._reset_streaming_state()
                self.current_tool_id += 1

        return StreamingParseResult(calls=calls)

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
