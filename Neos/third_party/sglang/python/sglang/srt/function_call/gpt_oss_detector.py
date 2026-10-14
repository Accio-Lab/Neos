import json
import logging
import re
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.parser.harmony_parser import HarmonyParser

logger = logging.getLogger(__name__)


class GptOssDetector(BaseFormatDetector):
    """
    Detector for T4-style function calls using HarmonyParser.

    Handles tool calls in the format:
    <|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{args}<|call|>
    """

    def __init__(self):
        super().__init__()
        self.harmony_parser = HarmonyParser()
        self.bot_token = "<|start|>assistant<|channel|>commentary"
        self.eot_token = "<|call|>"

        
        self.tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains TypeScript-style function call markers."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse TypeScript-style function calls from complete text."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        
        events = self.harmony_parser.parse(text)
        
        events += self.harmony_parser.parse("")

        tool_indices = self._get_tool_indices(tools)
        calls = []
        normal_parts = []
        tool_index = 0

        for event in events:
            if event.event_type == "tool_call":
                
                tool_call = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    tool_indices,
                    tool_index,
                )
                if tool_call:
                    calls.append(tool_call)
                    tool_index += 1
            elif event.event_type == "normal":
                normal_parts.append(event.content)
            

        normal_text = " ".join(normal_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse incremental streaming text for TypeScript-style function calls."""
        self._buffer += new_text

        
        events = self.harmony_parser.parse(new_text)

        
        
        
        if not events:
            has_harmony_markers = any(
                marker in self._buffer
                for marker in (
                    "<|start|>",
                    "<|channel|>",
                    "<|message|>",
                    "<|constrain|>",
                    "<|end|>",
                    "<|call|>",
                    "<|return|>",
                    "assistantfinal",
                )
            )
            if not has_harmony_markers:
                
                out = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=out, calls=[])

        
        if (
            "<|channel|>commentary to=" not in self._buffer
            and not self.current_tool_name_sent
        ):
            
            if (
                "<|channel|>final" in self._buffer
                or "assistantfinal" in self._buffer.lower()
            ):
                
                normal_text = "".join(
                    [e.content for e in events if e.event_type == "normal"]
                )
                if normal_text:
                    self._buffer = ""
                    return StreamingParseResult(normal_text=normal_text, calls=[])

            
            normal_text = "".join(
                [e.content for e in events if e.event_type == "normal"]
            )
            if normal_text or events:
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text, calls=[])
            else:
                
                return StreamingParseResult(normal_text="", calls=[])

        if not events:
            
            return StreamingParseResult(normal_text="", calls=[])

        
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls = []
        normal_text = ""

        for event in events:
            if event.event_type == "tool_call":
                
                tool_call_info = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    self._tool_indices,
                    self.current_tool_id if self.current_tool_id >= 0 else 0,
                )

                if tool_call_info:
                    
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_call_info.name,
                        "arguments": json.loads(tool_call_info.parameters),
                    }

                    
                    
                    calls.append(tool_call_info)

                    
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        tool_call_info.parameters
                    )

                    
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

            elif event.event_type == "normal":
                normal_text += event.content

        
        self._buffer = ""

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_tool_call_from_event(
        self, content: str, tool_indices: dict, tool_index: int
    ) -> Optional[ToolCallItem]:
        """
        Extract tool call information from HarmonyParser event content.

        Content format: "commentary to=functions.get_weather<|constrain|>json<|message|>{...}"
        """
        match = self.tool_extract_pattern.search(content)

        if not match:
            logger.debug(f"Could not extract tool call from: {content[:100]}")
            return None

        full_function_name = match.group(1)
        json_content = match.group(2)

        
        function_name = (
            full_function_name.split(".")[-1]
            if "." in full_function_name
            else full_function_name
        )

        
        if function_name not in tool_indices:
            logger.debug(f"Function {function_name} not in available tools")
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None  

        
        try:
            arguments = json.loads(json_content) if json_content.strip() else {}
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON arguments: {e}")
            return None

        return ToolCallItem(
            tool_index=tool_index,
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("structure_info not used with HarmonyParser")
