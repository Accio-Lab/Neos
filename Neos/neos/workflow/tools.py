
from __future__ import annotations

import asyncio
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from neos.workflow.image_codec import prepare_context_images
from neos.workflow.runtime import NeosRuntime
from neos.workflow.schemas import (
    CROP_IMAGE_FUNCTION,
    FETCH_IMAGE_FUNCTION,
    IMAGE_SEARCH_FUNCTION,
    REVERSE_IMAGE_SEARCH_FUNCTION,
    TEXT_SEARCH_FUNCTION,
    WEB_VISIT_FUNCTION,
    verl_openai_tool_schema,
)
from neos.workflow.visual_results import (
    output_refs,
    verify_execution_output_refs,
)


def _runtime(config: dict[str, Any]) -> NeosRuntime:
    mei = False if str(config.get("mei_mode", "vlm")).lower() in {
        "off",
        "false",
        "metadata",
    } else None
    return NeosRuntime(mei=mei, config=config)


def _registry(kwargs: dict[str, Any]) -> list:
    agent_data = kwargs.get("agent_data")
    return list(getattr(agent_data, "image_data", None) or [])


def _tool_response(
    result,
    registry_size: int = 0,
    codec_overrides: dict[str, Any] | None = None,
    tool_name: str | None = None,
):
    context_images = prepare_context_images(result.images, codec_overrides)
    text = result.text
    if tool_name is not None:
        verify_execution_output_refs(
            tool_name,
            result.metadata,
            output_refs(registry_size, len(context_images)),
            registry_size=registry_size,
        )
    return (
        
        
        
        ToolResponse(text=text, image=context_images),
        0.0,
        result.metadata,
    )


class _NeosTool(BaseTool):
    function: dict

    def __init__(
        self,
        config: dict,
        tool_schema: Optional[OpenAIFunctionToolSchema] = None,
    ):
        schema = tool_schema or OpenAIFunctionToolSchema.model_validate(
            verl_openai_tool_schema(self.function)
        )
        super().__init__(config, schema)
        self.runtime = _runtime(config)


class TextSearchTool(_NeosTool):
    function = TEXT_SEARCH_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        result = await asyncio.to_thread(
            self.runtime.execute, "TextSearch", parameters
        )
        return _tool_response(result)


class WebVisitTool(_NeosTool):
    function = WEB_VISIT_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        result = await asyncio.to_thread(
            self.runtime.execute, "WebVisit", parameters
        )
        return _tool_response(result)


class ImageSearchTool(_NeosTool):
    function = IMAGE_SEARCH_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        result = await asyncio.to_thread(
            self.runtime.execute, "ImageSearch", parameters
        )
        return _tool_response(result)


class ReverseImageSearchTool(_NeosTool):
    function = REVERSE_IMAGE_SEARCH_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        registry = _registry(kwargs)
        result = await asyncio.to_thread(
            self.runtime.execute,
            "ReverseImageSearch",
            parameters,
            registry,
        )
        return _tool_response(result)


class FetchImageTool(_NeosTool):
    function = FETCH_IMAGE_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        registry = _registry(kwargs)
        result = await asyncio.to_thread(
            self.runtime.execute, "FetchImage", parameters, registry
        )
        return _tool_response(
            result,
            len(registry),
            self.runtime.config,
            "FetchImage",
        )


class CropImageTool(_NeosTool):
    function = CROP_IMAGE_FUNCTION

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        registry = _registry(kwargs)
        result = await asyncio.to_thread(
            self.runtime.execute,
            "CropImage",
            parameters,
            registry,
        )
        return _tool_response(
            result,
            len(registry),
            self.runtime.config,
            "CropImage",
        )
