
from __future__ import annotations

from importlib import import_module

__all__ = [
    "TextSearchTool",
    "ImageSearchTool",
    "VisitTool",
    "CropTool",
    "VisualSearchTool",
    "ZoominSearchTool",
    "OcrTool",
    "PythonTool",
]

_MODULE_BY_NAME = {
    "TextSearchTool": "neos.tools.search_tools",
    "ImageSearchTool": "neos.tools.search_tools",
    "VisitTool": "neos.tools.search_tools",
    "CropTool": "neos.tools.visual_tools",
    "VisualSearchTool": "neos.tools.visual_tools",
    "ZoominSearchTool": "neos.tools.visual_tools",
    "OcrTool": "neos.tools.visual_tools",
    "PythonTool": "neos.tools.python_tool",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
