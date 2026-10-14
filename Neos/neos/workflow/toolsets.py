
from __future__ import annotations
import os
import re

TOOLSET_DISABLED = {
    'full': frozenset(),
    'no_webvisit': frozenset({'WebVisit'}),
    'no_fetchimage': frozenset({'FetchImage'}),
    'no_cropimage': frozenset({'CropImage'}),
    'no_imagesearch': frozenset({'ImageSearch'}),
    'no_reverseimagesearch': frozenset({'ReverseImageSearch'}),
}
NEOS_TOOLSET_PROFILE = os.getenv('NEOS_TOOLSET', 'full')
if NEOS_TOOLSET_PROFILE not in TOOLSET_DISABLED:
    raise ValueError(f'Unknown NEOS_TOOLSET={NEOS_TOOLSET_PROFILE!r}')

DISABLED_TOOL_NAMES = TOOLSET_DISABLED[NEOS_TOOLSET_PROFILE]
_MENTIONS = {
    'WebVisit': r'\b(?:web_?visit|web_?summary|visit)\b',
    'FetchImage': r'\b(?:FetchImage|fetch_image)\b',
    'CropImage': r'\b(?:CropImage|crop_image|image_zoom_in)\b',
    'ImageSearch': r'\b(?:ImageSearch|image_search)\b',
    'ReverseImageSearch': r'\b(?:ReverseImageSearch|reverse_image_search)\b',
}

def disabled_tool_mentions(text: str) -> list[str]:
    return [name for name in sorted(DISABLED_TOOL_NAMES)
            if re.search(_MENTIONS[name], text, flags=re.IGNORECASE)]

def adapt_system_prompt(text: str) -> str:
    pass
    text = '\n'.join(line for line in text.split('\n')
                     if not any(line.startswith(f'- {name} ') for name in DISABLED_TOOL_NAMES))
    if 'FetchImage' in DISABLED_TOOL_NAMES:
        text = text.replace(
            'A candidate image becomes visible evidence only after FetchImage loads it into the visual register.',
            'Candidate references are not new pixels; only input images and pixels already present in the visual register are available for visual observation.')
    if 'CropImage' in DISABLED_TOOL_NAMES:
        text = text.replace(' CropImage creates another reusable visual observation from pixels already in that register.', '')
    return text

def adapt_function_description(function: dict) -> None:
    if 'FetchImage' not in DISABLED_TOOL_NAMES:
        return
    if function['name'] == 'WebVisit':
        function['description'] = function['description'].replace(
            'Referenced images are not visually observed until FetchImage loads them.',
            'Page-image references are discovery signals, not new pixels in your visual register.')
    if function['name'] == 'ImageSearch':
        function['description'] = function['description'].replace(
            'Use FetchImage on a selected direct image URL before making a visual claim.',
            'Candidate references alone do not establish pixel-level facts.')

def redact_disabled_tool_mentions(text: str) -> str:
    pass
    for name in sorted(DISABLED_TOOL_NAMES):
        text = re.sub(_MENTIONS[name], '[unavailable_tool]', text, flags=re.IGNORECASE)
    return text
