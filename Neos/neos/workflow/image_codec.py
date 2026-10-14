from __future__ import annotations
import base64
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable
from PIL import Image
CONTEXT_IMAGE_CODEC_VERSION = 'neos-context-jpeg'
CONTEXT_IMAGE_FORMAT = 'JPEG'
CONTEXT_IMAGE_MEDIA_TYPE = 'image/jpeg'
CONTEXT_IMAGE_MAX_SIDE = 1536
CONTEXT_IMAGE_MAX_PIXELS = 786432
CONTEXT_IMAGE_JPEG_QUALITY = 85
CONTEXT_IMAGE_MAX_ASPECT_RATIO = 128
CONTEXT_IMAGE_JPEG_SUBSAMPLING = 0
_CODEC_SETTINGS_KEY = '_neos_context_codec_settings'
_JPEG_BYTES_KEY = '_neos_context_jpeg_bytes'

def codec_config(overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    values = {'version': CONTEXT_IMAGE_CODEC_VERSION, 'format': CONTEXT_IMAGE_FORMAT, 'media_type': CONTEXT_IMAGE_MEDIA_TYPE, 'max_side': CONTEXT_IMAGE_MAX_SIDE, 'max_pixels': CONTEXT_IMAGE_MAX_PIXELS, 'jpeg_quality': CONTEXT_IMAGE_JPEG_QUALITY, 'jpeg_subsampling': CONTEXT_IMAGE_JPEG_SUBSAMPLING, 'max_aspect_ratio': CONTEXT_IMAGE_MAX_ASPECT_RATIO}
    source = overrides or {}
    mapping = {'context_image_max_side': 'max_side', 'context_image_max_pixels': 'max_pixels', 'context_image_jpeg_quality': 'jpeg_quality', 'context_image_max_aspect_ratio': 'max_aspect_ratio'}
    for source_key, target_key in mapping.items():
        if source_key in source and source[source_key] is not None:
            values[target_key] = int(source[source_key])
    for key in ('max_side', 'max_pixels', 'jpeg_quality', 'max_aspect_ratio'):
        if int(values[key]) <= 0:
            raise ValueError(f'Neos image codec {key} must be positive')
    if int(values['jpeg_quality']) > 100:
        raise ValueError('Neos image codec jpeg_quality must be <= 100')
    return values

def _pad_extreme_aspect(image: Image.Image, max_ratio: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f'image has invalid dimensions: {width}x{height}')
    if width > height * max_ratio:
        padded_height = math.ceil(width / max_ratio)
        canvas = Image.new('RGB', (width, padded_height), 'white')
        canvas.paste(image, (0, (padded_height - height) // 2))
        return canvas
    if height > width * max_ratio:
        padded_width = math.ceil(height / max_ratio)
        canvas = Image.new('RGB', (padded_width, height), 'white')
        canvas.paste(image, ((padded_width - width) // 2, 0))
        return canvas
    return image

def _resize(image: Image.Image, *, max_side: int, max_pixels: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height), math.sqrt(max_pixels / (width * height)))
    if scale >= 1.0:
        return image
    size = (max(1, int(math.floor(width * scale))), max(1, int(math.floor(height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)

def prepare_context_image(image: Image.Image, overrides: dict[str, Any] | None=None) -> Image.Image:
    pass
    config = codec_config(overrides)
    settings = config
    if image.info.get(_CODEC_SETTINGS_KEY) == settings and isinstance(image.info.get(_JPEG_BYTES_KEY), bytes):
        return image
    normalized = _pad_extreme_aspect(image.convert('RGB'), int(config['max_aspect_ratio']))
    normalized = _resize(normalized, max_side=int(config['max_side']), max_pixels=int(config['max_pixels']))
    buffer = io.BytesIO()
    normalized.save(buffer, format=CONTEXT_IMAGE_FORMAT, quality=int(config['jpeg_quality']), subsampling=int(config['jpeg_subsampling']), optimize=False, progressive=False)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        prepared = decoded.convert('RGB')
        prepared.load()
    prepared.info[_CODEC_SETTINGS_KEY] = settings
    prepared.info[_JPEG_BYTES_KEY] = payload
    return prepared

def prepare_context_images(images: Iterable[Image.Image] | None, overrides: dict[str, Any] | None=None) -> list[Image.Image]:
    return [prepare_context_image(image, overrides) for image in images or []]

def context_image_jpeg_bytes(image: Image.Image, overrides: dict[str, Any] | None=None) -> bytes:
    prepared = prepare_context_image(image, overrides)
    payload = prepared.info.get(_JPEG_BYTES_KEY)
    if not isinstance(payload, bytes):
        raise RuntimeError('Neos context JPEG payload is unavailable')
    return payload

def context_image_data_uri(image: Image.Image, overrides: dict[str, Any] | None=None) -> str:
    payload = context_image_jpeg_bytes(image, overrides)
    return f'data:{CONTEXT_IMAGE_MEDIA_TYPE};base64,' + base64.b64encode(payload).decode('ascii')

def load_context_artifact_image(artifact, overrides=None):
    path = Path(artifact['path'])
    with Image.open(path) as opened:
        image = opened.convert('RGB')
        if artifact.get('context_pixels_prepared') and opened.format == 'JPEG':
            image.info[_CODEC_SETTINGS_KEY] = codec_config(overrides)
            image.info[_JPEG_BYTES_KEY] = path.read_bytes()
            return image
    return prepare_context_image(image, overrides)
