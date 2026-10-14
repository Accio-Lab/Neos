

import math
from typing import Any


def _image_size(image_or_size: Any) -> tuple[int, int]:
    size = getattr(image_or_size, "size", image_or_size)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("image_or_size must be an image or (width, height)")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    return width, height


def normalize_qwen_bbox(bbox_2d: Any) -> tuple[float, float, float, float]:
    pass
    if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
        raise ValueError(
            "bbox_2d must be [left, top, right, bottom] in normalized "
            "0--1000 coordinates"
        )
    try:
        values = tuple(float(value) for value in bbox_2d)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox_2d coordinates must be numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox_2d coordinates must be finite")
    left, top, right, bottom = (max(0.0, min(1000.0, value)) for value in values)
    if right <= left or bottom <= top:
        raise ValueError("bbox_2d must have right > left and bottom > top")
    return left, top, right, bottom


def qwen_bbox_to_pixels(
    image_or_size: Any,
    bbox_2d: Any,
) -> tuple[int, int, int, int]:
    pass
    width, height = _image_size(image_or_size)
    left_n, top_n, right_n, bottom_n = normalize_qwen_bbox(bbox_2d)
    left = max(0, min(math.floor(left_n * width / 1000), width - 1))
    top = max(0, min(math.floor(top_n * height / 1000), height - 1))
    right = max(left + 1, min(math.ceil(right_n * width / 1000), width))
    bottom = max(top + 1, min(math.ceil(bottom_n * height / 1000), height))
    return left, top, right, bottom


def pixels_to_qwen_bbox(
    image_or_size: Any,
    pixel_bbox: Any,
) -> list[int]:
    pass
    width, height = _image_size(image_or_size)
    if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4:
        raise ValueError("pixel_bbox must be [left, top, right, bottom]")
    try:
        values = tuple(float(value) for value in pixel_bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel_bbox coordinates must be numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pixel_bbox coordinates must be finite")
    left, top, right, bottom = values
    left = max(0.0, min(float(width), left))
    top = max(0.0, min(float(height), top))
    right = max(0.0, min(float(width), right))
    bottom = max(0.0, min(float(height), bottom))
    if right <= left or bottom <= top:
        raise ValueError("pixel_bbox must have right > left and bottom > top")
    return [
        max(0, min(1000, int(round(left * 1000 / width)))),
        max(0, min(1000, int(round(top * 1000 / height)))),
        max(0, min(1000, int(round(right * 1000 / width)))),
        max(0, min(1000, int(round(bottom * 1000 / height)))),
    ]


def crop_qwen(image: Any, bbox_2d: Any):
    pass
    return image.crop(qwen_bbox_to_pixels(image, bbox_2d))
