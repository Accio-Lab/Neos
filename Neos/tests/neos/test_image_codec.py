import base64
import io

from PIL import Image

from neos.workflow.image_codec import (
    CONTEXT_IMAGE_MAX_PIXELS,
    CONTEXT_IMAGE_MAX_SIDE,
    codec_config,
    context_image_data_uri,
    prepare_context_image,
)


def test_codec_is_idempotent_and_bounds_resolution():
    source = Image.effect_noise((2400, 1600), 48).convert("RGB")
    prepared = prepare_context_image(source)
    again = prepare_context_image(prepared)
    assert again is prepared
    assert max(prepared.size) <= CONTEXT_IMAGE_MAX_SIDE
    assert prepared.width * prepared.height <= CONTEXT_IMAGE_MAX_PIXELS


def test_data_uri_is_jpeg_and_decodes_to_prepared_pixels():
    source = Image.new("RGB", (320, 240), (40, 80, 120))
    prepared = prepare_context_image(source)
    uri = context_image_data_uri(prepared)
    assert uri.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as decoded:
        assert decoded.size == prepared.size
        assert decoded.convert("RGB").tobytes() == prepared.tobytes()


def test_codec_manifest_is_stable_and_override_sensitive():
    assert codec_config()["jpeg_quality"] == 85
    assert codec_config({"context_image_jpeg_quality": 90})["jpeg_quality"] == 90
