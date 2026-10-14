import io
import json
from PIL import Image
from neos.workflow.image_codec import codec_config, prepare_context_image
from neos.workflow.visual_results import output_refs, render_visual_result
VISUAL_REGISTER_REUSE_VERSION = 'neos-visual-register-reuse'

def pixel_identity(image):
    rgb = image if image.mode == 'RGB' else image.convert('RGB')
    return (rgb.size, rgb.tobytes())

def _prepare_new_pixels(image, identity, overrides):
    payload = image.info.get('_neos_context_jpeg_bytes')
    if image.info.get('_neos_context_codec_settings') == codec_config(overrides) and isinstance(payload, bytes):
        try:
            with Image.open(io.BytesIO(payload)) as decoded:
                if decoded.size == image.size and pixel_identity(decoded) == identity:
                    return image
        except OSError:
            pass
    fresh = image.copy()
    fresh.info.pop('_neos_context_codec_settings', None)
    fresh.info.pop('_neos_context_jpeg_bytes', None)
    return prepare_context_image(fresh, overrides)

def reuse_loaded_images(tool_name, execution, registry, codec_overrides=None):
    pass
    execution.metadata = dict(execution.metadata or {})
    execution.metadata['visual_register_reuse_version'] = VISUAL_REGISTER_REUSE_VERSION
    if not execution.images:
        return execution
    records = json.loads(execution.text)
    success = 'loaded' if tool_name == 'FetchImage' else 'cropped'
    positions = [i for i, record in enumerate(records) if record.get('status') == success]
    if len(positions) != len(execution.images):
        raise ValueError('Visual result records do not match returned pixels')
    known = list(registry)

    def existing_ref(image):
        size, pixels = pixel_identity(image)
        for index, prior in enumerate(known, 1):
            if prior is image or (prior.size == size and pixel_identity(prior)[1] == pixels):
                return f'img_{index}'
        return None
    new_images, retained_positions, reused = ([], [], [])
    for image_index, record_index in enumerate(positions):
        image = execution.images[image_index]
        identity = pixel_identity(image)
        ref = existing_ref(image)
        if ref is None:
            image = _prepare_new_pixels(image, identity, codec_overrides)
            identity = pixel_identity(image)
            ref = existing_ref(image)
        record = records[record_index]
        if ref is not None:
            record.update(status='reused', output_ref=ref, observation='These pixels are already visible under this visual-register ID.')
            reused.append(ref)
        else:
            ref = f'img_{len(registry) + len(new_images) + 1}'
            known.append(image)
            record['output_ref'] = ref
            new_images.append(image)
            retained_positions.append(image_index)
    refs = output_refs(len(registry), len(new_images))
    for field in ('image_source_urls', 'image_kinds'):
        values = list(execution.metadata.get(field) or [])
        execution.metadata[field] = [values[i] if i < len(values) else None for i in retained_positions]
    execution.metadata.update(output_refs=refs, reused_refs=list(dict.fromkeys(reused)), n_images=len(new_images), n_reused_images=len(reused))
    execution.images = new_images
    execution.text = render_visual_result(tool_name, records, refs)
    return execution
