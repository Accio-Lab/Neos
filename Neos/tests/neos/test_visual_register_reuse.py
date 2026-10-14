import io
import json

from PIL import Image
import pytest

from neos.workflow.image_codec import context_image_jpeg_bytes, prepare_context_image
from neos.workflow.runtime import NeosRuntime
from neos.workflow.turn_credit import _append_inline_trajectory
from neos.workflow.visual_results import verify_execution_output_refs


class Client:
    def fetch_image(self, url):
        return Image.new('RGB', (64, 48), 'red' if 'red' in url else 'blue')


def runtime():
    return NeosRuntime(client=Client(), mei=False, config={'reuse_loaded_images': True})


def test_repeat_fetch_and_alias_url_keep_one_image_and_one_id():
    tool = runtime()
    first = tool.execute('FetchImage', {'urls': ['https://example.org/red.jpg']}, [])
    assert first.metadata['output_refs'] == ['img_1']
    for _ in range(40):
        reused = tool.execute('FetchImage', {'urls': ['https://example.org/alias-red.jpg']}, first.images)
        assert reused.images == []
        assert reused.metadata['output_refs'] == []
        assert reused.metadata['reused_refs'] == ['img_1']
        assert json.loads(reused.text)[0]['output_ref'] == 'img_1'
        verify_execution_output_refs('FetchImage', reused.metadata, [], registry_size=1)


def test_mixed_new_and_reused_images_preserve_request_order():
    tool = runtime()
    first = tool.execute('FetchImage', {'urls': ['https://example.org/red.jpg']}, [])
    result = tool.execute('FetchImage', {'urls': ['https://example.org/red.jpg',
        'https://example.org/blue.jpg', 'https://example.org/alias-blue.jpg']}, first.images)
    records = json.loads(result.text)
    assert [r['output_ref'] for r in records] == ['img_1', 'img_2', 'img_2']
    assert [r['status'] for r in records] == ['reused', 'loaded', 'reused']
    assert len(result.images) == 1
    verify_execution_output_refs('FetchImage', result.metadata, ['img_2'], registry_size=1)


def test_crop_does_not_inherit_parent_jpeg_and_repeated_crop_is_reused():
    tool = runtime()
    first = tool.execute('FetchImage', {'urls': ['https://example.org/red.jpg']}, [])
    args = {'crop_config': {'img_1': [0, 0, 500, 500]}}
    crop = tool.execute('CropImage', args, first.images)
    assert crop.metadata['output_refs'] == ['img_2']
    assert crop.images[0].size == (32, 24)
    with Image.open(io.BytesIO(context_image_jpeg_bytes(crop.images[0]))) as decoded:
        assert decoded.size == (32, 24)
    repeated = tool.execute('CropImage', args, first.images + crop.images)
    assert repeated.images == [] and repeated.metadata['reused_refs'] == ['img_2']


def test_credit_reuses_original_id_and_first_observation_position():
    events = [
        {'role': 'user', 'content': 'Question', 'image_refs': ['img_1']},
        {'role': 'tool', 'turn': 1, 'content': 'Fetched', 'output_refs': ['img_2']},
        {'role': 'tool', 'turn': 2, 'content': '[{"status":"reused","output_ref":"img_2"}]'},
        {'role': 'tool', 'turn': 3, 'content': 'Repeated explicit ref', 'image_refs': ['img_2']},
    ]
    visuals = [{'ref': 'img_1', 'data_url': 'data:image/jpeg;base64,ONE'},
               {'ref': 'img_2', 'data_url': 'data:image/jpeg;base64,TWO'}]
    content = []
    _append_inline_trajectory(content, label='REFERENCE', events=events, visuals=visuals)
    assert [r['image_url']['url'] for r in content if r['type'] == 'image_url'] == [v['data_url'] for v in visuals]
    assert 'reused_refs' in '\n'.join(r.get('text', '') for r in content)
    with pytest.raises(ValueError, match='before their first observation'):
        _append_inline_trajectory([], label='BAD', events=[events[0], events[2], events[1]], visuals=visuals)


def test_unknown_reused_ref_is_not_silently_accepted():
    with pytest.raises(RuntimeError):
        verify_execution_output_refs('FetchImage', {'output_refs': [], 'reused_refs': ['img_3']}, [], registry_size=1)



