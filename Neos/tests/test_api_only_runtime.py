
import inspect
import io
from unittest.mock import Mock

import pytest
from PIL import Image

from neos.tools.online_client import OnlineSearchClient
from neos.workflow.mei import NeosMEI
from neos.workflow.runtime import NeosRuntime


def configured_client():
    return OnlineSearchClient(
        serper_endpoint="https://text.example.test/search", serper_api_key="test",
        base_url="https://images.example.test/search", api_key="test",
        reader_url="https://reader.example.test",
    )


def test_public_client_has_no_result_storage_interface():
    client = configured_client()
    assert not hasattr(client, "cache")
    assert not hasattr(client, "cached_visit_extract")
    assert "cache" not in inspect.signature(OnlineSearchClient).parameters
    assert "cache" not in inspect.signature(NeosMEI).parameters
    assert not any("cache" in key for key in client.config_status())
    assert not any("cache" in key for key in client.provider_status())


@pytest.mark.parametrize("operation", ["text", "image", "reverse", "page", "pixels"])
def test_every_repeated_public_operation_makes_a_request(operation, monkeypatch):
    client = configured_client()
    request = Mock(return_value={"organic": [{"title": "result"}],
                                 "images_results": [{"title": "image"}],
                                 "visual_matches": [{"title": "match"}]})
    monkeypatch.setattr(client, "_request_json", request)
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    download = Mock(return_value=(buffer.getvalue(), "image/png"))
    if operation == "page":
        download.return_value = (b"Page evidence", "text/plain")
    monkeypatch.setattr(client, "_download_limited", download)
    operations = {
        "text": lambda: client.search("query"),
        "image": lambda: client.image_search("query"),
        "reverse": lambda: client.google_lens_search("https://pixels.example.test/a.png"),
        "page": lambda: client.fetch_page_document("https://page.example.test/a"),
        "pixels": lambda: client.fetch_image("https://pixels.example.test/a.png"),
    }
    for _ in range(2):
        assert operations[operation]() is not None
    assert (download if operation in {"page", "pixels"} else request).call_count == 2


def test_mei_calls_and_trace_capture_are_not_memoized(monkeypatch):
    monkeypatch.setattr("neos.workflow.mei.get_mei_trace_store", lambda: None)
    gateway = Mock(model="test-mei", fallback_models=())
    gateway.chat.return_value = '{"rational":"useful","evidence":"fact","summary":"fact","relevant_images":[]}'
    gateway.take_last_call.return_value = {"actual_model": "test-mei"}
    mei = NeosMEI(gateway)
    trace = Mock()
    trace.record.return_value = dict(trace_id="t", example_id="e", asset_count=0)
    mei.trace_store = trace
    for _ in range(2):
        assert mei.summarize_web(url="https://page.example.test", goal="question", page_text="fact", image_refs=[])["evidence"] == "fact"
        timing = mei.take_last_timing()
        assert timing["compute_called"] and timing["mei_trace_id"] == "t"
        assert not any("cache" in key for key in timing)
    assert gateway.chat.call_count == trace.record.call_count == 2


def test_webvisit_does_not_warm_image_downloads(monkeypatch):
    client = configured_client()
    client.fetch_page_document = Mock(return_value={"text": "evidence", "images": [{"url": "https://pixels.example.test/a.jpg", "alt": "clue"}]})
    client.fetch_image = Mock(side_effect=AssertionError("WebVisit must not download unused pixels"))
    runtime = NeosRuntime(client=client, mei=False)
    try:
        for _ in range(2):
            result = runtime.web_visit({"url": ["https://page.example.test"], "goal": "question"})
            assert result.metadata["n_fetched"] == 1
            assert not result.images
        assert client.fetch_page_document.call_count == 2
        client.fetch_image.assert_not_called()
    finally:
        runtime._text_search_executor.shutdown(wait=True)
