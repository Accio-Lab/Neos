import io
from unittest.mock import Mock
from PIL import Image
import pytest
from neos.tools.network import OnlineSearchClient,OnlineSearchConfigurationError


def client():
    return OnlineSearchClient(serper_endpoint='https://text.example.test/search',serper_api_key='unit-only',
        base_url='https://image.example.test/search',api_key='unit-only',
        reader_url='https://reader.example.test',reader_api_key='unit-only')


def test_unconfigured_provider_has_no_default_route(monkeypatch):
    for key in ['NEOS_SERPER_ENDPOINT','SERPER_API_KEY','NEOS_IMAGE_SEARCH_ENDPOINT','NEOS_IMAGE_SEARCH_API_KEY','NEOS_JINA_ENDPOINT']:
        monkeypatch.delenv(key,raising=False)
    c=OnlineSearchClient()
    with pytest.raises(OnlineSearchConfigurationError):c.search('x')
    with pytest.raises(OnlineSearchConfigurationError):c.image_search('x')
    with pytest.raises(OnlineSearchConfigurationError):c.fetch_page_document('https://page.example.test')


def test_serper_text_contract():
    c=client();response=Mock();response.json.return_value={'organic':[{'title':'A','link':'https://a.example.test','snippet':'B'}]}
    session=Mock();session.request.return_value=response;c._tls.session=session
    assert c.search('target',num=5)[0]['title']=='A'
    c.search('target',num=5)
    assert session.request.call_count==2
    args,kwargs=session.request.call_args
    assert args==('POST','https://text.example.test/search')
    assert kwargs['json']['q']=='target' and kwargs['headers']['X-API-KEY']=='unit-only'


def test_image_and_reverse_share_one_gateway(monkeypatch):
    c=client();calls=[]
    def request(method,url,**kw):
        calls.append((method,url,kw))
        return {'images_results':[{'original':'https://image.example.test/a.jpg'}],
                'visual_matches':[{'link':'https://source.example.test'}]}
    monkeypatch.setattr(c,'_request_json',request)
    c.image_search('target')
    c.google_lens_search('https://pixels.example.test/a.jpg')
    assert {row[1] for row in calls}=={'https://image.example.test/search'}
    assert [row[2]['params']['engine'] for row in calls]==['google_images','google_lens']


def test_jina_normalizes_text_and_image_references(monkeypatch):
    c=client();seen=[]
    def get(url,**kw):
        seen.append((url,kw));return b'Title\n![clue](https://pixels.example.test/a.jpg)\nEvidence','text/markdown'
    monkeypatch.setattr(c,'_download_limited',get)
    for _ in range(2):
        doc=c.fetch_page_document('https://page.example.test/item')
        assert doc['images']==[{'url':'https://pixels.example.test/a.jpg','alt':'clue'}]
    assert len(seen)==2 and seen[0][0]=='https://reader.example.test/https://page.example.test/item'


def test_repeated_downloads_really_request_pixels(monkeypatch):
    c=client();buffer=io.BytesIO();Image.new('RGB',(16,16),'red').save(buffer,format='PNG')
    fetch=Mock(return_value=(buffer.getvalue(),'image/png'));monkeypatch.setattr(c,'_download_limited',fetch)
    assert c.fetch_image('https://pixels.example.test/a.png') is not None
    assert c.fetch_image('https://pixels.example.test/a.png') is not None
    assert fetch.call_count==2
