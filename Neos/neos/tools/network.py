from __future__ import annotations
from neos.identifiers import new_id
import datetime
import html
from html.parser import HTMLParser
import importlib.util
import io
import json
import logging
import os
from dataclasses import dataclass
import re
import threading
import time
from urllib.parse import quote, urljoin, urlparse
import requests
from PIL import Image
from neos.protocol import VISIT_PAGE_TEXT_CHARS
logger = logging.getLogger(__name__)

class OnlineSearchConfigurationError(ValueError):
    pass

class ReverseImageSearchUnavailable(RuntimeError):
    pass

class ImageHostingUnavailable(ReverseImageSearchUnavailable):
    pass

class ImageStagingPreflightError(ImageHostingUnavailable):
    pass

@dataclass(frozen=True)
class ImageFetchResult:
    image: Image.Image | None
    attempts: int

def _valid_http_url(value):
    parsed = urlparse(str(value))
    return parsed.scheme in {'http', 'https'} and bool(parsed.netloc)

class _PageImages(HTMLParser):

    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'img':
            return
        attrs = dict(attrs)
        value = attrs.get('src') or attrs.get('data-src') or attrs.get('data-original') or ''
        if not value and attrs.get('srcset'):
            value = attrs['srcset'].split(',')[0].strip().split(' ')[0]
        value = urljoin(self.url, value)
        if _valid_http_url(value):
            self.images.append({'url': value, 'alt': attrs.get('alt') or attrs.get('title') or ''})

class OnlineSearchClient:
    pass

    def __init__(self, base_url=None, timeout=30, *, api_key=None, serper_endpoint=None, serper_api_key=None, reader_url=None, reader_api_key=None, image_upload_url=None, image_upload_api_key=None, image_upload_provider=None, oss_bucket=None, oss_region=None, oss_endpoint=None, oss_public_base_url=None, oss_public_read=None, oss_prefix=None, oss_presign_ttl=None, max_retries=3):
        self.base_url = base_url if base_url is not None else os.getenv('NEOS_IMAGE_SEARCH_ENDPOINT', '')
        self.api_key = api_key if api_key is not None else os.getenv('NEOS_IMAGE_SEARCH_API_KEY', '')
        self.serper_endpoint = serper_endpoint if serper_endpoint is not None else os.getenv('NEOS_SERPER_ENDPOINT', '')
        self.serper_api_key = serper_api_key if serper_api_key is not None else os.getenv('SERPER_API_KEY', '')
        self.reader_url = reader_url if reader_url is not None else os.getenv('NEOS_JINA_ENDPOINT', '')
        self.reader_api_key = reader_api_key if reader_api_key is not None else os.getenv('JINA_API_KEY', '')
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self.image_upload_url = image_upload_url if image_upload_url is not None else os.getenv('NEOS_IMAGE_UPLOAD_URL', '')
        self.image_upload_api_key = image_upload_api_key if image_upload_api_key is not None else os.getenv('NEOS_IMAGE_UPLOAD_API_KEY', '')
        self.oss_bucket = oss_bucket if oss_bucket is not None else os.getenv('NEOS_OSS_BUCKET', '')
        self.oss_region = oss_region if oss_region is not None else os.getenv('NEOS_OSS_REGION', '')
        self.oss_endpoint = oss_endpoint if oss_endpoint is not None else os.getenv('NEOS_OSS_ENDPOINT', '')
        self.oss_public_base_url = oss_public_base_url if oss_public_base_url is not None else os.getenv('NEOS_OSS_PUBLIC_BASE_URL', '')
        self.oss_public_read = bool(oss_public_read) if oss_public_read is not None else os.getenv('NEOS_OSS_PUBLIC_READ', '0') == '1'
        self.oss_prefix = (oss_prefix if oss_prefix is not None else os.getenv('NEOS_OSS_PREFIX', 'neos-images')).strip('/')
        self.oss_presign_ttl = int(oss_presign_ttl or os.getenv('NEOS_OSS_PRESIGN_TTL', '3600'))
        if not 300 <= self.oss_presign_ttl <= 604800:
            raise ValueError('OSS signed URL lifetime must be300–604800 seconds')
        provider = image_upload_provider if image_upload_provider is not None else os.getenv('NEOS_IMAGE_UPLOAD_PROVIDER', 'auto')
        if provider == 'auto':
            provider = 'http' if self.image_upload_url else 'oss' if self.oss_bucket else 'none'
        if provider not in {'none', 'http', 'oss'}:
            raise ValueError('Unknown image upload provider')
        self.image_upload_provider = provider
        self.oss_sdk_available = importlib.util.find_spec('alibabacloud_oss_v2') is not None
        self.oss_credentials_configured = bool(os.getenv('OSS_ACCESS_KEY_ID') and os.getenv('OSS_ACCESS_KEY_SECRET'))
        self._tls = threading.local()
        self._stats_lock = threading.Lock()
        self._stats = {'requests': 0, 'failures': 0, 'inflight': 0, 'peak_inflight': 0}
        limit = int(os.getenv('NEOS_SEARCH_CONCURRENCY', '128'))
        if limit < 1:
            raise ValueError('Search concurrency must be positive')
        self._admission = threading.BoundedSemaphore(limit)

    def _session(self):
        if not hasattr(self._tls, 'session'):
            self._tls.session = requests.Session()
            self._tls.session.trust_env = False
        return self._tls.session

    def require_interfaces(self):
        self._require(self.serper_endpoint, 'NEOS_SERPER_ENDPOINT', self.serper_api_key)
        self._require(self.reader_url, 'NEOS_JINA_ENDPOINT')
        self._require(self.base_url, 'NEOS_IMAGE_SEARCH_ENDPOINT', self.api_key)

    def _require(self, endpoint, name, key=None):
        if not endpoint or not _valid_http_url(endpoint):
            raise OnlineSearchConfigurationError(f'Configure{name} with a complete HTTP(S) endpoint')
        if key is not None and (not key):
            raise OnlineSearchConfigurationError(f'Configure the credential for{name}')

    def _request_json(self, method, endpoint, *, headers=None, params=None, payload=None):
        last = None
        for attempt in range(self.max_retries):
            try:
                with self._admission:
                    with self._stats_lock:
                        self._stats['requests'] += 1
                        self._stats['inflight'] += 1
                        self._stats['peak_inflight'] = max(self._stats['peak_inflight'], self._stats['inflight'])
                    try:
                        response = self._session().request(method, endpoint, headers=headers, params=params, json=payload, timeout=self.timeout)
                        response.raise_for_status()
                        result = response.json()
                        if not isinstance(result, dict):
                            raise ValueError('Provider response must be a JSON object')
                        return result
                    finally:
                        with self._stats_lock:
                            self._stats['inflight'] -= 1
            except (requests.RequestException, ValueError) as error:
                last = type(error).__name__
                with self._stats_lock:
                    self._stats['failures'] += 1
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f'Online provider request failed after bounded retries ({last})') from None

    def search(self, q, num=10, tbs=None, tbm=None):
        self._require(self.serper_endpoint, 'NEOS_SERPER_ENDPOINT', self.serper_api_key)
        payload = {'q': str(q).strip(), 'num': int(num), 'gl': 'us', 'hl': 'en'}
        if tbs:
            payload['tbs'] = tbs
        if tbm:
            payload['tbm'] = tbm
        result = self._request_json('POST', self.serper_endpoint, headers={'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}, payload=payload)
        rows = result.get('organic', result.get('organic_results', []))
        return rows if isinstance(rows, list) else []

    def _image_gateway(self, params):
        self._require(self.base_url, 'NEOS_IMAGE_SEARCH_ENDPOINT', self.api_key)
        return self._request_json('GET', self.base_url, headers={'x-api-key': self.api_key}, params=params)

    def image_search(self, q, num=10):
        result = self._image_gateway({'engine': 'google_images', 'q': str(q).strip(), 'gl': 'us', 'hl': 'en', 'num': int(num)})
        rows = result.get('images_results', result.get('images', []))
        return rows if isinstance(rows, list) else []

    def _download_limited(self, url, *, timeout, max_bytes, extra_headers=None):
        if not _valid_http_url(url):
            raise ValueError('Expected a complete HTTP(S) URL')
        headers = {'User-Agent': 'Neos/1.0'}
        headers.update(extra_headers or {})
        with self._session().get(url, headers=headers, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError('Response exceeds the configured byte limit')
                chunks.append(chunk)
            return (b''.join(chunks), response.headers.get('content-type', ''))

    def _fetch_page_payload(self, url, *, timeout=20):
        self._require(self.reader_url, 'NEOS_JINA_ENDPOINT')
        if not _valid_http_url(url):
            raise ValueError('Invalid webpage URL')
        endpoint = self.reader_url.replace('{url}', url) if '{url}' in self.reader_url else self.reader_url.rstrip('/') + '/' + url
        headers = {'Authorization': 'Bearer ' + self.reader_api_key} if self.reader_api_key else {}
        try:
            return self._download_limited(endpoint, timeout=max(timeout, 60), max_bytes=int(os.getenv('NEOS_MAX_PAGE_BYTES', str(8 * 1024 * 1024))), extra_headers=headers)
        except (requests.RequestException, ValueError):
            logger.warning('Page reader request failed')
            return None

    @staticmethod
    def _html_to_text(raw, content_type):
        text = raw.decode('utf-8', errors='replace')
        if 'html' in content_type.lower() or '<html' in text[:1000].lower():
            text = re.sub('(?is)<(script|style|noscript|header|footer|nav|form)[^>]*>.*?</\\1>', ' ', text)
            text = html.unescape(re.sub('<[^>]+>', ' ', text))
        return re.sub('\\s+', ' ', text).strip()

    @staticmethod
    def _extract_page_image_refs(raw, content_type, page_url, *, max_images):
        text = raw.decode('utf-8', errors='replace')
        if 'html' in content_type.lower() or '<html' in text[:1000].lower():
            parser = _PageImages(page_url)
            parser.feed(text)
            candidates = parser.images
        else:
            candidates = [{'alt': m[0], 'url': m[1]} for m in re.findall('!\\[([^\\]]*)\\]\\((https?://[^\\s)]+)(?:\\s+[^)]*)?\\)', text)]
        result = []
        seen = set()
        for row in candidates:
            if len(result) >= max(0, int(max_images)):
                break
            if row['url'] in seen or not _valid_http_url(row['url']):
                continue
            seen.add(row['url'])
            result.append(row)
        return result

    def fetch_page_document(self, url, *, max_chars=VISIT_PAGE_TEXT_CHARS, max_images=15, timeout=20):
        payload = self._fetch_page_payload(str(url).strip(), timeout=timeout)
        if payload is None:
            return {'url': url, 'text': '', 'images': []}
        raw, kind = payload
        return {'url': url, 'text': self._html_to_text(raw, kind)[:int(max_chars)], 'images': self._extract_page_image_refs(raw, kind, url, max_images=max_images), 'content_type': kind}

    def fetch_page_text(self, url, max_chars=VISIT_PAGE_TEXT_CHARS, timeout=20):
        return self.fetch_page_document(url, max_chars=max_chars, max_images=0, timeout=timeout)['text']

    @staticmethod
    def _decode_image_bytes(raw):
        if not raw:
            return None
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                return image.convert('RGB')
        except (OSError, ValueError):
            return None

    def fetch_image_bytes(self, url, timeout=15):
        try:
            raw, _ = self._download_limited(url, timeout=timeout, max_bytes=int(os.getenv('NEOS_MAX_IMAGE_BYTES', str(32 * 1024 * 1024))))
            return raw if self._decode_image_bytes(raw) is not None else None
        except (requests.RequestException, ValueError):
            return None

    def fetch_image_with_retries(self, url, *, attempts=1, timeout=15):
        for attempt in range(1, max(1, int(attempts)) + 1):
            image = self._decode_image_bytes(self.fetch_image_bytes(url, timeout=timeout))
            if image is not None:
                return ImageFetchResult(image, attempt)
        return ImageFetchResult(None, max(1, int(attempts)))

    def fetch_image(self, url, timeout=15):
        return self.fetch_image_with_retries(url, timeout=timeout).image

    def _image_staging_configured(self):
        return bool(self.image_upload_provider == 'http' and self.image_upload_url or (self.image_upload_provider == 'oss' and self.oss_bucket and self.oss_region and self.oss_endpoint))

    def _image_staging_ready(self):
        return self._image_staging_configured() and (self.image_upload_provider != 'oss' or (self.oss_sdk_available and self.oss_credentials_configured))

    @staticmethod
    def _extract_uploaded_url(data):
        if isinstance(data, str):
            return data if _valid_http_url(data) else ''
        if not isinstance(data, dict):
            return ''
        for key in ['url', 'image_url', 'public_url', 'imageUrl', 'publicUrl']:
            if isinstance(data.get(key), str) and _valid_http_url(data[key]):
                return data[key]
        for key in ['data', 'result', 'payload']:
            value = OnlineSearchClient._extract_uploaded_url(data.get(key))
            if value:
                return value
        return ''

    def _oss_client(self):
        if not hasattr(self._tls, 'oss_client'):
            if not self.oss_sdk_available or not self.oss_credentials_configured:
                raise ImageHostingUnavailable('Configure OSS credentials and install its SDK')
            import alibabacloud_oss_v2 as oss
            config = oss.config.load_default()
            config.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
            config.region = self.oss_region
            config.endpoint = self.oss_endpoint
            self._tls.oss_client = oss.Client(config)
        return self._tls.oss_client

    def _put_image_oss(self, image_bytes, digest):
        import alibabacloud_oss_v2 as oss
        key = f'{self.oss_prefix}/{digest[:2]}/{digest[2:4]}/{digest}.jpg'
        self._oss_client().put_object(oss.PutObjectRequest(bucket=self.oss_bucket, key=key, body=image_bytes, content_type='image/jpeg'))
        return key

    def _oss_image_url(self, key):
        if self.oss_public_read:
            self._require(self.oss_public_base_url, 'NEOS_OSS_PUBLIC_BASE_URL')
            return self.oss_public_base_url.rstrip('/') + '/' + quote(key, safe='/')
        import alibabacloud_oss_v2 as oss
        return str(self._oss_client().presign(oss.GetObjectRequest(bucket=self.oss_bucket, key=key), expires=datetime.timedelta(seconds=self.oss_presign_ttl)).url)

    def _stage_image_http(self, image_bytes):
        self._require(self.image_upload_url, 'NEOS_IMAGE_UPLOAD_URL')
        headers = {'Content-Type': 'application/octet-stream'}
        if self.image_upload_api_key:
            headers['x-api-key'] = self.image_upload_api_key
        response = self._session().post(self.image_upload_url, headers=headers, data=image_bytes, timeout=max(60, self.timeout))
        response.raise_for_status()
        url = self._extract_uploaded_url(response.json())
        if not url:
            raise ImageHostingUnavailable('Image upload response does not contain a public URL')
        return url

    def stage_image(self, image_bytes, *, source_url=None):
        if source_url and _valid_http_url(source_url):
            return source_url
        if not self._image_staging_ready():
            raise ImageHostingUnavailable('Configure image hosting for local reverse-search images')
        try:
            if self.image_upload_provider == 'http':
                return self._stage_image_http(image_bytes)
            key = self._put_image_oss(image_bytes, new_id())
            return self._oss_image_url(key)
        except Exception as error:
            raise ImageHostingUnavailable('Image upload failed; check the configured provider') from None

    def google_lens_search(self, image_url, *, top_k=5, query=None, search_type='all'):
        if not _valid_http_url(image_url):
            raise ValueError('Lens requires a public HTTP(S) image URL')
        if search_type not in {'all', 'about_this_image', 'products', 'exact_matches', 'visual_matches'}:
            raise ValueError('Unsupported Lens search type')
        params = {'engine': 'google_lens', 'url': image_url, 'hl': 'en', 'country': 'us', 'type': search_type}
        if query:
            params['q'] = str(query).strip()
        data = self._image_gateway(params)
        rows = data.get('visual_matches') or data.get('exact_matches') or []
        return rows[:int(top_k)] if isinstance(rows, list) else []

    def reverse_image_search(self, image_bytes, top_k=5, *, source_url=None, query=None):
        url = self.stage_image(image_bytes, source_url=source_url)
        return self.google_lens_search(url, top_k=top_k, query=query)

    def live_image_staging_preflight(self):
        if not self._image_staging_ready():
            raise ImageStagingPreflightError('Image hosting is not configured')
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), 'white').save(buffer, format='JPEG')
        raw = buffer.getvalue()
        url = self.stage_image(raw)
        if self.fetch_image(url, timeout=30) is None:
            raise ImageStagingPreflightError('Uploaded image could not be fetched')
        return {'ok': True, 'provider': self.image_upload_provider}

    def config_status(self):
        settings = {'text': 'serper', 'reader': 'jina', 'images': 'gateway', 'text_endpoint': self.serper_endpoint, 'reader_endpoint': self.reader_url, 'image_endpoint': self.base_url, 'image_hosting_provider': self.image_upload_provider, 'image_upload_url': self.image_upload_url}
        return {'search_available': bool(self.serper_endpoint and self.serper_api_key), 'google_lens_configured': bool(self.base_url and self.api_key), 'image_search_available': bool(self.base_url and self.api_key), 'image_upload_provider': self.image_upload_provider, 'image_staging_configured': self._image_staging_configured(), 'image_staging_ready': self._image_staging_ready(), 'local_reverse_image_ready': bool(self.base_url and self.api_key and self._image_staging_ready()), 'reverse_image_search_available': bool(self.base_url and self.api_key), 'oss_sdk_available': self.oss_sdk_available}

    def provider_status(self):
        with self._stats_lock:
            return dict(self._stats, search_provider='serper', page_provider='jina', image_provider='gateway')
SearchGateway = OnlineSearchClient
