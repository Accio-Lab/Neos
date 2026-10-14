from __future__ import annotations
import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Any
from PIL import Image
from neos.protocol import VISIT_PAGE_TEXT_CHARS
from neos.tools.coordinates import normalize_qwen_bbox, qwen_bbox_to_pixels
from neos.tools.online_client import OnlineSearchClient
logger = logging.getLogger(__name__)
EXTRACTOR_PROMPT_VERSION = 'browsecomp-webpage-extractor'
EXTRACTOR_PROMPT = 'Please process the following webpage content and user goal to extract relevant information:\n\n## **Webpage Content**\n{webpage_content}\n\n## **User Goal**\n{goal}\n\n## **Task Guidelines**\n1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user\'s goal within the webpage content\n2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.\n3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.\n\n**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**\n'
DEFAULTS = {'search_top_k': 10, 'search_max_queries': 3, 'search_snippet_chars': 400, 'visit_max_pages': 3, 'visit_page_chars': VISIT_PAGE_TEXT_CHARS, 'visit_use_extractor': True, 'image_top_k': 3, 'image_max_side': 768, 'reverse_top_k': 5, 'reverse_query_max_side': 1024, 'reverse_return_images': 0, 'reverse_ref_max_side': 512, 'crop_min_size': 1, 'crop_upscale_to': 512, 'ocr_min_score': 0.4, 'ocr_max_lines': 50}

@dataclass
class ToolExecution:
    text: str
    images: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

def downscale(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert('RGB')
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / max(width, height)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

def resolve_image_ref(ref: Any, images: list[Image.Image]) -> Image.Image | None:
    pass
    if isinstance(ref, int):
        index = ref - 1
    else:
        text = str(ref or '').strip()
        if text.startswith('img_'):
            text = text[4:]
        if not text.isdigit():
            return None
        index = int(text) - 1
    if 0 <= index < len(images):
        image = images[index]
        if isinstance(image, Image.Image):
            return image.convert('RGB')
    return None

def image_ref_from_parameters(parameters: dict[str, Any]) -> str:
    pass
    ref = parameters.get('image')
    if ref not in (None, ''):
        return str(ref)
    index = parameters.get('image_index')
    try:
        return f'img_{int(index) + 1}'
    except (TypeError, ValueError):
        return 'img_1'

def _result_url(item: dict) -> str:
    return str(item.get('thumbnailUrl') or item.get('thumbnail_url') or item.get('thumbnail') or item.get('imageUrl') or item.get('image_url') or item.get('image') or item.get('original') or '')

def _source_url(item: dict) -> str:
    return str(item.get('link') or item.get('sourceUrl') or item.get('source_url') or item.get('url') or '')
_default_extractor = None
_default_extractor_lock = threading.Lock()

def _get_default_extractor():
    global _default_extractor
    if _default_extractor is None:
        with _default_extractor_lock:
            if _default_extractor is None:
                from neos.eval import secrets
                from neos.eval.gateway import LLMGateway
                _default_extractor = LLMGateway(model=secrets.EXTRACTOR_MODEL)
    return _default_extractor

class OnlineToolRuntime:
    pass

    def __init__(self, client: OnlineSearchClient | None=None, extractor=None, config: dict[str, Any] | None=None):
        self.client = client or OnlineSearchClient()
        self.extractor = extractor
        self.config = {**DEFAULTS, **(config or {})}

    def search(self, parameters: dict[str, Any]) -> ToolExecution:
        value = parameters.get('query') or parameters.get('q') or ''
        queries = [value] if isinstance(value, str) else list(value)
        queries = [query.strip() for query in queries if isinstance(query, str) and query.strip()][:int(self.config['search_max_queries'])]
        if not queries:
            return ToolExecution("Error: 'query' must be a non-empty string or array of strings.")
        top_k = int(self.config['search_top_k'])
        snippet_chars = int(self.config['search_snippet_chars'])
        blocks: list[str] = []
        successful = 0
        for query in queries:
            try:
                results = self.client.search(query, num=top_k)
            except Exception as exc:
                blocks.append(f"Search for '{query}' failed: {type(exc).__name__}: {exc}")
                continue
            successful += 1
            if not results:
                blocks.append(f"No search results found for '{query}'.")
                continue
            lines = [f"Search results for '{query}':"]
            for index, result in enumerate(results[:top_k], 1):
                title = result.get('title') or 'Untitled'
                url = result.get('link') or result.get('url') or ''
                snippet = str(result.get('snippet') or '')[:snippet_chars]
                lines.append(f'{index}. {title}\n   URL: {url}\n   {snippet}')
            blocks.append('\n'.join(lines))
        return ToolExecution('\n\n'.join(blocks), metadata={'n_queries': len(queries), 'n_successful': successful})

    def _extractor_instance(self):
        if self.extractor is False:
            return None
        if self.extractor is not None:
            return self.extractor
        if not self.config.get('visit_use_extractor', True):
            return None
        return _get_default_extractor()

    def visit(self, parameters: dict[str, Any]) -> ToolExecution:
        value = parameters.get('url') or ''
        urls = [value] if isinstance(value, str) else list(value)
        urls = [url.strip() for url in urls if isinstance(url, str) and url.strip().lower().startswith(('http://', 'https://'))][:int(self.config['visit_max_pages'])]
        goal = str(parameters.get('goal') or '').strip()
        if not urls:
            return ToolExecution("Error: 'url' must be a complete http(s) URL or array of URLs.")
        outputs: list[str] = []
        fetched = 0
        for url in urls:
            content = self.client.fetch_page_text(url, max_chars=int(self.config['visit_page_chars']))
            if not content:
                outputs.append(f'The webpage at {url} could not be fetched (empty or blocked).')
                continue
            fetched += 1
            try:
                extractor = self._extractor_instance()
            except Exception as exc:
                extractor = None
                logger.warning('visit extractor unavailable: %s', exc)
            if extractor is None:
                outputs.append(f'Content of {url}:\n{content}')
                continue
            extractor_id = str(getattr(extractor, 'model', None) or type(extractor).__qualname__)

            def compute() -> str:
                return extractor.chat([{'role': 'user', 'content': EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}], max_completion_tokens=4096)
            try:
                extracted = compute()
                outputs.append(f'Summary of {url}:\n{extracted}')
            except Exception as exc:
                outputs.append(f'Raw content of {url} (extractor failed: {type(exc).__name__}: {exc}):\n{content[:3000]}')
        return ToolExecution('\n\n'.join(outputs), metadata={'n_pages': len(urls), 'n_fetched': fetched})

    def image_search(self, parameters: dict[str, Any]) -> ToolExecution:
        query = str(parameters.get('query') or '').strip()
        if not query:
            return ToolExecution("Error: 'query' is required.")
        try:
            top_k = max(1, min(int(parameters.get('top_k') or self.config['image_top_k']), 5))
        except (TypeError, ValueError):
            return ToolExecution("Error: 'top_k' must be an integer.")
        try:
            results = self.client.image_search(query, num=max(top_k * 2, 6))
        except Exception as exc:
            return ToolExecution(f"Image search failed for '{query}': {type(exc).__name__}: {exc}")
        images: list[Image.Image] = []
        image_urls: list[str] = []
        lines: list[str] = []
        for result in results:
            if len(images) >= top_k:
                break
            image_url = _result_url(result)
            if not image_url:
                continue
            image = self.client.fetch_image(image_url)
            if image is None:
                continue
            images.append(downscale(image, int(self.config['image_max_side'])))
            image_urls.append(image_url)
            source_url = _source_url(result)
            title = result.get('title') or 'Untitled'
            source = result.get('source') or ''
            line = f'- image {len(images)}: {title}'
            if source:
                line += f' (source: {source})'
            if source_url:
                line += f'\n  Source page: {source_url}'
            line += f'\n  Direct image URL: {image_url}'
            lines.append(line)
        if not images:
            return ToolExecution(f"No usable online images could be downloaded for '{query}'.")
        return ToolExecution(f"Image search for '{query}' returned {len(images)} images (shown above, in order):\n" + '\n'.join(lines), images=images, metadata={'n_images': len(images), 'image_urls': image_urls})

    @staticmethod
    def _jpeg_bytes(image: Image.Image, max_side: int) -> bytes:
        buffer = io.BytesIO()
        downscale(image, max_side).save(buffer, format='JPEG', quality=90, optimize=False)
        return buffer.getvalue()

    def reverse_image_search(self, parameters: dict[str, Any], image: Image.Image) -> ToolExecution:
        try:
            top_k = max(1, min(int(parameters.get('top_k') or self.config['reverse_top_k']), 10))
        except (TypeError, ValueError):
            return ToolExecution("Error: 'top_k' must be an integer.")
        image_bytes = self._jpeg_bytes(image, int(self.config['reverse_query_max_side']))
        source_url = parameters.get('_source_url')
        if not source_url:
            ref = parameters.get('image')
            if isinstance(ref, str) and ref.startswith(('http://', 'https://')):
                source_url = ref
        try:
            results = self.client.reverse_image_search(image_bytes, top_k=top_k, source_url=source_url, query=str(parameters.get('query') or '').strip() or None)
        except Exception as exc:
            return ToolExecution(f'Reverse image search failed: {type(exc).__name__}: {exc}')
        if not results:
            return ToolExecution('Reverse image search returned no online matches.')
        lines: list[str] = []
        references: list[Image.Image] = []
        reference_urls: list[str] = []
        for index, result in enumerate(results[:top_k], 1):
            title = result.get('title') or result.get('name') or 'Untitled'
            link = _source_url(result)
            snippet = result.get('snippet') or result.get('description') or ''
            score = result.get('score')
            head = f'{index}. {title}'
            if score is not None:
                try:
                    head += f' (score {float(score):.3f})'
                except (TypeError, ValueError):
                    pass
            if link:
                head += f'\n   URL: {link}'
            if snippet:
                head += f'\n   {str(snippet)[:400]}'
            lines.append(head)
            if len(references) < int(self.config['reverse_return_images']):
                image_url = _result_url(result)
                if image_url:
                    reference = self.client.fetch_image(image_url)
                    if reference is not None:
                        references.append(downscale(reference, int(self.config['reverse_ref_max_side'])))
                        reference_urls.append(image_url)
        return ToolExecution('Google Lens reverse image search matches:\n' + '\n'.join(lines), images=references, metadata={'n_matches': len(lines), 'n_images': len(references), 'image_urls': reference_urls, 'engine': 'google_lens'})

    def zoomin_search(self, parameters: dict[str, Any], image: Image.Image) -> ToolExecution:
        pass
        crop_result = self.crop(parameters, image)
        if not crop_result.images:
            return crop_result
        zoomed = crop_result.images[0]
        reverse_parameters = {'image': 'zoomin_crop', 'top_k': parameters.get('top_k', self.config['reverse_top_k'])}
        if parameters.get('query'):
            reverse_parameters['query'] = parameters['query']
        reverse_result = self.reverse_image_search(reverse_parameters, zoomed)
        metadata = {**crop_result.metadata, **reverse_result.metadata, 'operation': 'zoomin_search', 'n_images': 1 + len(reverse_result.images)}
        return ToolExecution(crop_result.text + '\n\n' + reverse_result.text, images=[zoomed, *reverse_result.images], metadata=metadata)

    @staticmethod
    def _normalized_bbox(parameters: dict[str, Any]) -> tuple[float, float, float, float] | None:
        bbox = parameters.get('bbox_2d')
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return tuple((float(value) for value in bbox))
            except (TypeError, ValueError):
                return None
        required = ('x', 'y', 'width', 'height')
        if all((key in parameters for key in required)):
            try:
                x = float(parameters['x'])
                y = float(parameters['y'])
                return (x, y, x + float(parameters['width']), y + float(parameters['height']))
            except (TypeError, ValueError):
                return None
        return None

    def crop(self, parameters: dict[str, Any], image: Image.Image) -> ToolExecution:
        bbox = self._normalized_bbox(parameters)
        if bbox is None:
            return ToolExecution("Error: 'bbox_2d' must be [left, top, right, bottom] using coordinates from 0 to 1000.")
        try:
            x1n, y1n, x2n, y2n = normalize_qwen_bbox(bbox)
        except ValueError:
            return ToolExecution('Error: bbox_2d must have right > left and bottom > top.')
        width, height = image.size
        left, top, right, bottom = qwen_bbox_to_pixels(image, (x1n, y1n, x2n, y2n))
        min_size = int(self.config['crop_min_size'])
        if right - left < min_size:
            right = min(width, left + min_size)
            left = max(0, right - min_size)
        if bottom - top < min_size:
            bottom = min(height, top + min_size)
            top = max(0, bottom - min_size)
        crop = image.convert('RGB').crop((left, top, right, bottom))
        crop_width, crop_height = crop.size
        upscale_to = int(self.config['crop_upscale_to'])
        if upscale_to > 0 and max(crop_width, crop_height) < upscale_to:
            scale = upscale_to / max(crop_width, crop_height)
            crop = crop.resize((max(1, int(crop_width * scale)), max(1, int(crop_height * scale))), Image.Resampling.LANCZOS)
        normalized = [int(round(x1n)), int(round(y1n)), int(round(x2n)), int(round(y2n))]
        return ToolExecution(f'Zoomed bbox_2d={normalized} (normalized 0–1000) maps to pixel box [{left}, {top}, {right}, {bottom}] in the original {width}x{height} image; the region is shown above.', images=[crop], metadata={'bbox_2d': normalized, 'pixel_box': [left, top, right, bottom], 'source_size': [width, height]})
    _ocr_engine = None
    _ocr_lock = threading.Lock()

    @classmethod
    def _get_ocr_engine(cls):
        if cls._ocr_engine is None:
            with cls._ocr_lock:
                if cls._ocr_engine is None:
                    from rapidocr_onnxruntime import RapidOCR
                    cls._ocr_engine = RapidOCR()
        return cls._ocr_engine

    def ocr(self, parameters: dict[str, Any], image: Image.Image) -> ToolExecution:
        min_score = float(self.config['ocr_min_score'])
        max_lines = int(self.config['ocr_max_lines'])

        def compute() -> list[str]:
            import numpy as np
            result, _ = self._get_ocr_engine()(np.array(image.convert('RGB')))
            return [text.strip() for _box, text, score in result or [] if score >= min_score and text.strip()][:max_lines]
        try:
            lines = compute()
        except Exception as exc:
            return ToolExecution(f'OCR failed: {type(exc).__name__}: {exc}')
        ref = image_ref_from_parameters(parameters)
        if not lines:
            return ToolExecution(f'No text detected in {ref}.')
        return ToolExecution(f'OCR result for {ref} ({len(lines)} text regions, reading order):\n' + '\n'.join(lines), metadata={'n_lines': len(lines)})
