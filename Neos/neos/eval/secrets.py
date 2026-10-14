
import json
import os
from urllib.parse import urlparse


def endpoint(value, setting):
    if not value or not str(value).strip():
        raise ValueError(f"Configure {setting}; no model endpoint is bundled")
    parsed=urlparse(value)
    if parsed.scheme not in {"http","https"} or not parsed.netloc:
        raise ValueError(f"{setting} must be an HTTP(S) endpoint")
    return value


def request_headers(provider):
    value=os.getenv(f"NEOS_{provider}_HEADERS","")
    result=json.loads(value) if value.strip() else {}
    if not isinstance(result,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in result.items()):
        raise ValueError(f"NEOS_{provider}_HEADERS must be a JSON object of strings")
    return result


AZURE_ENDPOINT=os.getenv("NEOS_AZURE_ENDPOINT","")
AZURE_API_KEY=os.getenv("NEOS_AZURE_API_KEY","")
AZURE_API_VERSION=os.getenv("NEOS_AZURE_API_VERSION","")
DEEPSEEK_ENDPOINT=os.getenv("NEOS_DEEPSEEK_ENDPOINT","")
DEEPSEEK_API_KEY=os.getenv("NEOS_DEEPSEEK_API_KEY","")
DASHSCOPE_ENDPOINT=os.getenv("NEOS_DASHSCOPE_ENDPOINT","")
DASHSCOPE_API_KEY=os.getenv("NEOS_DASHSCOPE_API_KEY","")
GEMINI_ENDPOINT=os.getenv("NEOS_GEMINI_ENDPOINT","")
GEMINI_API_KEY=os.getenv("NEOS_GEMINI_API_KEY","")
RELAY_CHAT_ENDPOINT=os.getenv("NEOS_RELAY_CHAT_ENDPOINT","")
RELAY_CHAT_API_KEY=os.getenv("NEOS_RELAY_CHAT_API_KEY","")
RELAY_RESPONSES_ENDPOINT=os.getenv("NEOS_RELAY_RESPONSES_ENDPOINT","")
RELAY_RESPONSES_API_KEY=os.getenv("NEOS_RELAY_RESPONSES_API_KEY","")
LOCAL_MEI_ENDPOINT=os.getenv("NEOS_LOCAL_MEI_ENDPOINT","")
LOCAL_MEI_API_KEY=os.getenv("NEOS_LOCAL_MEI_API_KEY","")

JUDGE_MODEL=os.getenv("NEOS_JUDGE_MODEL","qwen3.7-plus")
EXTRACTOR_MODEL=os.getenv("NEOS_EXTRACTOR_MODEL","qwen3.7-flash")
DASHSCOPE_MEI_MODEL=os.getenv("NEOS_MEI_MODEL_SPEC","local/neos-mei")
DASHSCOPE_MEI_FALLBACK_MODELS=tuple(x.strip() for x in os.getenv("NEOS_MEI_FALLBACK_MODELS","").split(",") if x.strip())
DEEPSEEK_JUDGE_MIN_COMPLETION_TOKENS=int(os.getenv("NEOS_DEEPSEEK_JUDGE_MIN_COMPLETION_TOKENS","8192"))
DASHSCOPE_MIN_COMPLETION_TOKENS=int(os.getenv("NEOS_DASHSCOPE_MIN_COMPLETION_TOKENS","8192"))
RELAY_REASONING_EFFORT=os.getenv("NEOS_RELAY_REASONING_EFFORT","high")
RELAY_TIMEOUT_SECONDS=int(os.getenv("NEOS_RELAY_TIMEOUT_SECONDS","300"))
SEARCH_BASE_URL=os.getenv("NEOS_IMAGE_SEARCH_ENDPOINT","")
SEARCH_API_KEY_ENV=os.getenv("NEOS_IMAGE_SEARCH_API_KEY","")
