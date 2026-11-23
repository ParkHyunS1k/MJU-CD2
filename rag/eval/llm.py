"""평가 스크립트용 Gemini 호출. 서버 설정(app.core.config)에 의존하지 않는다."""

import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-2.5-flash"
_RETRY_WAITS = (15, 30, 60, 60)  # 무료 등급 한도는 분 단위로 풀린다
_RETRYABLE = {429, 500, 503}

log = logging.getLogger(__name__)


def get_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _status(error: Exception) -> int | None:
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def call_with_retry(fn):
    """429/500/503이면 기다렸다 다시 호출한다."""
    for attempt, wait in enumerate((*_RETRY_WAITS, None)):
        try:
            return fn()
        except Exception as e:
            if wait is None or _status(e) not in _RETRYABLE:
                raise
            log.warning("Gemini %s, retry %d in %ss", _status(e), attempt + 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def generate_json(client: genai.Client, prompt: str, schema: type[BaseModel]) -> BaseModel:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    res = call_with_retry(lambda: client.models.generate_content(model=MODEL, contents=prompt, config=config))
    return schema.model_validate_json(res.text)
