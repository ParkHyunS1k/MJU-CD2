"""평가 스크립트용 Gemini 호출. 서버 설정(app.core.config)에 의존하지 않는다."""

import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-2.5-flash"
_RETRY_WAITS = (5, 10, 20, 40)
_RETRYABLE = {429, 500, 503}

log = logging.getLogger(__name__)


def get_client() -> genai.Client:
    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _status(error: Exception) -> int | None:
    return getattr(error, "code", None) or getattr(error, "status_code", None)


def generate_json(client: genai.Client, prompt: str, schema: type[BaseModel]) -> BaseModel:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt, wait in enumerate((*_RETRY_WAITS, None)):
        try:
            res = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            return schema.model_validate_json(res.text)
        except Exception as e:
            if wait is None or _status(e) not in _RETRYABLE:
                raise
            log.warning("Gemini %s, retry %d in %ss", _status(e), attempt + 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")
