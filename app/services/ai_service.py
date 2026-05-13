import json
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.schemas.inbody import AIReport, InBodyExtraction

MODEL = "gemini-2.5-flash"


@lru_cache
def get_gemini() -> genai.Client:
    return genai.Client(api_key=get_settings().GEMINI_API_KEY)


SYSTEM_INSTRUCTION = (
    "당신은 공인 운동 전문가이자 영양 코치입니다. "
    "사용자의 인바디 데이터를 분석해 객관적인 평가와 실천 가능한 조언을 한국어로 제공합니다. "
    "의학적 진단은 하지 말고, 우려가 있으면 cautions 필드에 전문의 상담을 권유하세요. "
    "반드시 지정된 JSON 스키마로만 응답하세요."
)


def analyze_inbody(
    current: dict,
    previous: dict | None,
    user_profile: dict,
) -> dict:
    """현재 + 직전 인바디 측정과 사용자 프로필을 받아 구조화된 분석을 생성한다."""
    client = get_gemini()

    payload = {
        "user_profile": user_profile,
        "current_measurement": current,
        "previous_measurement": previous,
    }
    prompt = (
        "다음 데이터를 분석하세요. previous_measurement 가 null 이면 첫 측정입니다.\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AIReport,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return json.loads(res.text)


EXTRACT_SYSTEM_INSTRUCTION = (
    "당신은 인바디(InBody) 측정 결과지를 읽고 수치를 추출하는 OCR 어시스턴트입니다. "
    "주어진 이미지/PDF가 인바디 결과지인지 먼저 판단하고, 결과지가 맞다면 각 항목의 수치를 정확히 읽어 지정된 JSON 스키마로 반환합니다. "
    "값이 명확하지 않거나 누락된 항목은 null로 둡니다. "
    "confidence는 전체적 가독성과 추출 확신도를 0~1로 표현합니다. "
    "결과지가 아니면 is_inbody=false로 두고 다른 수치는 모두 null로 두세요."
)


def extract_inbody_from_image(file_bytes: bytes, mime_type: str) -> dict:
    """이미지/PDF에서 인바디 수치를 추출. InBodyExtraction 스키마 dict를 반환."""
    client = get_gemini()

    res = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            "이 파일에서 인바디 측정 수치를 추출해 JSON 스키마로 반환하세요.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=EXTRACT_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=InBodyExtraction,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return json.loads(res.text)
