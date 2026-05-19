import json
from functools import lru_cache

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.schemas.inbody import InBodyExtraction

MODEL = "gemini-2.5-flash"


@lru_cache
def get_gemini() -> genai.Client:
    return genai.Client(api_key=get_settings().GEMINI_API_KEY)


_REPORT_SYSTEM_INSTRUCTION = (
    "당신은 공인 운동 전문가이자 영양 코치입니다. "
    "ML 모델이 예측한 성인병 위험도와 SHAP 주요 기여 요인을 바탕으로 "
    "생활 패턴 개선 방향과 영양제를 한국어로 추천하세요. "
    "trend_analysis가 제공된 경우 수치 변화 추세를 반영해 더 구체적인 조언을 하세요. "
    "특히 SHAP 주요 요인(is_shap_important=true)이 악화 방향으로 변하고 있다면 강조하세요. "
    "의학적 진단은 하지 말고, 우려가 있으면 cautions 필드에 전문의 상담을 권유하세요. "
    "반드시 지정된 JSON 스키마로만 응답하세요."
)


class _GeminiReportSchema(BaseModel):
    lifestyle_tips: list[str]
    supplement_tips: list[str]
    cautions: str | None


def _parse_json_response(text: str | None, schema: type[BaseModel]) -> dict:
    if not text:
        raise ValueError("Empty AI response")
    try:
        return schema.model_validate_json(text).model_dump(mode="json")
    except ValidationError as e:
        raise ValueError(f"Invalid AI response schema: {e}") from e


def generate_report(
    disease_risks: list[dict],
    user_profile: dict,
    trend_analysis: dict | None = None,
) -> dict:
    """ML 예측 결과와 유저 프로필을 받아 생활습관·영양제 추천을 생성한다.
    trend_analysis가 있으면 시계열 추세도 반영한다.
    """
    client = get_gemini()

    payload = {
        "user_profile": {k: user_profile.get(k) for k in ("gender", "birth_year", "fitness_goal")},
        "disease_risks": disease_risks,
        "trend_analysis": trend_analysis,
    }
    prompt = (
        "다음 성인병 위험도 분석 결과를 바탕으로 생활 패턴 개선 방향과 영양제를 추천하세요. "
        "trend_analysis가 null이 아니면 수치 변화 추세도 함께 반영하세요.\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_REPORT_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiReportSchema,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return _parse_json_response(res.text, _GeminiReportSchema)


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
    return _parse_json_response(res.text, InBodyExtraction)
