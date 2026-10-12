"""FastAPI 엔드포인트. 앱이 온디바이스에서 추출·확인한 '숫자'만 받는다 (이미지는 받지 않음).

앱에서 검증을 이미 거쳤더라도 서버에서 같은 검증을 다시 돌린다 (앱 버그·변조 방어).
"""

from dataclasses import asdict
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .schema import FIELD_SPECS, InBodyExtraction
from .validate import validate

app = FastAPI(title="InBody measurements")


class MeasurementIn(BaseModel):
    input_type: Literal["photo", "capture", "manual"]
    sex: Optional[Literal["M", "F"]] = None
    measured_at: Optional[str] = None
    values: dict[str, Optional[float]]
    # 사용자가 확인 화면에서 직접 고친 필드. OCR 정확도·수정률 집계에 사용
    edited_fields: list[str] = []
    parser_version: Optional[str] = None


@app.post("/v1/inbody/measurements")
def submit_measurement(body: MeasurementIn):
    unknown = set(body.values) - set(FIELD_SPECS)
    if unknown:
        raise HTTPException(422, f"알 수 없는 필드: {sorted(unknown)}")

    ext = InBodyExtraction(is_inbody_sheet=True, sex=body.sex, measured_at=body.measured_at, **body.values)
    report = validate(ext)
    # 사용자가 확인했더라도 범위 밖·불일치 값은 위험도 예측에 넣지 않도록 표시만 하고 저장 여부는 백엔드가 결정
    return {
        "fields": {k: asdict(v) for k, v in report.fields.items()},
        "checks": [asdict(c) for c in report.checks],
        "needs_review": report.needs_review,
        "missing": report.missing,
        "edited_fields": body.edited_fields,
        "parser_version": body.parser_version,
    }
