"""공용 테스트 벡터. Dart/TS 포팅본도 같은 파일로 검증한다.

tests/vectors/        합성·가공 데이터 (Git 포함)
tests/vectors_local/  실물 결과지 OCR (개인 건강정보 → Git 제외, 있으면 함께 실행)
"""

import json
from pathlib import Path

import pytest

from inbody_ocr.ocr_types import OcrLine
from inbody_ocr.parser import parse

HERE = Path(__file__).parent
VECTORS = sorted(HERE.glob("vectors/*.json")) + sorted(HERE.glob("vectors_local/*.json"))


@pytest.mark.parametrize("path", VECTORS, ids=[p.stem for p in VECTORS])
def test_vector(path: Path):
    v = json.loads(path.read_text())
    got = parse([OcrLine.from_json(l) for l in v["lines"]]).extraction.model_dump(mode="json")
    wrong = {k: (got[k], want) for k, want in v["expected"].items() if got[k] != want}
    assert not wrong, f"(got, want): {wrong}"
