"""OCR 엔진 공통 입력 형식.

파서는 이 형식만 받는다. 앱에서는 ML Kit TextLine을, 개발용 Mac에서는 Apple Vision 결과를
이 형식으로 변환해 넘긴다. JSON 직렬화 형태(tests/vectors)도 동일하게 유지해
Dart/TS 포팅 시 같은 테스트 벡터로 검증한다.

좌표: 이미지 크기로 나눈 0~1 값, 원점은 좌상단 (ML Kit 픽셀 좌표 / 이미지 크기).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @classmethod
    def from_json(cls, d: dict) -> "OcrLine":
        x0, y0, x1, y1 = d["box"]
        return cls(d["text"], x0, y0, x1, y1)

    def to_json(self) -> dict:
        return {"text": self.text, "box": [round(v, 4) for v in (self.x0, self.y0, self.x1, self.y1)]}
