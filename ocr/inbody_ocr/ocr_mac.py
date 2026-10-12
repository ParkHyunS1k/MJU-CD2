"""개발용: Mac의 Apple Vision으로 OCR (iOS 온디바이스 Vision과 같은 계열 엔진).

앱의 실제 엔진(ML Kit)과 결과가 다를 수 있으므로 파서 개발·디버깅용으로만 쓰고,
최종 평가는 실기기에서 내보낸 OCR JSON으로 한다.
"""

from PIL import Image, ImageOps

from .ocr_types import OcrLine


def recognize(path: str) -> list[OcrLine]:
    from ocrmac import ocrmac  # macOS 전용 dev 의존성

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    results = ocrmac.OCR(img, recognition_level="accurate", language_preference=["ko-KR", "en-US"]).recognize()
    lines = []
    for text, _conf, (x, y, w, h) in results:
        # Vision 좌표: 0~1, 원점 좌하단 → 좌상단 기준으로 변환
        lines.append(OcrLine(text, x, 1 - (y + h), x + w, 1 - y))
    return lines
