"""개발용: Mac의 Apple Vision으로 OCR (iOS 온디바이스 Vision과 같은 계열 엔진).

앱의 실제 엔진(ML Kit)과 결과가 다를 수 있으므로 파서 개발·디버깅용으로만 쓰고,
최종 평가는 실기기에서 내보낸 OCR JSON으로 한다.

페이지 전체를 한 번에 읽으면 작은 숫자(운동소비열량 오른쪽 열, 신체변화 그래프)를 놓치거나
소수점을 빠뜨린다. 그래서 겹치는 조각(타일)으로 나눠 다시 읽고 합친다.
"""

from difflib import SequenceMatcher

from PIL import Image, ImageOps

from .ocr_types import OcrLine

TILE_OVERLAP = 0.08  # 이웃 조각과 겹치는 폭 (페이지 비율)


def _ocr(img: Image.Image, x0: float = 0, y0: float = 0, sx: float = 1, sy: float = 1) -> list[OcrLine]:
    """img를 읽어 페이지 좌표(0~1, 좌상단 원점)로 변환. (x0, y0, sx, sy): 조각의 페이지 내 위치·크기."""
    from ocrmac import ocrmac  # macOS 전용 dev 의존성

    results = ocrmac.OCR(img, recognition_level="accurate", language_preference=["ko-KR", "en-US"]).recognize()
    # Vision 좌표: 0~1, 원점 좌하단 → 좌상단 기준으로 변환
    return [
        OcrLine(text, x0 + x * sx, y0 + (1 - (y + h)) * sy, x0 + (x + w) * sx, y0 + (1 - y) * sy)
        for text, _conf, (x, y, w, h) in results
    ]


def _inside(p: OcrLine, q: OcrLine) -> bool:
    """p의 중심이 q 박스 안에 있는가."""
    cx = (p.x0 + p.x1) / 2
    return q.x0 <= cx <= q.x1 and q.y0 <= p.cy <= q.y1


def _same(p: OcrLine, q: OcrLine) -> bool:
    """같은 줄을 두 번 읽은 것인가: 위치가 겹치고 글자도 비슷해야 한다.
    ('BMI'와 그 아래 'Body Mass Index'를 한 줄로 합쳐 읽은 경우 위치만 보면 'BMI'가 지워진다)"""
    if not (_inside(p, q) or _inside(q, p)):
        return False
    a, b = "".join(p.text.split()), "".join(q.text.split())
    # 포함 관계는 길이가 비슷할 때만 (')' 한 글자가 '13.4 ( 11.1~13.5 )' 줄을 지우지 않게)
    contained = (a in b or b in a) and min(len(a), len(b)) >= 0.5 * max(len(a), len(b))
    return contained or SequenceMatcher(None, a, b).ratio() >= 0.6


def recognize(path, tiles: tuple[int, int] | None = (2, 3)) -> list[OcrLine]:
    """path: 파일 경로 또는 파일 객체. tiles: (가로, 세로) 조각 수. None이면 전체 페이지만 읽는다."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    full = _ocr(img)
    if not tiles:
        return full

    W, H = img.size
    nx, ny = tiles
    kept: list[OcrLine] = []
    for i in range(nx):
        for j in range(ny):
            fx0, fx1 = max(0, i / nx - TILE_OVERLAP / 2), min(1, (i + 1) / nx + TILE_OVERLAP / 2)
            fy0, fy1 = max(0, j / ny - TILE_OVERLAP / 2), min(1, (j + 1) / ny + TILE_OVERLAP / 2)
            crop = img.crop((round(fx0 * W), round(fy0 * H), round(fx1 * W), round(fy1 * H)))
            for l in _ocr(crop, fx0, fy0, fx1 - fx0, fy1 - fy0):
                # 페이지 안쪽 경계에 닿은 줄은 잘렸을 수 있다. 이웃 조각이나 전체 페이지 결과에 맡긴다
                edge = 0.004
                if (fx0 > 0 and l.x0 - fx0 < edge) or (fx1 < 1 and fx1 - l.x1 < edge) or \
                   (fy0 > 0 and l.y0 - fy0 < edge) or (fy1 < 1 and fy1 - l.y1 < edge):
                    continue
                dup = next((k for k in kept if _same(l, k)), None)
                if dup is None:
                    kept.append(l)
                elif len(l.text) > len(dup.text):  # 겹치는 조각에서 같은 줄을 두 번 읽음: 더 긴 쪽을 남긴다
                    kept[kept.index(dup)] = l
    # 조각들이 놓친 자리(조각 경계를 가로지르는 긴 줄 등)는 전체 페이지 결과로 채운다
    return kept + [l for l in full if not any(_same(l, k) for k in kept)]
