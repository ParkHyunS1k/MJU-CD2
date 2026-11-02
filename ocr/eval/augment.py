"""캡처 결과지 → 폰 촬영을 흉내 낸 변형 이미지 (강건성 점검용, 결정적).

기울기·원근 왜곡·흐림·그림자·저해상도·JPEG 압축과 그 조합. 실제 사진 평가를 대신하지는 않는다.

사용법 (numpy 필요):
  uv run --with numpy python -m eval.augment --labels data/labels_capture.jsonl \
      --images <캡처 폴더> --out eval/out/aug   # 이미지 + labels_aug.jsonl 생성
"""

import argparse
import io
import json
import random
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BG = (90, 90, 90)  # 책상 등 종이 바깥


def _pad(img: Image.Image, frac: float = 0.06) -> Image.Image:
    w, h = img.size
    p = int(w * frac)
    out = Image.new("RGB", (w + 2 * p, h + 2 * p), BG)
    out.paste(img, (p, p))
    return out


def _rot(img: Image.Image, deg: float) -> Image.Image:
    return img.rotate(deg, Image.BICUBIC, expand=True, fillcolor=BG)


def _perspective(img: Image.Image, k: float, seed: int) -> Image.Image:
    """위쪽이 좁아지는 사다리꼴 + 약간의 무작위 (비스듬히 내려다보고 찍은 경우)."""
    r = random.Random(seed)
    w, h = img.size
    d = lambda: r.uniform(0, k) * w  # noqa: E731
    src = [(d(), d() * 0.3), (w - d(), d() * 0.3), (w - d() * 0.2, h), (d() * 0.2, h)]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    a, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        a += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    coef = np.linalg.solve(np.array(a, float), np.array(b, float))
    return img.transform((w, h), Image.PERSPECTIVE, tuple(coef), Image.BICUBIC, fillcolor=BG)


def _shadow(img: Image.Image, strength: float) -> Image.Image:
    w, h = img.size
    grad = Image.linear_gradient("L").resize((w, h)).rotate(35, fillcolor=128)
    grad = grad.point(lambda v: 255 - int(v * strength))
    return Image.composite(img, Image.new("RGB", (w, h)), grad)


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _scale(img: Image.Image, width: int) -> Image.Image:
    w, h = img.size
    return img.resize((width, int(h * width / w)), Image.LANCZOS)


def _photo(img: Image.Image, seed: int) -> Image.Image:
    """원근 + 회전(3~7°) + 그림자 + 흐림 + 저해상도 + JPEG를 무작위로 섞은 어려운 사진."""
    r = random.Random(seed)
    img = _perspective(_pad(img, 0.1), r.uniform(0.04, 0.08), seed)
    img = _rot(img, r.choice([-1, 1]) * r.uniform(3, 7))
    img = _shadow(img, r.uniform(0.3, 0.7)).filter(ImageFilter.GaussianBlur(r.uniform(1.0, 1.8)))
    return _jpeg(_scale(img, r.choice([1000, 1200, 1500])), r.randint(35, 70))


VARIANTS = {
    "rot-3": lambda im, s: _rot(_pad(im), -3),
    "rot+5": lambda im, s: _rot(_pad(im), 5),
    "rot-8": lambda im, s: _rot(_pad(im), -8),
    "persp_strong": lambda im, s: _perspective(_pad(im), 0.10, s),
    "blur2": lambda im, s: im.filter(ImageFilter.GaussianBlur(2)),
    "small900": lambda im, s: _scale(im, 900),
    "jpeg30": lambda im, s: _jpeg(im, 30),
    "shadow": lambda im, s: _shadow(im, 0.6),
    "photo_a": _photo,
    "photo_b": lambda im, s: _photo(im, s + 1000),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in (json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()):
        img = Image.open(args.images / row["image"]).convert("RGB")
        for name, fn in VARIANTS.items():
            vid = f"{row['id']}__{name}"
            dst = args.out / f"{vid}.jpg"
            if not dst.exists():
                fn(img, zlib.crc32(vid.encode())).save(dst, quality=92)
            rows.append({**row, "id": vid, "image": dst.name, "input_type": name})
    (args.out / "labels_aug.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "images")


if __name__ == "__main__":
    main()
