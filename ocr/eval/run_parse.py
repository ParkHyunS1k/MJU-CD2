"""테스트셋 OCR → 파서 → 예측 JSONL.

OCR 입력은 두 가지:
  --ocr-dir  실기기(ML Kit)에서 내보낸 OCR JSON 폴더 ({id}.json, 형식은 tests/vectors의 "lines")
             → 최종 평가는 이것으로 한다.
  --images   이미지 폴더를 Mac Apple Vision으로 OCR (개발용 근사치). OCR 결과는 --ocr-dir에 캐시.

사용법:
  uv run python -m eval.run_parse --labels <drive>/labels.jsonl --ocr-dir <drive>/ocr_mlkit --out eval/out/preds_B.jsonl
  uv run python -m eval.run_parse --labels <drive>/labels.jsonl --images <drive>/images \
      --ocr-dir eval/out/ocr_vision --out eval/out/preds_B_vision.jsonl
"""

import argparse
import json
from pathlib import Path

from inbody_ocr.ocr_types import OcrLine
from inbody_ocr.parser import PARSER_VERSION, parse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--ocr-dir", required=True, type=Path)
    ap.add_argument("--images", type=Path, help="지정하면 OCR JSON이 없는 항목을 Mac Vision으로 생성")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.ocr_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]

    with args.out.open("w") as f:
        for row in labels:
            ocr_path = args.ocr_dir / f"{row['id']}.json"
            rec = {"id": row["id"], "parser_version": PARSER_VERSION}
            try:
                if not ocr_path.exists():
                    if args.images is None:
                        raise FileNotFoundError(ocr_path)
                    from inbody_ocr.ocr_mac import recognize

                    lines = recognize(str(args.images / row["image"]))
                    ocr_path.write_text(json.dumps({"lines": [l.to_json() for l in lines]}, ensure_ascii=False))
                lines = [OcrLine.from_json(l) for l in json.loads(ocr_path.read_text())["lines"]]
                rec["extraction"] = parse(lines).extraction.model_dump()
            except Exception as e:  # 실패도 기록 (채점에서 전 필드 오답 처리)
                rec["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(row["id"], "ERROR " + rec["error"] if "error" in rec else "ok")


if __name__ == "__main__":
    main()
