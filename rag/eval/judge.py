"""쿼리 하나의 후보 전체를 Gemini로 한 번에 판정한다(무료 한도 안에서 쿼리당 1회 호출)."""

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from rag.eval.metrics import compact


class CandidateJudgment(BaseModel):
    candidate_id: str
    relevant: bool
    evidence: list[str]


class QueryJudgment(BaseModel):
    judgments: list[CandidateJudgment]


def judge_prompt(query: str, candidates: list[dict]) -> str:
    blocks = "\n\n".join(f"[c{i}]\n{c['content']}" for i, c in enumerate(candidates, 1))
    return (
        "당신은 의학·영양 문서 검색 평가자입니다. 아래 검색 쿼리에 대해 각 후보 문단이 "
        "직접 답이 되는 내용을 담고 있는지 판정하세요.\n"
        "- relevant: 쿼리에 대한 답(기준, 권고, 방법, 수치)이 문단에 있으면 true. 주제만 비슷하면 false.\n"
        "- evidence: relevant가 true일 때 답이 되는 문장 1~3개를 문단에서 한 글자도 바꾸지 말고 그대로 복사. "
        "false면 빈 배열.\n"
        "- 모든 후보에 대해 candidate_id(c1, c2, ...)로 하나씩 답하세요.\n\n"
        f"검색 쿼리: {query}\n\n{blocks}"
    )


def judge_query(ask: Callable[[str], QueryJudgment], query: str, candidates: list[dict]) -> dict[str, dict]:
    by_id = {f"c{i}": c for i, c in enumerate(candidates, 1)}
    result = {c["content_hash"]: {"relevant": None, "evidence": []} for c in candidates}
    for judgment in ask(judge_prompt(query, candidates)).judgments:
        candidate = by_id.get(judgment.candidate_id)
        if candidate is None:
            continue
        content = compact(candidate["content"])
        evidence = [e for e in judgment.evidence if len(compact(e)) >= 5 and compact(e) in content]
        result[candidate["content_hash"]] = {
            "relevant": judgment.relevant,
            "evidence": evidence if judgment.relevant else [],
        }
    return result


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self._rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._rows[row["query_id"]] = row["result"]

    def get(self, query_id: str) -> dict | None:
        return self._rows.get(query_id)

    def put(self, query_id: str, result: dict) -> None:
        self._rows[query_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"query_id": query_id, "result": result}, ensure_ascii=False) + "\n")


def gemini_ask() -> Callable[[str], QueryJudgment]:
    from rag.eval.llm import generate_json, get_client

    client = get_client()
    return lambda prompt: generate_json(client, prompt, QueryJudgment)


_CODEX_SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object",
        "properties": {"candidate_id": {"type": "string"}, "relevant": {"type": "boolean"},
                       "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["candidate_id", "relevant", "evidence"],
        "additionalProperties": False,
    }}},
    "required": ["judgments"],
    "additionalProperties": False,
}


def codex_ask(run=subprocess.run, model: type[BaseModel] = QueryJudgment,
              schema: dict = _CODEX_SCHEMA) -> Callable[[str], BaseModel]:
    """ChatGPT 로그인된 Codex CLI로 판정한다(Gemini 무료 한도 대안). 읽기 전용 샌드박스, 세션 저장 안 함.
    model·schema를 바꾸면 다른 판정(예: 리포트 근거 평가)에도 쓴다."""

    def ask(prompt: str) -> BaseModel:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path, out = Path(tmp) / "schema.json", Path(tmp) / "out.json"
            schema_path.write_text(json.dumps(schema))
            cmd = ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                   "-C", tmp, "--output-schema", str(schema_path), "-o", str(out), "-"]
            try:
                run(cmd, input=prompt, text=True, capture_output=True, timeout=900, check=True)
            except subprocess.CalledProcessError as e:
                # 사용 한도 등 원인은 Codex의 ERROR 줄에만 있다
                errors = [line for line in f"{e.stdout or ''}\n{e.stderr or ''}".splitlines() if "ERROR" in line]
                raise RuntimeError(f"codex exec failed: {' | '.join(errors) or (e.stderr or '')[-500:]}") from e
            return model.model_validate_json(out.read_text())

    return ask
