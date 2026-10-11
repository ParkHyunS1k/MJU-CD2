import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.services.rag_taxonomy import CONDITION_KEYS, Category


class SourcesError(Exception):
    pass


class SectionRule(BaseModel):
    match: str = Field(min_length=1)
    category: Category


class SectionSpec(BaseModel):
    """책갈피가 없는 PDF에서 넣을 장. pages는 PDF 쪽 번호(1부터, 양끝 포함)."""

    title: str = Field(min_length=1)
    pages: tuple[int, int]
    category: Category | None = None

    @field_validator("pages")
    @classmethod
    def _valid_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        start, end = value
        if start < 1 or end < start:
            raise ValueError(f"invalid page range: {value}")
        return value


class SourceDoc(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    file: str
    title: str = Field(min_length=1)
    publisher: str | None = None
    year: int | None = None
    url: str | None = None
    license_note: str | None = None
    conditions: list[str] = Field(min_length=1)
    category: Category
    section_rules: list[SectionRule] = []
    sections: list[SectionSpec] = []

    @field_validator("conditions")
    @classmethod
    def _known_conditions(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - CONDITION_KEYS)
        if unknown:
            raise ValueError(f"unknown condition keys: {unknown}")
        return value

    @model_validator(mode="after")
    def _sections_consistent(self) -> "SourceDoc":
        titles = [s.title for s in self.sections]
        if len(titles) != len(set(titles)):
            raise ValueError("duplicate section titles")
        ranges = sorted(s.pages for s in self.sections)
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
            if next_start <= prev_end:
                raise ValueError(f"overlapping section pages: {ranges}")
        return self


def load_sources(sources_path: Path, raw_dir: Path) -> list[tuple[SourceDoc, Path]]:
    try:
        raw = json.loads(sources_path.read_text(encoding="utf-8"))
        docs = [SourceDoc.model_validate(entry) for entry in raw]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as e:
        raise SourcesError(f"{sources_path}: {e}") from e

    ids = [doc.id for doc in docs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SourcesError(f"duplicate source ids: {duplicates}")

    resolved = []
    for doc in docs:
        path = raw_dir / doc.file
        if not path.is_file():
            raise SourcesError(f"[{doc.id}] file not found: {path}")
        resolved.append((doc, path))
    return resolved
