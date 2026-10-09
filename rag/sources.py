import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.services.rag_taxonomy import CONDITION_KEYS, Category


class SourcesError(Exception):
    pass


class SectionRule(BaseModel):
    match: str = Field(min_length=1)
    category: Category


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

    @field_validator("conditions")
    @classmethod
    def _known_conditions(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - CONDITION_KEYS)
        if unknown:
            raise ValueError(f"unknown condition keys: {unknown}")
        return value


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
