from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    nickname: str | None = Field(None, max_length=30)
    gender: Literal["male", "female"] | None = None
    birth_year: int | None = Field(None, ge=1900, le=2100)
    fitness_goal: str | None = None


class UserUpdate(UserBase):
    pass


class UserResponse(UserBase):
    id: str
    created_at: datetime
