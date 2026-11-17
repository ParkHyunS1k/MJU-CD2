from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.routers.users as users
from app.schemas.user import UserResponse, UserUpdate

CURRENT = SimpleNamespace(id="u1")


class FakeTable:
    """supabase 쿼리 빌더 흉내. update()에 들어온 값을 기록한다."""

    def __init__(self, row):
        self.row = row
        self.updated = None

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def limit(self, *_):
        return self

    def update(self, data):
        self.updated = data
        self.row = {**self.row, **data}
        return self

    def execute(self):
        return SimpleNamespace(data=[self.row])


@pytest.fixture
def table(monkeypatch):
    t = FakeTable({"id": "u1", "created_at": "2026-10-12T00:00:00+00:00", "nickname": "a",
                   "gender": "male", "birth_year": 1990, "fitness_goal": None})
    monkeypatch.setattr(users, "get_supabase_admin", lambda: SimpleNamespace(table=lambda _: t))
    t.refreshed = []
    monkeypatch.setattr(users, "refresh_user", lambda sb, user_id, profile: t.refreshed.append((user_id, profile)))
    return t


def test_patch_saves_explicit_null_family_history(table):
    """'예'에서 '모름'(null)으로 바꿀 수 있어야 한다."""
    users.update_me(UserUpdate(fh_diabetes=None, fh_hypertension=True), CURRENT)
    assert table.updated == {"fh_diabetes": None, "fh_hypertension": True}


def test_patch_does_not_send_unset_fields(table):
    users.update_me(UserUpdate(nickname="b"), CURRENT)
    assert table.updated == {"nickname": "b"}


def test_patch_empty_body_rejected(table):
    with pytest.raises(HTTPException) as e:
        users.update_me(UserUpdate(), CURRENT)
    assert e.value.status_code == 400 and table.updated is None


def test_response_defaults_family_history_to_none(table):
    r = UserResponse.model_validate(users.get_me(CURRENT))
    assert (r.fh_hypertension, r.fh_diabetes, r.fh_dyslipidemia) == (None, None, None)


def test_patch_refreshes_risks_only_for_risk_fields(table):
    users.update_me(UserUpdate(nickname="b"), CURRENT)
    assert table.refreshed == []
    users.update_me(UserUpdate(fh_diabetes=True), CURRENT)
    assert table.refreshed == [("u1", table.row)]
