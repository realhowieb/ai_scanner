"""Rate quotas survive key pressure and expire using their own windows."""
import pytest
from fastapi import HTTPException

from api import ratelimit


@pytest.fixture(autouse=True)
def clean_limits():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_key_pressure_preserves_long_window_quota(monkeypatch):
    monkeypatch.setattr(ratelimit, "_MAX_KEYS", 1)
    for _ in range(5):
        ratelimit.check("signup", "existing", now=0)
    with pytest.raises(HTTPException) as pressure:
        ratelimit.check("login", "new", now=601)
    assert pressure.value.status_code == 429
    assert len(ratelimit._hits) == 1
    with pytest.raises(HTTPException):
        ratelimit.check("signup", "existing", now=601)


def test_expired_keys_free_capacity_at_window_boundary(monkeypatch):
    monkeypatch.setattr(ratelimit, "_MAX_KEYS", 1)
    ratelimit.check("login", "old", now=0)
    ratelimit.check("signup", "new", now=600)
    assert set(ratelimit._hits) == {("signup", "new")}


def test_existing_key_keeps_working_when_capacity_is_full(monkeypatch):
    monkeypatch.setattr(ratelimit, "_MAX_KEYS", 1)
    ratelimit.check("login", "existing", now=0)
    ratelimit.check("login", "existing", now=1)
    with pytest.raises(HTTPException):
        ratelimit.check("login", "new", now=1)
    assert len(ratelimit._hits) == 1


def test_retry_after_rounds_up_until_request_can_succeed():
    for _ in range(5):
        ratelimit.check("signup", "user", now=0)
    with pytest.raises(HTTPException) as caught:
        ratelimit.check("signup", "user", now=3598.5)
    assert caught.value.headers["Retry-After"] == "2"
    ratelimit.check("signup", "user", now=3600.5)
