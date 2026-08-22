from __future__ import annotations

import os

import pytest


os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""

from app.security import rate_limiter


@pytest.fixture(autouse=True)
def reset_security_state():
    rate_limiter.reset()
    yield
    rate_limiter.reset()
