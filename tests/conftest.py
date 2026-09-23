from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from valchamps.config import Settings
from valchamps.data import db

FIXTURES = Path(__file__).parent / "fixtures" / "vlr"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_url=f"sqlite:///{tmp_path / 'test.db'}",
        cache_dir=tmp_path / "cache",
        base_url="https://vlr.test",
        request_interval=0.0,
        user_agent="valchamps-tests",
    )


@pytest.fixture
def engine(settings: Settings) -> Engine:
    eng = db.get_engine(settings.db_url)
    db.init_db(eng)
    return eng
