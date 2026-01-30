from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memory_condense.db import Database


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def db(tmp_dir: Path) -> Database:
    d = Database(tmp_dir / "test.db")
    yield d
    d.close()
