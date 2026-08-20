from pathlib import Path

import pytest


@pytest.fixture
def catalog_root() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"
