from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture
def data_dir():
    return DATA
