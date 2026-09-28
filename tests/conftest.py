from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from baloto_ml.config import Paths
from baloto_ml.data.synthetic import synthetic_history


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def history(rng: np.random.Generator) -> pd.DataFrame:
    """60 sorteos uniformes por juego desde el sáb 2025-05-03 (cruza el inicio de los lunes)."""
    return synthetic_history(60, rng, start=date(2025, 5, 3), start_n=1000)


@pytest.fixture
def paths(tmp_path) -> Paths:
    return Paths(tmp_path)
