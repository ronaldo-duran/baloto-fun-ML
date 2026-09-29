from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baloto_ml.config import EvalConfig
from baloto_ml.evaluation.walk_forward import walk_forward
from baloto_ml.models.base import Forecaster, GameData, Probabilities
from baloto_ml.models.baselines import ConstantBaseline


class SpyForecaster(Forecaster):
    """Registra con qué sorteos se entrena y cuáles predice."""

    log: list[tuple[np.ndarray, np.ndarray]] = []

    def fit(self, data: GameData, train_idx: np.ndarray) -> Forecaster:
        self.train_idx = train_idx
        return self

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        SpyForecaster.log.append((self.train_idx, idx))
        return ConstantBaseline().predict(data, idx)


def test_walk_forward_never_trains_on_the_future(history: pd.DataFrame) -> None:
    data = GameData.from_draws(history, "baloto")  # 60 sorteos
    cfg = EvalConfig(warmup=10, min_train=15, refit_every=7)
    SpyForecaster.log = []
    result = walk_forward(SpyForecaster, data, cfg)

    assert result.test_idx.tolist() == list(range(25, 60))
    assert result.n_refits == len(SpyForecaster.log) == 5  # ceil(35 / 7)
    predicted = np.concatenate([idx for _, idx in SpyForecaster.log])
    assert predicted.tolist() == list(range(25, 60))  # cada sorteo de prueba, una sola vez
    for train_idx, idx in SpyForecaster.log:
        assert train_idx.min() == cfg.warmup
        assert train_idx.max() < idx.min()  # solo el pasado
        assert train_idx.max() == idx.min() - 1  # ventana creciente: todo el pasado disponible
    assert result.probs.balls.shape == (35, 43)


def test_walk_forward_requires_enough_history(history: pd.DataFrame) -> None:
    data = GameData.from_draws(history, "baloto")
    with pytest.raises(ValueError, match="Se necesitan"):
        walk_forward(ConstantBaseline, data, EvalConfig(warmup=40, min_train=30))
