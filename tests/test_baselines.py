from __future__ import annotations

import numpy as np
import pandas as pd

from baloto_ml.config import P_BALL, P_SUPER
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.models.base import GameData
from baloto_ml.models.baselines import ConstantBaseline, FrequencyBaseline


def game(history: pd.DataFrame, juego: str = "baloto") -> GameData:
    return GameData.from_draws(history, juego)


def test_constant_baseline(history: pd.DataFrame) -> None:
    p = ConstantBaseline().predict(game(history), np.arange(10, 20))
    assert p.balls.shape == (10, 43) and p.sb.shape == (10, 16)
    np.testing.assert_allclose(p.balls, P_BALL)
    np.testing.assert_allclose(p.sb, P_SUPER)
    np.testing.assert_allclose(p.balls.sum(axis=1), 5)


def test_frequency_baseline_matches_manual_computation(history: pd.DataFrame) -> None:
    data = game(history)
    t, a = 37, 10.0
    p = FrequencyBaseline(pseudo_draws=a).predict(data, np.array([0, t]))
    np.testing.assert_allclose(p.balls[0], P_BALL)  # sin historia: el uniforme
    expected = (data.y_balls[:t].sum(axis=0) + a * P_BALL) / (t + a)
    np.testing.assert_allclose(p.balls[1], expected)
    np.testing.assert_allclose(p.balls.sum(axis=1), 5)
    np.testing.assert_allclose(p.sb.sum(axis=1), 1)


def test_frequency_baseline_ignores_the_future(rng: np.random.Generator) -> None:
    """Cambiar los sorteos desde t en adelante no altera la predicción de t."""
    h1 = synthetic_history(80, rng, juegos=("baloto",))
    h2 = h1.copy()
    future = synthetic_history(80, np.random.default_rng(999), juegos=("baloto",))
    cols = ["b1", "b2", "b3", "b4", "b5", "superbalota"]
    h2.loc[50:, cols] = future.loc[50:, cols].to_numpy()
    idx = np.arange(0, 51)  # hasta t = 50 inclusive
    p1 = FrequencyBaseline().predict(game(h1), idx)
    p2 = FrequencyBaseline().predict(game(h2), idx)
    np.testing.assert_array_equal(p1.balls, p2.balls)
    np.testing.assert_array_equal(p1.sb, p2.sb)
    assert not np.array_equal(game(h1).y_balls[50:], game(h2).y_balls[50:])
