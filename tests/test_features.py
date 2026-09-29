"""Las features de un sorteo solo pueden usar sorteos anteriores (sin fuga de información)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from baloto_ml.data.encoding import multi_hot
from baloto_ml.data.synthetic import random_draws, synthetic_history
from baloto_ml.features.build import (
    GAP_CAP,
    PER_NUMBER_NAMES,
    SHARED_NAMES,
    build_features,
    draws_since_last,
)
from baloto_ml.models.base import GameData

FREQ_10 = PER_NUMBER_NAMES.index("freq_10")
FREQ_100 = PER_NUMBER_NAMES.index("freq_100")
GAP = PER_NUMBER_NAMES.index("sorteos_desde_ultima")
TREND = PER_NUMBER_NAMES.index("tendencia_15v15")


@pytest.fixture
def data(rng: np.random.Generator) -> GameData:
    return GameData.from_draws(synthetic_history(160, rng, juegos=("baloto",)), "baloto")


def test_shapes_include_next_draw(data: GameData) -> None:
    fs = data.features
    n = len(data)
    assert fs.balls.per_number.shape == (n + 1, 43, len(PER_NUMBER_NAMES))
    assert fs.sb.per_number.shape == (n + 1, 16, len(PER_NUMBER_NAMES))
    assert fs.balls.shared.shape == (n + 1, len(SHARED_NAMES))
    assert (fs.balls.shared.sum(axis=1) == 1).all()  # un día de sorteo por fila
    assert fs.balls.shared[n].argmax() == [0, 2, 5].index(data.next_date.weekday())


def test_features_of_t_are_computable_from_the_past_only(data: GameData) -> None:
    """Las features de t calculadas con el historial completo son idénticas a las que se
    obtienen conociendo SOLO los sorteos 0..t-1 (como en producción)."""
    full = data.features
    for t in range(1, len(data) + 1):
        past = data.head(t).features  # su última fila (t) es "el sorteo siguiente"
        np.testing.assert_array_equal(past.balls.per_number[t], full.balls.per_number[t])
        np.testing.assert_array_equal(past.sb.per_number[t], full.sb.per_number[t])
        np.testing.assert_array_equal(past.balls.shared[t], full.balls.shared[t])


def test_changing_the_future_does_not_change_past_features(data: GameData) -> None:
    t = 100
    balls, sb = random_draws(len(data) - t, np.random.default_rng(99))
    y_future = data.y_balls.copy()
    y_future[t:] = multi_hot(balls)
    altered = replace(data, y_balls=y_future)
    assert not np.array_equal(altered.y_balls[t:], data.y_balls[t:])
    np.testing.assert_array_equal(
        altered.features.balls.per_number[: t + 1], data.features.balls.per_number[: t + 1]
    )
    assert not np.array_equal(
        altered.features.balls.per_number[t + 1 :], data.features.balls.per_number[t + 1 :]
    )


def test_current_draw_is_not_in_its_own_features(data: GameData) -> None:
    """El resultado del sorteo t no aparece en las features de t."""
    t, j = 120, 7
    y = data.y_balls.copy()
    y[t] = 0
    y[t, [j, 10, 20, 30, 40]] = 1
    flipped = y.copy()
    flipped[t] = 0
    flipped[t, [0, 1, 2, 3, 4]] = 1
    a = build_features(y, data.y_sb, data.weekday)
    b = build_features(flipped, data.y_sb, data.weekday)
    np.testing.assert_array_equal(a.balls.per_number[t], b.balls.per_number[t])


def test_feature_values_match_manual_computation(data: GameData) -> None:
    y, t, j = data.y_balls, 130, 11
    f = data.features.balls.per_number[t, j]
    assert f[FREQ_10] == pytest.approx(y[t - 10 : t, j].mean())
    assert f[FREQ_100] == pytest.approx(y[t - 100 : t, j].mean())
    assert f[TREND] == pytest.approx(y[t - 15 : t, j].mean() - y[t - 30 : t - 15, j].mean())
    last = np.flatnonzero(y[:t, j])
    assert f[GAP] == min(t - last[-1], GAP_CAP)


def test_draws_since_last_edge_cases() -> None:
    y = np.zeros((6, 2), dtype=np.uint8)
    y[1, 0] = 1  # el número 0 sale en el sorteo 1; el 1 nunca sale
    gaps = draws_since_last(y, np.arange(7), cap=4)
    assert gaps[:, 0].tolist() == [1, 2, 1, 2, 3, 4, 4]  # t=0: nunca visto -> t + 1
    assert gaps[:, 1].tolist() == [1, 2, 3, 4, 4, 4, 4]


def test_cannot_build_beyond_next_draw(data: GameData) -> None:
    with pytest.raises(ValueError):
        build_features(data.y_balls, data.y_sb, np.zeros(len(data) + 2))


def test_next_date_follows_calendar(history: pd.DataFrame) -> None:
    data = GameData.from_draws(history, "baloto")
    last = pd.Timestamp(data.fechas[-1]).date()
    assert data.next_date > last
    assert data.next_date.weekday() in (0, 2, 5)
