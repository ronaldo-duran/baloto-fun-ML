"""La trampa (lo que NO se debe hacer) debe seguir siendo convincente... y detectable."""

from __future__ import annotations

import numpy as np
import pytest

from baloto_ml.config import EXPECTED_HITS, EvalConfig
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.evaluation.trap import rolling_frequency, selection_bias_demo, trap_grid
from baloto_ml.models.base import GameData


@pytest.fixture(scope="module")
def fair() -> GameData:
    """Lotería justa: aquí no hay nada que aprender."""
    h = synthetic_history(330, np.random.default_rng(11), juegos=("baloto",))
    return GameData.from_draws(h, "baloto")


def test_leaky_window_includes_the_current_draw(fair: GameData) -> None:
    y, t = fair.y_balls, 200
    leaky = rolling_frequency(y, include_current=True)
    honest = rolling_frequency(y, include_current=False)
    np.testing.assert_allclose(leaky[t, :, 0], y[t - 9 : t + 1].mean(axis=0))  # incluye t
    np.testing.assert_allclose(honest[t, :, 0], y[t - 10 : t].mean(axis=0))  # hasta t-1


def test_leak_fabricates_signal_on_a_fair_lottery(fair: GameData) -> None:
    cfg = EvalConfig(warmup=100, min_train=100, refit_every=50)
    rows = {(r["features"][:5], r["validacion"]): r for r in trap_grid(fair, cfg)}
    for validation in ("split aleatorio", "walk-forward"):
        leaky, honest = rows[("con f", validation)], rows[("corre", validation)]
        assert leaky["aciertos_top5"] > EXPECTED_HITS + 0.5  # "¡predice la lotería!"
        assert leaky["skill_log_loss"] > 0.03
        assert honest["skill_log_loss"] < 0.01  # sin fuga: nada


def test_selection_bias_demo(fair: GameData) -> None:
    cfg = EvalConfig(warmup=100, min_train=100, refit_every=50)
    demo = selection_bias_demo(fair, cfg)
    assert len(demo["estrategias"]) == 80
    best = demo["mejor_en_A"]
    assert best["aciertos_A"] == max(r["aciertos_A"] for r in demo["estrategias"])
    assert demo["periodos"]["A"]["n_sorteos"] + demo["periodos"]["B"]["n_sorteos"] == 130
