from __future__ import annotations

import numpy as np
import pytest

from baloto_ml.config import EXPECTED_HITS, P_BALL, P_SUPER, EvalConfig
from baloto_ml.data.encoding import multi_hot, one_hot
from baloto_ml.data.synthetic import random_draws, synthetic_history
from baloto_ml.evaluation.metrics import LOG_LOSS_BALL_CONST, evaluate_probabilities
from baloto_ml.evaluation.significance import (
    outcome_null,
    outcome_significance,
    permute_draws,
    resampling_null,
    significance_for_game,
    synthetic_draws,
    winner_percentiles,
)
from baloto_ml.models.base import GameData, Probabilities
from baloto_ml.models.baselines import FrequencyBaseline


def uniform_probs(m: int) -> Probabilities:
    return Probabilities(np.full((m, 43), P_BALL), np.full((m, 16), P_SUPER))


def outcomes(m: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    balls, sb = random_draws(m, np.random.default_rng(seed))
    return multi_hot(balls), one_hot(sb)


def test_outcome_null_under_uniform_predictions() -> None:
    null = outcome_null(uniform_probs(300), 3000, np.random.default_rng(0))
    assert abs(null["aciertos_top5"].mean() - EXPECTED_HITS) < 0.005
    # Con probabilidades constantes la log-loss no depende del resultado: siempre H(5/43).
    np.testing.assert_allclose(null["log_loss_balotas"], LOG_LOSS_BALL_CONST, rtol=1e-6)
    assert abs(null["accuracy_superbalota"].mean() - P_SUPER) < 0.002


def test_outcome_null_log_loss_matches_direct_simulation() -> None:
    """El atajo algebraico de la log-loss coincide en media con simular los sorteos."""
    rng = np.random.default_rng(1)
    p = np.clip(rng.normal(P_BALL, 0.03, size=(50, 43)), 0.01, 0.5)
    probs = Probabilities(p, np.full((50, 16), P_SUPER))
    fast = outcome_null(probs, 4000, np.random.default_rng(2))["log_loss_balotas"]
    direct = []
    for s in range(400):
        y_b, y_s = outcomes(50, seed=100 + s)
        direct.append(evaluate_probabilities(probs, y_b, y_s, rng)["balotas"]["log_loss"])
    assert fast.mean() == pytest.approx(np.mean(direct), rel=2e-3)


def test_outcome_significance_flags_an_oracle() -> None:
    y_b, y_s = outcomes(200)
    oracle = Probabilities(np.clip(y_b * 0.9 + 0.01, 0, 1), np.clip(y_s * 0.8 + 0.01, 0, 1))
    metrics = evaluate_probabilities(oracle, y_b, y_s, np.random.default_rng(0))
    sig = outcome_significance(oracle, metrics, 2000, np.random.default_rng(1))
    assert sig["aciertos_top5"]["p_valor"] < 0.001
    assert sig["log_loss_balotas"]["p_valor"] < 0.001
    assert sig["n_simulaciones"] == 2000


def test_winner_percentiles() -> None:
    y_b, y_s = outcomes(40)
    flat = winner_percentiles(uniform_probs(40), y_b, y_s, np.random.default_rng(0), 500)
    np.testing.assert_allclose(flat, 0.5)  # todo empata: percentil medio
    oracle = Probabilities(np.clip(y_b * 0.6 + 0.05, 0, 1), np.clip(y_s * 0.6 + 0.02, 0, 1))
    high = winner_percentiles(oracle, y_b, y_s, np.random.default_rng(0), 500)
    assert (high > 0.99).all()


def test_resamplers_keep_structure(history) -> None:
    data = GameData.from_draws(history, "baloto")
    perm = permute_draws(data, np.random.default_rng(0))
    assert sorted(map(tuple, perm.y_balls)) == sorted(map(tuple, data.y_balls))
    np.testing.assert_array_equal(perm.fechas, data.fechas)
    synth = synthetic_draws(data, np.random.default_rng(0))
    assert (synth.y_balls.sum(axis=1) == 5).all() and (synth.y_sb.sum(axis=1) == 1).all()


def test_resampling_null_is_reproducible_and_parallel_safe(history) -> None:
    data = GameData.from_draws(history, "baloto")
    cfg = EvalConfig(warmup=10, min_train=20, refit_every=10)
    a = resampling_null(FrequencyBaseline, data, "permutacion", 4, cfg, n_jobs=1)
    b = resampling_null(FrequencyBaseline, data, "permutacion", 4, cfg, n_jobs=2)
    assert len(a["aciertos_top5"]) == 4
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_significance_for_game_structure() -> None:
    h = synthetic_history(80, np.random.default_rng(4), juegos=("baloto",))
    data = GameData.from_draws(h, "baloto")
    cfg = EvalConfig(warmup=10, min_train=30, refit_every=20)
    rep = significance_for_game(data, {"frecuencia": (3, 2)}, cfg, n_jobs=1)
    res = rep["modelos"]["frecuencia"]
    assert res["permutacion"]["n"] == 3 and res["sintetico"]["n"] == 2
    s = res["permutacion"]["metricas"]["skill_log_loss_balotas"]
    assert len(s["muestras"]) == 3 and 0 < s["p_valor"] <= 1
