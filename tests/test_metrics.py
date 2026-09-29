from __future__ import annotations

import math

import numpy as np

from baloto_ml.config import EXPECTED_HITS, P_BALL
from baloto_ml.data.encoding import multi_hot, one_hot
from baloto_ml.data.synthetic import random_draws
from baloto_ml.evaluation.metrics import (
    BRIER_BALL_CONST,
    BRIER_SB_CONST,
    LOG_LOSS_BALL_CONST,
    LOG_LOSS_SB_CONST,
    evaluate_probabilities,
    hits_at_k,
    top_k,
)
from baloto_ml.models.base import Probabilities


def outcomes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    balls, sb = random_draws(n, np.random.default_rng(seed))
    return multi_hot(balls), one_hot(sb)


def uniform_probs(n: int) -> Probabilities:
    return Probabilities(np.full((n, 43), P_BALL), np.full((n, 16), 1 / 16))


def test_constant_references_are_exact_for_any_outcome() -> None:
    y_b, y_s = outcomes(300)
    m = evaluate_probabilities(uniform_probs(300), y_b, y_s, np.random.default_rng(1))
    assert math.isclose(m["balotas"]["log_loss"], LOG_LOSS_BALL_CONST, rel_tol=1e-12)
    assert math.isclose(m["balotas"]["brier"], BRIER_BALL_CONST, rel_tol=1e-12)
    assert math.isclose(m["superbalota"]["log_loss"], LOG_LOSS_SB_CONST, rel_tol=1e-12)
    assert math.isclose(m["superbalota"]["brier"], BRIER_SB_CONST, rel_tol=1e-12)
    assert abs(m["balotas"]["skill_log_loss"]) < 1e-12
    assert math.isclose(LOG_LOSS_SB_CONST, math.log(16))
    assert math.isclose(BRIER_SB_CONST, 15 / 16)


def test_perfect_predictions() -> None:
    y_b, y_s = outcomes(50)
    probs = Probabilities(y_b.astype(float), y_s.astype(float))
    m = evaluate_probabilities(probs, y_b, y_s, np.random.default_rng(1))
    assert m["balotas"]["aciertos_top5_media"] == 5
    assert m["balotas"]["log_loss"] < 1e-10
    assert m["superbalota"]["accuracy"] == 1
    assert m["balotas"]["skill_log_loss"] > 0.99


def test_top_k_breaks_ties_randomly_but_reproducibly() -> None:
    p = np.full((4, 43), P_BALL)
    a = top_k(p, 5, np.random.default_rng(3))
    b = top_k(p, 5, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert len({tuple(sorted(r)) for r in a}) > 1  # no siempre los mismos números
    p[:, [7, 8]] = 0.5  # los más probables van primero
    assert set(top_k(p, 2, np.random.default_rng(0))[0]) == {7, 8}


def test_random_play_hits_match_chance() -> None:
    y_b, _ = outcomes(20_000, seed=5)
    hits = hits_at_k(np.full(y_b.shape, P_BALL), y_b, np.random.default_rng(6))
    assert abs(hits.mean() - EXPECTED_HITS) < 0.02


def test_evaluation_summary_is_consistent() -> None:
    y_b, y_s = outcomes(120)
    m = evaluate_probabilities(uniform_probs(120), y_b, y_s, np.random.default_rng(1))
    b = m["balotas"]
    assert sum(b["distribucion_aciertos"].values()) == 120
    assert math.isclose(sum(b["distribucion_esperada_azar"].values()), 120)
    assert len(b["log_loss_por_balota"]) == 43
    lo, hi = b["aciertos_top5_ic95"]
    assert lo < b["aciertos_top5_media"] < hi
