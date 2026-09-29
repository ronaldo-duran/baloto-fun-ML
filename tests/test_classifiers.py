from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from baloto_ml.config import EXPECTED_HITS, EvalConfig
from baloto_ml.data.encoding import multi_hot, one_hot
from baloto_ml.data.synthetic import random_draws, sticky_draws, synthetic_history
from baloto_ml.evaluation.significance import walk_forward_metrics
from baloto_ml.models.base import GameData
from baloto_ml.models.classifiers import (
    LOGISTIC_PARAMS,
    PerNumberLogistic,
    PooledGradientBoosting,
)

MODELS = [PerNumberLogistic, PooledGradientBoosting]


@pytest.fixture(scope="module")
def data() -> GameData:
    h = synthetic_history(220, np.random.default_rng(5), juegos=("baloto",))
    return GameData.from_draws(h, "baloto")


@pytest.mark.parametrize("model_cls", MODELS)
def test_probabilities_are_valid(model_cls, data: GameData) -> None:
    model = model_cls().fit(data, np.arange(100, 200))
    p = model.predict(data, np.arange(200, 221))  # incluye el sorteo siguiente (fila n)
    assert p.balls.shape == (21, 43) and p.sb.shape == (21, 16)
    assert ((p.balls > 0) & (p.balls < 1)).all()
    np.testing.assert_allclose(p.sb.sum(axis=1), 1)
    assert 3.5 < p.balls.sum(axis=1).mean() < 6.5  # cerca de 5


@pytest.mark.parametrize("model_cls", MODELS)
def test_models_are_deterministic(model_cls, data: GameData) -> None:
    idx = np.arange(200, 220)
    a = model_cls().fit(data, np.arange(100, 200)).predict(data, idx)
    b = model_cls().fit(data, np.arange(100, 200)).predict(data, idx)
    np.testing.assert_array_equal(a.balls, b.balls)
    np.testing.assert_array_equal(a.sb, b.sb)


@pytest.mark.parametrize("model_cls", MODELS)
def test_fit_ignores_draws_after_training(model_cls, data: GameData) -> None:
    """Cambiar los sorteos posteriores al entrenamiento no cambia la predicción del siguiente."""
    train = np.arange(100, 180)
    balls, sb = random_draws(len(data) - 180, np.random.default_rng(1))
    y_b, y_s = data.y_balls.copy(), data.y_sb.copy()
    y_b[180:], y_s[180:] = multi_hot(balls), one_hot(sb)
    altered = replace(data, y_balls=y_b, y_sb=y_s)
    p1 = model_cls().fit(data, train).predict(data, np.array([180]))
    p2 = model_cls().fit(altered, train).predict(altered, np.array([180]))
    np.testing.assert_allclose(p1.balls, p2.balls)
    np.testing.assert_allclose(p1.sb, p2.sb)


def test_block_scaler_matches_sklearn_pipeline(data: GameData) -> None:
    """El escalado vectorizado equivale a un pipeline de scikit-learn por número."""
    train, idx = np.arange(100, 200), np.arange(200, 215)
    fast = PerNumberLogistic().fit(data, train).predict(data, idx)
    nf, y = data.features.balls, data.y_balls
    gap, n_pn = 3, nf.per_number.shape[2]
    for j in (0, 17, 42):
        pre = ColumnTransformer(
            [
                ("continuas", StandardScaler(), [0, 1, 2, 4]),
                ("sequia", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), [gap]),
                ("dia", "passthrough", list(range(n_pn, n_pn + 3))),
            ]
        )
        ref = make_pipeline(pre, LogisticRegression(**LOGISTIC_PARAMS))
        ref.fit(nf.design(train, j), y[train, j])
        np.testing.assert_allclose(
            fast.balls[:, j], ref.predict_proba(nf.design(idx, j))[:, 1], rtol=1e-5
        )


def test_legacy_preprocessing_is_kept_for_the_record(data: GameData) -> None:
    train, idx = np.arange(100, 200), np.arange(200, 210)
    legacy = PerNumberLogistic(legacy_preprocessing=True).fit(data, train).predict(data, idx)
    current = PerNumberLogistic().fit(data, train).predict(data, idx)
    assert not np.allclose(legacy.balls, current.balls)


def test_positive_control_gradient_boosting_detects_planted_signal() -> None:
    """Con señal plantada (lo que salió pesa 2,5 veces más en el siguiente sorteo), el
    gradient boosting compartido la detecta: supera al azar en aciertos y en log-loss."""
    balls, sb = sticky_draws(400, np.random.default_rng(3), effect=1.5)
    template = GameData.from_draws(synthetic_history(400, np.random.default_rng(0)), "baloto")
    planted = replace(template, y_balls=multi_hot(balls), y_sb=one_hot(sb))
    cfg = EvalConfig(warmup=30, min_train=120, refit_every=50)
    m = walk_forward_metrics(PooledGradientBoosting, planted, cfg)
    assert m["aciertos_top5"] > EXPECTED_HITS + 0.3
    assert m["skill_log_loss_balotas"] > 0.005


def test_logistic_ranking_reacts_to_planted_signal() -> None:
    balls, sb = sticky_draws(400, np.random.default_rng(3), effect=1.5)
    template = GameData.from_draws(synthetic_history(400, np.random.default_rng(0)), "baloto")
    planted = replace(template, y_balls=multi_hot(balls), y_sb=one_hot(sb))
    cfg = EvalConfig(warmup=30, min_train=120, refit_every=50)
    m = walk_forward_metrics(partial(PerNumberLogistic, C=0.1), planted, cfg)
    assert m["aciertos_top5"] > EXPECTED_HITS + 0.05
