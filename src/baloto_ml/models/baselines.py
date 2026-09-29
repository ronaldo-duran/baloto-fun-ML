"""Baselines obligatorios: el listón que cualquier modelo tendría que superar."""

from __future__ import annotations

import numpy as np

from baloto_ml.config import N_BALLS, N_SUPER, P_BALL, P_SUPER
from baloto_ml.features.history import past_counts
from baloto_ml.models.base import Forecaster, GameData, Probabilities


class ConstantBaseline(Forecaster):
    """5/43 por balota y 1/16 por superbalota: el modelo correcto si el sorteo es justo."""

    name = "constante"

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        m = len(idx)
        return Probabilities(np.full((m, N_BALLS), P_BALL), np.full((m, N_SUPER), P_SUPER))


class FrequencyBaseline(Forecaster):
    """Frecuencia histórica acumulada, suavizada hacia el uniforme, con sorteos anteriores a t.

    p_j(t) = (c_j(t) + a * p0) / (t + a), donde c_j(t) son las apariciones de j en los t sorteos
    previos, p0 = 5/43 (balotas) o 1/16 (superbalota) y a = `pseudo_draws` sorteos ficticios.
    Las probabilidades de las balotas suman 5 y las de la superbalota suman 1.
    """

    name = "frecuencia"

    def __init__(self, pseudo_draws: float = 10.0) -> None:
        self.pseudo_draws = pseudo_draws

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        idx = np.asarray(idx)
        a = self.pseudo_draws
        t = idx.astype(float)[:, None]
        balls = (past_counts(data.y_balls)[idx] + a * P_BALL) / (t + a)
        sb = (past_counts(data.y_sb)[idx] + a * P_SUPER) / (t + a)
        return Probabilities(balls, sb)
