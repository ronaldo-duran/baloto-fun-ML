"""Clasificadores con hiperparámetros FIJADOS A PRIORI.

Los valores se eligieron antes de ver cualquier resultado y no se ajustan después: probar
configuraciones hasta que algo "funcione" es la forma más segura de encontrar señal en el ruido.
Ambos modelos ven exactamente las mismas features (ver `baloto_ml.features.build`).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from baloto_ml.config import N_BALLS, N_SUPER, SEED
from baloto_ml.features.build import PER_NUMBER_NAMES, NumberFeatures
from baloto_ml.models.base import Forecaster, GameData, Probabilities

LOGISTIC_PARAMS: dict = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}  # L2
HGB_PARAMS: dict = {
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    # Su early stopping valida con un split ALEATORIO interno: prohibido en series de tiempo.
    "early_stopping": False,
}


class _ConstantRate:
    """Respaldo cuando un número no aparece (o siempre aparece) en el entrenamiento."""

    def __init__(self, y: np.ndarray) -> None:
        self.p = (y.sum() + 1) / (len(y) + 2)  # suavizado de Laplace

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def _normalize(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


_GAP = PER_NUMBER_NAMES.index("sorteos_desde_ultima")
_N_PER_NUMBER = len(PER_NUMBER_NAMES)


def design_all(nf: NumberFeatures, idx: np.ndarray) -> np.ndarray:
    """(len(idx), k, F + S): la matriz de diseño de los k modelos de un bombo, en bloque."""
    m, k = len(idx), nf.n_numbers
    shared = np.broadcast_to(nf.shared[idx][:, None, :], (m, k, nf.shared.shape[1]))
    return np.concatenate([nf.per_number[idx], shared], axis=2)


class BomboScaler:
    """Escalado de los k modelos de un bombo a la vez (equivale a un StandardScaler por número).

    Versión actual: log1p en la sequía (cola larga), estandarización de frecuencias, tendencia y
    sequía, e indicadores de día intactos (0/1).

    `legacy=True` reproduce la versión inicial, que estandarizaba todo tal cual. Tenía dos
    defectos que producían probabilidades imposibles (hasta 0,99 para una balota): la variable
    "lunes" casi siempre vale 0 al comienzo y, estandarizada, un lunes valía ~10 desviaciones,
    lo que anulaba la regularización; y una sequía larga quedaba a 6-8 desviaciones, donde la
    logística extrapola sin freno. Se conserva para documentar el hallazgo.
    """

    def __init__(self, legacy: bool = False) -> None:
        self.legacy = legacy

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        if not self.legacy:
            x[..., _GAP] = np.log1p(x[..., _GAP])
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare(x)
        self.cols_ = slice(None) if self.legacy else slice(0, _N_PER_NUMBER)
        self.mean_ = x[:, :, self.cols_].mean(axis=0)
        std = x[:, :, self.cols_].std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)  # como StandardScaler: varianza 0 -> escala 1
        return self._apply(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._apply(self._prepare(x))

    def _apply(self, x: np.ndarray) -> np.ndarray:
        x[:, :, self.cols_] = (x[:, :, self.cols_] - self.mean_) / self.scale_
        return x


class PerNumberLogistic(Forecaster):
    """Regresión logística multietiqueta: un modelo por balota (43) y por superbalota (16).

    El modelo del número j ve solo las features de j y el día de la semana, así que puede
    capturar un sesgo propio de cada número. Las 16 probabilidades de la superbalota se
    normalizan para sumar 1; las 43 de las balotas quedan tal cual (suman ~5).
    """

    name = "logistica"

    def __init__(self, legacy_preprocessing: bool = False, **params) -> None:
        self.legacy_preprocessing = legacy_preprocessing
        self.params = LOGISTIC_PARAMS | params

    def _fit_bombo(
        self, nf: NumberFeatures, y: np.ndarray, idx: np.ndarray
    ) -> tuple[BomboScaler, list]:
        scaler = BomboScaler(self.legacy_preprocessing)
        x = scaler.fit_transform(design_all(nf, idx))
        models: list[LogisticRegression | _ConstantRate] = []
        for j in range(nf.n_numbers):
            target = y[idx, j]
            if 0 < target.sum() < len(target):
                models.append(LogisticRegression(**self.params).fit(x[:, j, :], target))
            else:
                models.append(_ConstantRate(target))
        return scaler, models

    @staticmethod
    def _predict_bombo(
        fitted: tuple[BomboScaler, list], nf: NumberFeatures, idx: np.ndarray
    ) -> np.ndarray:
        scaler, models = fitted
        x = scaler.transform(design_all(nf, idx))
        return np.column_stack([m.predict_proba(x[:, j, :])[:, 1] for j, m in enumerate(models)])

    def fit(self, data: GameData, train_idx: np.ndarray) -> PerNumberLogistic:
        fs = data.features
        self.balls_ = self._fit_bombo(fs.balls, data.y_balls, train_idx)
        self.sb_ = self._fit_bombo(fs.sb, data.y_sb, train_idx)
        return self

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        fs = data.features
        balls = self._predict_bombo(self.balls_, fs.balls, idx)
        sb = _normalize(self._predict_bombo(self.sb_, fs.sb, idx))
        return Probabilities(balls, sb)


class PooledGradientBoosting(Forecaster):
    """Gradient boosting compartido: un modelo para las 43 balotas y otro para la superbalota.

    Formato largo: una fila por (sorteo, número) con las features de ese número y sin su
    identidad. Busca una dinámica común a todos ("caliente/frío", rachas, día de la semana) con
    43 veces más filas que un modelo por número.
    """

    name = "gradient_boosting"

    def __init__(self, seed: int = SEED, **params) -> None:
        self.params = HGB_PARAMS | {"random_state": seed} | params

    def _fit_bombo(self, nf: NumberFeatures, y: np.ndarray, idx: np.ndarray):
        model = HistGradientBoostingClassifier(**self.params)
        return model.fit(nf.long_design(idx), y[idx].ravel())

    def fit(self, data: GameData, train_idx: np.ndarray) -> PooledGradientBoosting:
        fs = data.features
        self.balls_ = self._fit_bombo(fs.balls, data.y_balls, train_idx)
        self.sb_ = self._fit_bombo(fs.sb, data.y_sb, train_idx)
        return self

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        fs, m = data.features, len(idx)
        balls = self.balls_.predict_proba(fs.balls.long_design(idx))[:, 1].reshape(m, N_BALLS)
        sb = self.sb_.predict_proba(fs.sb.long_design(idx))[:, 1].reshape(m, N_SUPER)
        return Probabilities(balls, _normalize(sb))
