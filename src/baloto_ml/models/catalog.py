"""Catálogo de modelos evaluados y el modelo de producción."""

from __future__ import annotations

from collections.abc import Callable

from baloto_ml.models.base import Forecaster
from baloto_ml.models.baselines import ConstantBaseline, FrequencyBaseline
from baloto_ml.models.classifiers import PerNumberLogistic, PooledGradientBoosting

BASELINES: dict[str, Callable[[], Forecaster]] = {
    "constante": ConstantBaseline,
    "frecuencia": FrequencyBaseline,
}
CLASSIFIERS: dict[str, Callable[[], Forecaster]] = {
    "logistica": PerNumberLogistic,
    "gradient_boosting": PooledGradientBoosting,
}
MODELS: dict[str, Callable[[], Forecaster]] = BASELINES | CLASSIFIERS

# El modelo que alimenta el registro en vivo se fija DE ANTEMANO (el más simple), no se elige
# por métricas: escoger "el que mejor salió" en la validación sería sesgo de selección.
PRODUCTION_MODEL = "logistica"
