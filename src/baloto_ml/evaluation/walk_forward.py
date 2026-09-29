"""Validación walk-forward con ventana creciente: nunca se entrena con el futuro.

Para cada bloque de `refit_every` sorteos se entrena con TODOS los sorteos objetivo anteriores
al bloque (desde `warmup`) y se predice el bloque. Las features de cada sorteo ya usan solo el
pasado, así que dentro del bloque se actualizan sorteo a sorteo como en producción.
Un split aleatorio (train_test_split con shuffle) está prohibido: mezcla pasado y futuro.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from baloto_ml.config import EVAL, N_BALLS, N_SUPER, EvalConfig
from baloto_ml.models.base import Forecaster, GameData, Probabilities


@dataclass(frozen=True)
class WalkForwardResult:
    test_idx: np.ndarray  # índices de los sorteos de prueba
    probs: Probabilities  # predicciones alineadas con test_idx
    n_refits: int


def walk_forward(
    factory: Callable[[], Forecaster], data: GameData, cfg: EvalConfig = EVAL
) -> WalkForwardResult:
    """Predicciones fuera de muestra para todos los sorteos desde `cfg.first_test_index`."""
    n, start = len(data), cfg.first_test_index
    if n <= start:
        raise ValueError(f"Se necesitan más de {start} sorteos para el walk-forward; hay {n}")
    test_idx = np.arange(start, n)
    balls = np.empty((n - start, N_BALLS))
    sb = np.empty((n - start, N_SUPER))
    n_refits = 0
    for block_start in range(start, n, cfg.refit_every):
        block = np.arange(block_start, min(block_start + cfg.refit_every, n))
        train_idx = np.arange(cfg.warmup, block_start)  # estrictamente antes del bloque
        model = factory().fit(data, train_idx)
        n_refits += 1
        p = model.predict(data, block)
        balls[block - start] = p.balls
        sb[block - start] = p.sb
    return WalkForwardResult(test_idx, Probabilities(balls, sb), n_refits)
