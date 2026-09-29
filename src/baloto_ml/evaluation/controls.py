"""Controles del método, SOLO con simulaciones (los datos reales no intervienen).

- Control positivo: historiales sintéticos con señal PLANTADA ("pegajosa": lo que salió en el
  sorteo anterior pesa más). Si un modelo no la detecta, su resultado negativo con los datos
  reales no prueba nada.
- Sensibilidad a la regularización de la logística: la misma evaluación con varios C, en
  historiales nulos y con señal. Es un análisis, NO una selección: C queda fijo en su valor a
  priori (1,0); elegirlo mirando resultados sería otra forma de buscar hasta encontrar.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
from joblib import Parallel, delayed

from baloto_ml.config import EVAL, EvalConfig
from baloto_ml.data.encoding import multi_hot, one_hot
from baloto_ml.data.synthetic import random_draws, sticky_draws
from baloto_ml.evaluation.significance import walk_forward_metrics
from baloto_ml.evaluation.stats import rng_for
from baloto_ml.models.base import GameData
from baloto_ml.models.catalog import MODELS
from baloto_ml.models.classifiers import PerNumberLogistic

EFFECTS: tuple[float, ...] = (0.0, 0.5, 1.5)  # 0 = historial nulo (sin señal)
SEEDS: tuple[int, ...] = (1, 2)
C_GRID: tuple[float, ...] = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)


def simulated_game(template: GameData, effect: float, seed: int) -> GameData:
    """Historial sintético con las fechas de `template`: nulo si effect == 0, pegajoso si no."""
    rng = rng_for("control", effect, seed)
    n = len(template)
    balls, sb = sticky_draws(n, rng, effect) if effect > 0 else random_draws(n, rng)
    return replace(template, y_balls=multi_hot(balls), y_sb=one_hot(sb))


def _run(factory, template: GameData, effect: float, seed: int, cfg: EvalConfig) -> dict:
    return walk_forward_metrics(factory, simulated_game(template, effect, seed), cfg)


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, g in groups.items():
        out.append(
            dict(zip(keys, key, strict=True))
            | {
                "n_semillas": len(g),
                "aciertos_top5": float(np.mean([r["aciertos_top5"] for r in g])),
                "skill_log_loss_balotas": float(np.mean([r["skill_log_loss_balotas"] for r in g])),
            }
        )
    return out


def positive_control(
    template: GameData,
    models: tuple[str, ...] = ("frecuencia", "logistica", "gradient_boosting"),
    cfg: EvalConfig = EVAL,
    n_jobs: int = -1,
) -> list[dict]:
    """Walk-forward de cada modelo en historiales nulos y con señal plantada."""
    jobs = [(m, e, s) for m in models for e in EFFECTS for s in SEEDS]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run)(MODELS[m], template, e, s, cfg) for m, e, s in jobs
    )
    rows = [
        {"modelo": m, "efecto": e, "semilla": s} | r
        for (m, e, s), r in zip(jobs, results, strict=True)
    ]
    return _aggregate(rows, ("modelo", "efecto"))


def regularization_sensitivity(
    template: GameData, cfg: EvalConfig = EVAL, n_jobs: int = -1
) -> list[dict]:
    """La logística con cada C de la grilla, en historiales nulos y con señal plantada."""
    jobs = [(c, e, s) for c in C_GRID for e in EFFECTS for s in SEEDS]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run)(partial(PerNumberLogistic, C=c), template, e, s, cfg) for c, e, s in jobs
    )
    rows = [
        {"C": c, "efecto": e, "semilla": s} | r for (c, e, s), r in zip(jobs, results, strict=True)
    ]
    return _aggregate(rows, ("C", "efecto"))
