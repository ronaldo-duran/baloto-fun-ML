"""Significancia: ¿se distingue el resultado observado del azar?

Tres niveles, de más barato a más caro:

1. Monte Carlo de resultados (`outcome_significance`, parte de `evaluate`): se fijan las
   predicciones walk-forward del modelo y se simulan 10 000 secuencias de sorteos justos del
   mismo largo. Bajo H0 cada resultado es independiente de lo que el modelo predijo con el
   pasado, así que es la distribución exacta de sus métricas si no hubiera nada que aprender.
2. Permutación (`resampling_null(kind="permutacion")`, etapa `significance`): se baraja el
   orden de los sorteos reales y se repite TODO el walk-forward, reentrenando. Conserva las
   frecuencias reales de cada número y destruye solo el orden temporal.
3. Historiales sintéticos (`kind="sintetico"`): historiales uniformes del mismo tamaño, con todo
   el walk-forward. Muestra qué produce el pipeline completo cuando no hay señal.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps

from baloto_ml.config import (
    BALLS_PER_DRAW,
    EVAL,
    N_BALLS,
    N_SUPER,
    P_BALL,
    P_SUPER,
    EvalConfig,
)
from baloto_ml.data.encoding import multi_hot, one_hot
from baloto_ml.data.synthetic import random_draws
from baloto_ml.evaluation.metrics import EPS, evaluate_probabilities
from baloto_ml.evaluation.stats import (
    mc_p_value,
    mc_p_value_interval,
    normal_mean_ci,
    null_interval,
    percentile_of,
    rng_for,
    seed_sequence,
    verdict,
)
from baloto_ml.evaluation.walk_forward import walk_forward
from baloto_ml.models.base import Forecaster, GameData, Probabilities
from baloto_ml.models.catalog import MODELS

logger = logging.getLogger(__name__)


def summarize_null(observed: float, null: np.ndarray, better: str) -> dict:
    """Ubica el valor observado en su distribución por azar."""
    p = mc_p_value(null, observed, better)
    return {
        "observado": observed,
        "nulo_media": float(null.mean()),
        "nulo_ic95": null_interval(null),
        "percentil": percentile_of(null, observed),
        "p_valor": p,
        "p_valor_ic95": mc_p_value_interval(p, len(null)),
        "veredicto": verdict(p),
    }


# ------------------------------------------------------------- 1. Monte Carlo de resultados


def outcome_null(
    probs: Probabilities, n_sims: int, rng: np.random.Generator, chunk: int = 200
) -> dict[str, np.ndarray]:
    """Métricas del modelo en `n_sims` secuencias de sorteos justos (predicciones fijas)."""
    m = probs.balls.shape[0]
    # Aciertos del top-5: jugando CUALQUIER combinación, bajo H0 son hipergeométricos.
    hits = rng.hypergeometric(
        BALLS_PER_DRAW, N_BALLS - BALLS_PER_DRAW, BALLS_PER_DRAW, size=(n_sims, m)
    ).mean(axis=1)
    # Log-loss por balota de un sorteo D: a_t - (1/43) * sum_{j en D} logit(p_tj).
    p = np.clip(probs.balls, EPS, 1 - EPS)
    base = -np.log1p(-p).mean(axis=1)
    logit = (np.log(p) - np.log1p(-p))[None, :, :]
    ll = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        b = min(chunk, n_sims - start)
        keys = rng.random((b, m, N_BALLS), dtype=np.float32)
        drawn = np.argpartition(keys, BALLS_PER_DRAW, axis=2)[..., :BALLS_PER_DRAW]
        picked = np.take_along_axis(logit, drawn, axis=2).sum(axis=2)
        ll[start : start + b] = (base[None, :] - picked / N_BALLS).mean(axis=1)
    # Superbalota: se acierta con prob. 1/16 sin importar la elección; log-loss con k uniforme.
    acc_sb = rng.binomial(m, P_SUPER, size=n_sims) / m
    k = rng.integers(0, N_SUPER, size=(n_sims, m))
    ll_sb = -np.log(np.clip(probs.sb, EPS, 1))[np.arange(m)[None, :], k].mean(axis=1)
    return {
        "aciertos_top5": hits,
        "log_loss_balotas": ll,
        "accuracy_superbalota": acc_sb,
        "log_loss_superbalota": ll_sb,
    }


def outcome_significance(
    probs: Probabilities, metrics: dict, n_sims: int, rng: np.random.Generator
) -> dict:
    """p-valores de 'mejor que el azar' para las métricas de un modelo."""
    null = outcome_null(probs, n_sims, rng)
    observed = {
        "aciertos_top5": (metrics["balotas"]["aciertos_top5_media"], "greater"),
        "log_loss_balotas": (metrics["balotas"]["log_loss"], "less"),
        "accuracy_superbalota": (metrics["superbalota"]["accuracy"], "greater"),
        "log_loss_superbalota": (metrics["superbalota"]["log_loss"], "less"),
    }
    out = {k: summarize_null(obs, null[k], better) for k, (obs, better) in observed.items()}
    return out | {"n_simulaciones": n_sims}


def combination_scores(p_balls: np.ndarray, p_sb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Puntaje de cada número: log(p / p_azar). El de una jugada es la suma de sus 5 + 1."""
    return (
        np.log(np.clip(p_balls, EPS, 1) / P_BALL),
        np.log(np.clip(p_sb, EPS, 1) / P_SUPER),
    )


def winner_percentiles(
    probs: Probabilities,
    y_balls: np.ndarray,
    y_sb: np.ndarray,
    rng: np.random.Generator,
    n_random: int = 2000,
) -> np.ndarray:
    """Percentil (0-1) de la combinación ganadora real entre `n_random` jugadas al azar,
    según el puntaje que el modelo le daba ANTES del sorteo. Sin señal, es uniforme."""
    sw, ss = combination_scores(probs.balls, probs.sb)
    actual = (sw * y_balls).sum(axis=1) + (ss * y_sb).sum(axis=1)
    out = np.empty(len(actual))
    for t in range(len(actual)):
        keys = rng.random((n_random, N_BALLS), dtype=np.float32)
        picked = np.argpartition(keys, BALLS_PER_DRAW, axis=1)[:, :BALLS_PER_DRAW]
        scores = sw[t][picked].sum(axis=1) + ss[t][rng.integers(0, N_SUPER, n_random)]
        below = np.sum(scores < actual[t] - 1e-12)
        ties = np.sum(np.abs(scores - actual[t]) <= 1e-12)
        out[t] = (below + 0.5 * ties) / n_random
    return out


def winner_percentile_summary(percentiles: np.ndarray) -> dict:
    mean, lo, hi = normal_mean_ci(percentiles)
    counts, _ = np.histogram(percentiles, bins=10, range=(0, 1))
    return {
        "media": mean,
        "media_ic95": [lo, hi],
        "media_esperada_azar": 0.5,
        "p_valor_ks_uniforme": float(sps.kstest(percentiles, "uniform").pvalue),
        "histograma_deciles": counts.tolist(),
        "percentiles": [round(float(p), 4) for p in percentiles],
    }


# --------------------------------------------------- 2 y 3. Remuestreo con reentrenamiento


def permute_draws(data: GameData, rng: np.random.Generator) -> GameData:
    """Mismos sorteos en otro orden (fechas intactas: el día de la semana queda al azar)."""
    perm = rng.permutation(len(data))
    return replace(data, y_balls=data.y_balls[perm], y_sb=data.y_sb[perm])


def synthetic_draws(data: GameData, rng: np.random.Generator) -> GameData:
    """Historial uniforme del mismo tamaño y con las mismas fechas."""
    balls, sb = random_draws(len(data), rng)
    return replace(data, y_balls=multi_hot(balls), y_sb=one_hot(sb))


RESAMPLERS: dict[str, Callable[[GameData, np.random.Generator], GameData]] = {
    "permutacion": permute_draws,
    "sintetico": synthetic_draws,
}
RESAMPLING_BETTER = {
    "skill_log_loss_balotas": "greater",
    "aciertos_top5": "greater",
    "skill_log_loss_superbalota": "greater",
    "accuracy_superbalota": "greater",
}


def walk_forward_metrics(
    factory: Callable[[], Forecaster], data: GameData, cfg: EvalConfig = EVAL
) -> dict[str, float]:
    wf = walk_forward(factory, data, cfg)
    m = evaluate_probabilities(
        wf.probs,
        data.y_balls[wf.test_idx],
        data.y_sb[wf.test_idx],
        rng_for("desempate", data.juego),
    )
    return {
        "skill_log_loss_balotas": m["balotas"]["skill_log_loss"],
        "aciertos_top5": m["balotas"]["aciertos_top5_media"],
        "skill_log_loss_superbalota": m["superbalota"]["skill_log_loss"],
        "accuracy_superbalota": m["superbalota"]["accuracy"],
    }


def _one_resample(
    factory: Callable[[], Forecaster],
    data: GameData,
    kind: str,
    cfg: EvalConfig,
    seed: np.random.SeedSequence,
) -> dict[str, float]:
    resampled = RESAMPLERS[kind](data, np.random.default_rng(seed))
    return walk_forward_metrics(factory, resampled, cfg)


def resampling_null(
    factory: Callable[[], Forecaster],
    data: GameData,
    kind: str,
    n: int,
    cfg: EvalConfig = EVAL,
    n_jobs: int = -1,
    model_name: str = "",
) -> dict[str, np.ndarray]:
    """`n` walk-forward completos sobre historiales permutados o sintéticos (en paralelo)."""
    seeds = seed_sequence("remuestreo", kind, data.juego, model_name).spawn(n)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_resample)(factory, data, kind, cfg, s) for s in seeds
    )
    return {k: np.array([r[k] for r in rows]) for k in RESAMPLING_BETTER}


# Simulaciones por modelo: (permutaciones, historiales sintéticos). La logística es barata.
DEFAULT_RUNS: dict[str, tuple[int, int]] = {
    "logistica": (200, 1000),
    "gradient_boosting": (200, 200),
}


def significance_for_game(
    data: GameData,
    runs: Mapping[str, tuple[int, int]] = DEFAULT_RUNS,
    cfg: EvalConfig = EVAL,
    n_jobs: int = -1,
) -> dict:
    out: dict = {
        "juego": data.juego,
        "n_sorteos": len(data),
        "ultimo_sorteo": int(data.n_sorteo[-1]),
        "ventana": {
            "warmup": cfg.warmup,
            "min_train": cfg.min_train,
            "refit_every": cfg.refit_every,
        },
        "modelos": {},
    }
    for name, (n_perm, n_synth) in runs.items():
        factory = MODELS[name]
        observed = walk_forward_metrics(factory, data, cfg)
        result: dict = {"observado": observed}
        for kind, n in (("permutacion", n_perm), ("sintetico", n_synth)):
            if n <= 0:
                continue
            t0 = time.perf_counter()
            null = resampling_null(factory, data, kind, n, cfg, n_jobs, name)
            result[kind] = {
                "n": n,
                "metricas": {
                    k: summarize_null(observed[k], null[k], RESAMPLING_BETTER[k])
                    | {"muestras": null[k].tolist()}
                    for k in observed
                },
            }
            skill = result[kind]["metricas"]["skill_log_loss_balotas"]
            logger.info(
                "%s/%s %s x%d (%.0fs): skill log-loss %.4f, percentil %.0f, p=%.3f",
                data.juego,
                name,
                kind,
                n,
                time.perf_counter() - t0,
                observed["skill_log_loss_balotas"],
                skill["percentil"],
                skill["p_valor"],
            )
        out["modelos"][name] = result
    return out


def significance_frame(report: dict) -> pd.DataFrame:
    """Tabla resumida de un reporte de significancia (para notebooks y la app)."""
    rows = []
    for name, res in report["modelos"].items():
        for kind in ("permutacion", "sintetico"):
            if kind not in res:
                continue
            for metric, s in res[kind]["metricas"].items():
                rows.append(
                    {
                        "modelo": name,
                        "prueba": kind,
                        "n": res[kind]["n"],
                        "metrica": metric,
                        "observado": s["observado"],
                        "nulo_media": s["nulo_media"],
                        "percentil": s["percentil"],
                        "p_valor": s["p_valor"],
                    }
                )
    return pd.DataFrame(rows)
