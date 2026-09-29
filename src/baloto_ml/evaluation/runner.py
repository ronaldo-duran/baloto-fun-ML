"""Etapa evaluate: chequeos de los datos y evaluación walk-forward por juego."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from baloto_ml.config import EVAL, EXPECTED_HITS, JUEGOS, N_SIMS, SEED, EvalConfig, Paths
from baloto_ml.data.store import read_draws, write_json
from baloto_ml.evaluation.controls import positive_control, regularization_sensitivity
from baloto_ml.evaluation.metrics import (
    ACCURACY_SB_CHANCE,
    BRIER_BALL_CONST,
    BRIER_SB_CONST,
    LOG_LOSS_BALL_CONST,
    LOG_LOSS_SB_CONST,
    evaluate_probabilities,
)
from baloto_ml.evaluation.significance import (
    DEFAULT_RUNS,
    outcome_significance,
    significance_for_game,
    winner_percentile_summary,
    winner_percentiles,
)
from baloto_ml.evaluation.stats import (
    hits_variance,
    independence_between_games,
    rng_for,
    uniformity_balls,
    uniformity_super,
)
from baloto_ml.evaluation.walk_forward import walk_forward
from baloto_ml.models.base import Forecaster, GameData
from baloto_ml.models.catalog import MODELS, PRODUCTION_MODEL
from baloto_ml.models.classifiers import HGB_PARAMS, LOGISTIC_PARAMS

logger = logging.getLogger(__name__)


def analyze_data(draws: pd.DataFrame, n_sims: int = N_SIMS) -> dict:
    """Uniformidad de cada juego e independencia Baloto/Revancha."""
    datas = {j: GameData.from_draws(draws, j) for j in JUEGOS if (draws["juego"] == j).any()}
    out: dict = {"n_simulaciones": n_sims, "semilla": SEED, "uniformidad": {}}
    for juego, data in datas.items():
        balls = uniformity_balls(
            data.y_balls, n_sims, rng_for("uniformidad-balotas", juego)
        ).summary
        sb = uniformity_super(data.y_sb, n_sims, rng_for("uniformidad-superbalota", juego)).summary
        out["uniformidad"][juego] = {"balotas": balls, "superbalota": sb}
        logger.info(
            "%s: uniformidad balotas p_MC=%.3f | superbalota p_MC=%.3f",
            juego,
            balls["p_valor_monte_carlo"],
            sb["p_valor_monte_carlo"],
        )
    if len(datas) == 2:
        ind = independence_between_games(
            datas["baloto"], datas["revancha"], n_sims, rng_for("independencia")
        ).summary
        out["independencia_baloto_revancha"] = ind
        logger.info(
            "Independencia Baloto/Revancha: balotas en común %.3f (esperado %.3f), p_perm=%.3f",
            ind["balotas_en_comun"]["media_observada"],
            EXPECTED_HITS,
            ind["balotas_en_comun"]["p_valor_permutacion"],
        )
    return out


def evaluate_game(
    data: GameData,
    cfg: EvalConfig = EVAL,
    models: Mapping[str, Callable[[], Forecaster]] = MODELS,
    n_sims: int = N_SIMS,
) -> dict:
    """Walk-forward de cada modelo sobre la misma ventana de prueba, más su significancia
    Monte Carlo (predicciones fijas, 10 000 secuencias de sorteos justos)."""
    start = cfg.first_test_index
    n_test = len(data) - start
    sd_hits = float(np.sqrt(hits_variance() / n_test))
    out: dict = {
        "juego": data.juego,
        "ventana": {
            "warmup": cfg.warmup,
            "min_train": cfg.min_train,
            "refit_every": cfg.refit_every,
            "n_sorteos_prueba": n_test,
            "primer_sorteo_prueba": int(data.n_sorteo[start]),
            "ultimo_sorteo_prueba": int(data.n_sorteo[-1]),
            "fecha_inicio_prueba": str(pd.Timestamp(data.fechas[start]).date()),
            "fecha_fin_prueba": str(pd.Timestamp(data.fechas[-1]).date()),
        },
        "referencias_azar": {
            "log_loss_balota": LOG_LOSS_BALL_CONST,
            "brier_balota": BRIER_BALL_CONST,
            "aciertos_top5": EXPECTED_HITS,
            "aciertos_top5_banda95": [
                EXPECTED_HITS - 1.96 * sd_hits,
                EXPECTED_HITS + 1.96 * sd_hits,
            ],
            "accuracy_superbalota": ACCURACY_SB_CHANCE,
            "log_loss_superbalota": LOG_LOSS_SB_CONST,
            "brier_superbalota": BRIER_SB_CONST,
        },
        "modelo_produccion": PRODUCTION_MODEL,
        "hiperparametros": {
            "frecuencia": {"pseudo_sorteos": 10},
            "logistica": LOGISTIC_PARAMS,
            "gradient_boosting": HGB_PARAMS,
        },
        "modelos": {},
    }
    for name, factory in models.items():
        wf = walk_forward(factory, data, cfg)
        y_balls, y_sb = data.y_balls[wf.test_idx], data.y_sb[wf.test_idx]
        metrics = evaluate_probabilities(
            wf.probs, y_balls, y_sb, rng_for("desempate", data.juego, name)
        )
        metrics["significancia_monte_carlo"] = outcome_significance(
            wf.probs, metrics, n_sims, rng_for("mc-resultados", data.juego, name)
        )
        if name == PRODUCTION_MODEL:
            pct = winner_percentiles(
                wf.probs, y_balls, y_sb, rng_for("percentil-ganadoras", data.juego)
            )
            metrics["percentil_ganadoras"] = winner_percentile_summary(pct)
        out["modelos"][name] = metrics
        logger.info(
            "%s/%s: aciertos top-5 %.3f (azar %.3f) | skill log-loss %+.4f | acc SB %.3f",
            data.juego,
            name,
            metrics["balotas"]["aciertos_top5_media"],
            EXPECTED_HITS,
            metrics["balotas"]["skill_log_loss"],
            metrics["superbalota"]["accuracy"],
        )
    return out


def run_evaluate(paths: Paths, n_sims: int = N_SIMS, cfg: EvalConfig = EVAL) -> None:
    """Escribe reports/analysis.json y reports/evaluation/<juego>.json."""
    draws = read_draws(paths.processed_draws)
    if draws.empty:
        raise ValueError("No hay dataset procesado: corre antes `ingest` y `validate`")
    write_json(analyze_data(draws, n_sims), paths.analysis_report)
    for juego in JUEGOS:
        if (draws["juego"] == juego).any():
            data = GameData.from_draws(draws, juego)
            write_json(evaluate_game(data, cfg, n_sims=n_sims), paths.evaluation_report(juego))


def run_significance(
    paths: Paths,
    runs: Mapping[str, tuple[int, int]] = DEFAULT_RUNS,
    cfg: EvalConfig = EVAL,
    n_jobs: int = -1,
    controls: bool = True,
) -> None:
    """Etapa lenta (minutos, no corre en CI): permutación e historiales sintéticos con
    reentrenamiento por juego, más los controles del método sobre simulaciones."""
    draws = read_draws(paths.processed_draws)
    if draws.empty:
        raise ValueError("No hay dataset procesado: corre antes `ingest` y `validate`")
    datas = {j: GameData.from_draws(draws, j) for j in JUEGOS if (draws["juego"] == j).any()}
    for juego, data in datas.items():
        write_json(significance_for_game(data, runs, cfg, n_jobs), paths.significance_report(juego))
    if controls:
        template = next(iter(datas.values()))
        report = {
            "plantilla_fechas": template.juego,
            "control_positivo": positive_control(template, cfg=cfg, n_jobs=n_jobs),
            "regularizacion_logistica": regularization_sensitivity(template, cfg, n_jobs),
        }
        write_json(report, paths.controls_report)
        logger.info("Controles del método escritos en %s", paths.controls_report)
