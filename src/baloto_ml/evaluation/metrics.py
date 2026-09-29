"""Métricas por balota y para la superbalota, con sus referencias exactas por azar."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from baloto_ml.config import BALLS_PER_DRAW, EXPECTED_HITS, N_SUPER, P_BALL, P_SUPER
from baloto_ml.evaluation.stats import hits_pmf, normal_mean_ci
from baloto_ml.models.base import Probabilities

EPS = 1e-15

# Referencias del baseline constante. Son exactas para CUALQUIER resultado observado, porque
# cada sorteo tiene exactamente 5 unos entre 43 posiciones y un 1 entre 16.
LOG_LOSS_BALL_CONST = -(P_BALL * math.log(P_BALL) + (1 - P_BALL) * math.log(1 - P_BALL))
BRIER_BALL_CONST = P_BALL * (1 - P_BALL)
LOG_LOSS_SB_CONST = math.log(N_SUPER)
BRIER_SB_CONST = 1 - P_SUPER
ACCURACY_SB_CHANCE = P_SUPER


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, EPS, 1 - EPS)


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-loss binaria elemento a elemento (misma forma que `p`)."""
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def top_k(p: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de los k números más probables por fila; empates rotos al azar (con semilla)."""
    order = np.lexsort((rng.random(p.shape), -p), axis=-1)
    return order[:, :k]


def hits_at_k(
    p: np.ndarray, y: np.ndarray, rng: np.random.Generator, k: int = BALLS_PER_DRAW
) -> np.ndarray:
    """Aciertos por sorteo si se juegan los k números más probables."""
    return np.take_along_axis(y, top_k(p, k, rng), axis=1).sum(axis=1)


def reliability_curve(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibración por cuantiles de probabilidad predicha: (media predicha, frecuencia
    observada, semiancho del IC95 % binomial) por grupo."""
    p, y = np.ravel(p), np.ravel(y)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    pred, obs, half = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        f = y[mask].mean()
        pred.append(p[mask].mean())
        obs.append(f)
        half.append(1.96 * np.sqrt(max(f * (1 - f), 1e-12) / mask.sum()))
    return np.array(pred), np.array(obs), np.array(half)


def skill(value: float, reference: float) -> float:
    """1 - valor/referencia: positivo es mejor que la referencia, negativo es peor."""
    return 1.0 - value / reference


@dataclass(frozen=True)
class DrawScores:
    """Puntajes por sorteo objetivo; permiten intervalos y comparaciones pareadas."""

    log_loss_balls: np.ndarray  # media de la log-loss binaria de las 43 balotas
    brier_balls: np.ndarray
    hits: np.ndarray  # aciertos jugando el top-5 (0..5)
    log_loss_sb: np.ndarray
    brier_sb: np.ndarray
    hit_sb: np.ndarray  # 1 si la superbalota más probable salió


def score_draws(
    probs: Probabilities, y_balls: np.ndarray, y_sb: np.ndarray, rng: np.random.Generator
) -> DrawScores:
    return DrawScores(
        log_loss_balls=binary_log_loss(probs.balls, y_balls).mean(axis=1),
        brier_balls=((probs.balls - y_balls) ** 2).mean(axis=1),
        hits=hits_at_k(probs.balls, y_balls, rng),
        log_loss_sb=-np.log(_clip((probs.sb * y_sb).sum(axis=1))),
        brier_sb=((probs.sb - y_sb) ** 2).sum(axis=1),
        hit_sb=hits_at_k(probs.sb, y_sb, rng, k=1),
    )


def _ci(values: np.ndarray) -> list[float]:
    _, lo, hi = normal_mean_ci(values)
    return [lo, hi]


def evaluate_probabilities(
    probs: Probabilities, y_balls: np.ndarray, y_sb: np.ndarray, rng: np.random.Generator
) -> dict:
    """Resumen de métricas frente al azar (IC95 % normales sobre los puntajes por sorteo)."""
    s = score_draws(probs, y_balls, y_sb, rng)
    m = len(s.hits)
    ll_b, br_b = float(s.log_loss_balls.mean()), float(s.brier_balls.mean())
    ll_s, br_s = float(s.log_loss_sb.mean()), float(s.brier_sb.mean())
    dist = np.bincount(s.hits, minlength=BALLS_PER_DRAW + 1)
    expected = m * hits_pmf()
    return {
        "n_sorteos": m,
        "balotas": {
            "log_loss": ll_b,
            "log_loss_constante": LOG_LOSS_BALL_CONST,
            "skill_log_loss": skill(ll_b, LOG_LOSS_BALL_CONST),
            "delta_log_loss_ic95": _ci(s.log_loss_balls - LOG_LOSS_BALL_CONST),
            "brier": br_b,
            "brier_constante": BRIER_BALL_CONST,
            "skill_brier": skill(br_b, BRIER_BALL_CONST),
            "aciertos_top5_media": float(s.hits.mean()),
            "aciertos_top5_ic95": _ci(s.hits),
            "aciertos_esperados_azar": EXPECTED_HITS,
            "distribucion_aciertos": {str(k): int(c) for k, c in enumerate(dist)},
            "distribucion_esperada_azar": {str(k): float(e) for k, e in enumerate(expected)},
            "suma_probabilidades_media": float(probs.balls.sum(axis=1).mean()),
            "log_loss_por_balota": binary_log_loss(probs.balls, y_balls).mean(axis=0).tolist(),
            "brier_por_balota": ((probs.balls - y_balls) ** 2).mean(axis=0).tolist(),
        },
        "superbalota": {
            "accuracy": float(s.hit_sb.mean()),
            "accuracy_ic95": _ci(s.hit_sb),
            "accuracy_azar": ACCURACY_SB_CHANCE,
            "log_loss": ll_s,
            "log_loss_constante": LOG_LOSS_SB_CONST,
            "skill_log_loss": skill(ll_s, LOG_LOSS_SB_CONST),
            "delta_log_loss_ic95": _ci(s.log_loss_sb - LOG_LOSS_SB_CONST),
            "brier": br_s,
            "brier_constante": BRIER_SB_CONST,
            "skill_brier": skill(br_s, BRIER_SB_CONST),
        },
    }
