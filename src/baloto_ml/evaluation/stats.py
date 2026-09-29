"""Pruebas estadísticas: uniformidad, independencia entre juegos y utilidades Monte Carlo.

Convención de p-valores Monte Carlo: (1 + #extremos) / (1 + #simulaciones), que nunca da 0.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy import stats as sps

from baloto_ml.config import BALLS_PER_DRAW, EXPECTED_HITS, N_BALLS, N_SUPER, P_BALL, P_SUPER, SEED
from baloto_ml.models.base import GameData

ALPHA = 0.05


@dataclass(frozen=True)
class TestResult:
    """Resumen serializable (reportes) + arreglos para graficar (distribuciones nulas, etc.)."""

    summary: dict
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


# ------------------------------------------------------------------------------ utilidades


def seed_sequence(*keys: str | int, seed: int = SEED) -> np.random.SeedSequence:
    """Semilla derivada de claves estables (crc32, no `hash()`, que cambia entre ejecuciones)."""
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode("utf-8")) for k in keys]
    return np.random.SeedSequence([seed, *ints])


def rng_for(*keys: str | int, seed: int = SEED) -> np.random.Generator:
    """Generador reproducible e independiente por análisis (no depende del orden de ejecución)."""
    return np.random.default_rng(seed_sequence(*keys, seed=seed))


def hits_pmf(
    n_numbers: int = N_BALLS, n_drawn: int = BALLS_PER_DRAW, n_picked: int = BALLS_PER_DRAW
) -> np.ndarray:
    """P(k aciertos) al jugar `n_picked` números si se extraen `n_drawn` de `n_numbers`."""
    total = comb(n_numbers, n_drawn)
    return np.array(
        [
            comb(n_drawn, k) * comb(n_numbers - n_drawn, n_picked - k) / total
            for k in range(n_picked + 1)
        ]
    )


def hits_variance(
    n_numbers: int = N_BALLS, n_drawn: int = BALLS_PER_DRAW, n_picked: int = BALLS_PER_DRAW
) -> float:
    """Varianza hipergeométrica de los aciertos por sorteo (≈ 0,465 para 5 de 43)."""
    p = n_drawn / n_numbers
    return n_picked * p * (1 - p) * (n_numbers - n_picked) / (n_numbers - 1)


def mc_p_value(
    simulated: np.ndarray,
    observed: float,
    alternative: str = "greater",
    center: float | None = None,
) -> float:
    """P-valor Monte Carlo; en 'two-sided' mide la distancia a `center` (por defecto, la media)."""
    sims = np.asarray(simulated, dtype=float)
    if alternative == "greater":
        extreme = sims >= observed - 1e-12
    elif alternative == "less":
        extreme = sims <= observed + 1e-12
    elif alternative == "two-sided":
        c = sims.mean() if center is None else center
        extreme = np.abs(sims - c) >= abs(observed - c) - 1e-12
    else:
        raise ValueError(f"alternative desconocida: {alternative}")
    return float((1 + extreme.sum()) / (1 + len(sims)))


def mc_p_value_interval(p_value: float, n_sims: int, z: float = 1.96) -> list[float]:
    """IC de Wilson para un p-valor Monte Carlo: incertidumbre por usar simulaciones finitas."""
    n = n_sims + 1
    denom = 1 + z**2 / n
    center = (p_value + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_value * (1 - p_value) / n + z**2 / (4 * n**2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def normal_mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Media e IC95 % normal (sorteos independientes entre sí)."""
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    half = z * float(v.std(ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return mean, mean - half, mean + half


def null_interval(simulated: np.ndarray, level: float = 0.95) -> list[float]:
    """Intervalo central de la distribución por azar (percentiles)."""
    lo, hi = np.quantile(simulated, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def percentile_of(simulated: np.ndarray, observed: float) -> float:
    """Percentil (0-100) del valor observado dentro de la distribución simulada."""
    return float(100 * np.mean(np.asarray(simulated) <= observed))


def verdict(p_value: float, alpha: float = ALPHA) -> str:
    if p_value < alpha:
        return f"se rechaza H0 (p < {alpha})"
    return f"no se rechaza H0 (p >= {alpha})"


# ------------------------------------------------------------------------------ uniformidad


def simulate_ball_counts(
    n_draws: int, n_sims: int, rng: np.random.Generator, chunk: int = 250
) -> np.ndarray:
    """Conteos por balota de `n_sims` historiales uniformes de `n_draws` sorteos (5 de 43 sin
    reemplazo en cada sorteo), por bloques para acotar memoria."""
    out = np.empty((n_sims, N_BALLS), dtype=np.int64)
    for start in range(0, n_sims, chunk):
        b = min(chunk, n_sims - start)
        keys = rng.random((b * n_draws, N_BALLS), dtype=np.float32)
        picked = np.argpartition(keys, BALLS_PER_DRAW, axis=1)[:, :BALLS_PER_DRAW]
        sim_id = np.repeat(np.arange(b), n_draws * BALLS_PER_DRAW)
        flat = np.bincount(sim_id * N_BALLS + picked.ravel(), minlength=b * N_BALLS)
        out[start : start + b] = flat.reshape(b, N_BALLS)
    return out


def chi_square_statistic(counts: np.ndarray, expected: float) -> np.ndarray:
    return ((counts - expected) ** 2 / expected).sum(axis=-1)


def uniformity_balls(y_balls: np.ndarray, n_sims: int, rng: np.random.Generator) -> TestResult:
    """Chi-cuadrado de uniformidad de las 5 balotas, en tres versiones.

    - Aproximada: X² con 42 gl, como si las balotas fueran independientes. No lo son (un sorteo
      no repite números), así que bajo H0 E[X²] = 43 - 5 = 38 y no 42: la prueba es conservadora.
    - Corregida: X² * 42/38 ~ χ²(42) asintóticamente (covarianza del muestreo sin reemplazo).
    - Monte Carlo: compara con historiales uniformes simulados del mismo tamaño (exacta).
    """
    n = len(y_balls)
    counts = y_balls.sum(axis=0)
    expected = n * P_BALL
    x2 = float(chi_square_statistic(counts, expected))
    df = N_BALLS - 1
    factor = (N_BALLS - 1) / (N_BALLS - BALLS_PER_DRAW)
    sims = simulate_ball_counts(n, n_sims, rng)
    x2_sims = chi_square_statistic(sims, expected)
    sd = math.sqrt(n * P_BALL * (1 - P_BALL))
    z = (counts - expected) / sd
    max_z_sims = (np.abs(sims - expected) / sd).max(axis=1)
    p_mc = mc_p_value(x2_sims, x2)
    hot, cold = int(np.argmax(counts)), int(np.argmin(counts))
    summary = {
        "n_sorteos": n,
        "frecuencia_esperada": expected,
        "conteos": counts.tolist(),
        "chi2": x2,
        "gl": df,
        "p_valor_aproximado": float(sps.chi2.sf(x2, df)),
        "factor_correccion": factor,
        "chi2_corregido": x2 * factor,
        "p_valor_corregido": float(sps.chi2.sf(x2 * factor, df)),
        "p_valor_monte_carlo": p_mc,
        "p_valor_monte_carlo_ic95": mc_p_value_interval(p_mc, n_sims),
        "n_simulaciones": n_sims,
        "chi2_nulo_media": float(x2_sims.mean()),
        "chi2_nulo_ic95": null_interval(x2_sims),
        "veredicto": verdict(p_mc),
        "z_por_balota": z.tolist(),
        "balota_mas_frecuente": {"balota": hot + 1, "conteo": int(counts[hot]), "z": float(z[hot])},
        "balota_menos_frecuente": {
            "balota": cold + 1,
            "conteo": int(counts[cold]),
            "z": float(z[cold]),
        },
        "max_abs_z": float(np.abs(z).max()),
        "p_valor_max_abs_z_monte_carlo": mc_p_value(max_z_sims, float(np.abs(z).max())),
    }
    return TestResult(summary, {"chi2": x2_sims, "max_abs_z": max_z_sims})


def uniformity_super(y_sb: np.ndarray, n_sims: int, rng: np.random.Generator) -> TestResult:
    """Chi-cuadrado de la superbalota (una por sorteo: la prueba clásica con 15 gl es válida)."""
    n = len(y_sb)
    counts = y_sb.sum(axis=0)
    expected = n * P_SUPER
    x2 = float(chi_square_statistic(counts, expected))
    df = N_SUPER - 1
    sims = rng.multinomial(n, [P_SUPER] * N_SUPER, size=n_sims)
    x2_sims = chi_square_statistic(sims, expected)
    p_mc = mc_p_value(x2_sims, x2)
    summary = {
        "n_sorteos": n,
        "frecuencia_esperada": expected,
        "conteos": counts.tolist(),
        "chi2": x2,
        "gl": df,
        "p_valor": float(sps.chi2.sf(x2, df)),
        "p_valor_monte_carlo": p_mc,
        "p_valor_monte_carlo_ic95": mc_p_value_interval(p_mc, n_sims),
        "n_simulaciones": n_sims,
        "chi2_nulo_media": float(x2_sims.mean()),
        "chi2_nulo_ic95": null_interval(x2_sims),
        "veredicto": verdict(p_mc),
    }
    return TestResult(summary, {"chi2": x2_sims})


# ------------------------------------------------------------------ independencia entre juegos


def common_balls(y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
    """Balotas en común por sorteo entre dos matrices multi-hot alineadas."""
    return np.logical_and(y_a, y_b).sum(axis=1)


def _bin_common(values: np.ndarray) -> np.ndarray:
    """Agrupa conteos de balotas en común en {0, 1, 2, >=3} por fila (último eje)."""
    v = np.minimum(values, 3)
    return np.stack([(v == k).sum(axis=-1) for k in range(4)], axis=-1)


def _standardize(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    return (y - y.mean(axis=0)) / y.std(axis=0)


def independence_between_games(
    a: GameData,
    b: GameData,
    n_sims: int,
    rng: np.random.Generator,
    n_perm_corr: int = 2000,
) -> TestResult:
    """¿El resultado de un juego dice algo del otro en la misma fecha? (se espera que no).

    - Balotas en común por sorteo frente a la hipergeométrica (5 de 43): media y distribución,
      con p-valores Monte Carlo bajo "dos sorteos justos e independientes".
    - Permutación: se barajan los sorteos de `b` entre fechas. Solo supone intercambiabilidad,
      no uniformidad, y conserva las frecuencias marginales de cada juego.
    - Superbalota igual en ambos: binomial(n, 1/16) y permutación.
    - Correlaciones entre indicadores (balota j de `a`, balota k de `b`): la mayor en valor
      absoluto, comparada con su distribución bajo permutación (corrige comparaciones múltiples).
    """
    common_n = np.intersect1d(a.n_sorteo, b.n_sorteo)
    ia = np.searchsorted(a.n_sorteo, common_n)
    ib = np.searchsorted(b.n_sorteo, common_n)
    ya, yb = a.y_balls[ia], b.y_balls[ib]
    sa, sbb = a.y_sb[ia].argmax(axis=1), b.y_sb[ib].argmax(axis=1)
    n = len(common_n)

    common = common_balls(ya, yb)
    obs_mean = float(common.mean())
    pmf = hits_pmf()
    expected_binned = n * np.array([pmf[0], pmf[1], pmf[2], pmf[3:].sum()])

    hyper = rng.hypergeometric(
        BALLS_PER_DRAW, N_BALLS - BALLS_PER_DRAW, BALLS_PER_DRAW, (n_sims, n)
    )
    mc_means = hyper.mean(axis=1)
    x2_obs = float(((_bin_common(common) - expected_binned) ** 2 / expected_binned).sum())
    x2_sims = ((_bin_common(hyper) - expected_binned) ** 2 / expected_binned).sum(axis=1)

    perm_means = np.empty(n_sims)
    perm_sb = np.empty(n_sims)
    for i in range(n_sims):
        perm = rng.permutation(n)
        perm_means[i] = common_balls(ya, yb[perm]).mean()
        perm_sb[i] = (sa == sbb[perm]).sum()

    matches_sb = int((sa == sbb).sum())
    p_binom = float(sps.binomtest(matches_sb, n, P_SUPER).pvalue)
    za, zb = _standardize(ya), _standardize(yb)
    corr = za.T @ zb / n
    max_corr = float(np.abs(corr).max())
    max_corr_perm = np.array(
        [np.abs(za.T @ zb[rng.permutation(n)] / n).max() for _ in range(n_perm_corr)]
    )

    p_mean_mc = mc_p_value(mc_means, obs_mean, "two-sided", center=EXPECTED_HITS)
    p_perm = mc_p_value(perm_means, obs_mean, "two-sided")
    summary = {
        "n_sorteos_comunes": n,
        "balotas_en_comun": {
            "media_observada": obs_mean,
            "media_esperada": EXPECTED_HITS,
            "distribucion_observada": {
                str(k): int(c) for k, c in enumerate(np.bincount(common, minlength=6))
            },
            "distribucion_esperada": {str(k): float(n * p) for k, p in enumerate(pmf)},
            "p_valor_media_monte_carlo": p_mean_mc,
            "media_nula_ic95": null_interval(mc_means),
            "chi2_agrupado_0_1_2_3mas": x2_obs,
            "p_valor_chi2_monte_carlo": mc_p_value(x2_sims, x2_obs),
            "p_valor_permutacion": p_perm,
            "media_permutacion_ic95": null_interval(perm_means),
            "veredicto": verdict(min(p_mean_mc, p_perm)),
        },
        "superbalota_igual": {
            "coincidencias": matches_sb,
            "esperadas": n * P_SUPER,
            "p_valor_binomial": p_binom,
            "p_valor_permutacion": mc_p_value(perm_sb, matches_sb, "two-sided"),
            "veredicto": verdict(p_binom),
        },
        "correlaciones_balotas": {
            "max_abs_correlacion": max_corr,
            "p_valor_max_permutacion": mc_p_value(max_corr_perm, max_corr),
            "max_abs_nula_ic95": null_interval(max_corr_perm),
            "n_permutaciones": n_perm_corr,
        },
        "n_simulaciones": n_sims,
    }
    arrays = {
        "balotas_en_comun": common,
        "media_hipergeometrica": mc_means,
        "media_permutacion": perm_means,
        "superbalota_permutacion": perm_sb,
        "correlaciones": corr,
        "max_abs_correlacion_permutacion": max_corr_perm,
    }
    return TestResult(summary, arrays)
