from __future__ import annotations

import math

import numpy as np
import pandas as pd

from baloto_ml.config import EXPECTED_HITS
from baloto_ml.data.encoding import multi_hot
from baloto_ml.data.synthetic import synthetic_history
from baloto_ml.evaluation.stats import (
    chi_square_statistic,
    common_balls,
    hits_pmf,
    hits_variance,
    independence_between_games,
    mc_p_value,
    mc_p_value_interval,
    simulate_ball_counts,
    uniformity_balls,
    uniformity_super,
)
from baloto_ml.models.base import GameData


def test_hits_pmf() -> None:
    pmf = hits_pmf()
    k = np.arange(6)
    assert math.isclose(pmf.sum(), 1.0)
    assert math.isclose((k * pmf).sum(), EXPECTED_HITS)
    assert math.isclose(((k - EXPECTED_HITS) ** 2 * pmf).sum(), hits_variance())
    assert math.isclose(pmf[5], 1 / 962_598)  # 1 / C(43, 5)


def test_mc_p_value() -> None:
    sims = np.arange(100.0)
    assert mc_p_value(sims, 1000.0) == 1 / 101  # nunca 0
    assert mc_p_value(sims, -1.0) == 1.0
    assert mc_p_value(sims, 49.5, "two-sided", center=49.5) == 1.0
    lo, hi = mc_p_value_interval(0.2, 10_000)
    assert lo < 0.2 < hi and hi - lo < 0.02


def test_simulated_counts_and_chi_square_correction() -> None:
    """Bajo H0, E[X²] = 43 - 5 = 38 (no 42): justifica el factor 42/38."""
    n_draws = 200
    sims = simulate_ball_counts(n_draws, 4000, np.random.default_rng(0))
    assert (sims.sum(axis=1) == 5 * n_draws).all()
    x2 = chi_square_statistic(sims, n_draws * 5 / 43)
    assert abs(x2.mean() - 38) < 0.7
    assert abs((x2 * 42 / 38).mean() - 42) < 0.8


def test_uniformity_accepts_fair_and_flags_biased(rng: np.random.Generator) -> None:
    fair = GameData.from_draws(synthetic_history(400, rng, juegos=("baloto",)), "baloto")
    res = uniformity_balls(fair.y_balls, 1000, np.random.default_rng(1)).summary
    assert res["p_valor_monte_carlo"] > 0.01
    assert abs(res["chi2_nulo_media"] - 38) < 1.5

    biased = fair.y_balls.copy()
    rows = np.flatnonzero(biased[:, 0] == 0)[:60]  # la balota 1 sale 60 veces más
    for r in rows:
        drop = np.flatnonzero(biased[r])[-1]
        biased[r, drop], biased[r, 0] = 0, 1
    res = uniformity_balls(biased, 1000, np.random.default_rng(1)).summary
    assert res["p_valor_monte_carlo"] < 0.01
    assert res["balota_mas_frecuente"]["balota"] == 1

    sb = uniformity_super(fair.y_sb, 1000, np.random.default_rng(2)).summary
    assert sb["p_valor_monte_carlo"] > 0.01 and sb["gl"] == 15


def test_independence(rng: np.random.Generator) -> None:
    h = synthetic_history(300, rng)
    a, b = GameData.from_draws(h, "baloto"), GameData.from_draws(h, "revancha")
    res = independence_between_games(a, b, 500, np.random.default_rng(3), n_perm_corr=200).summary
    assert res["n_sorteos_comunes"] == 300
    assert res["balotas_en_comun"]["p_valor_permutacion"] > 0.01
    assert res["superbalota_igual"]["p_valor_binomial"] > 0.01

    # Revancha "copiada" de Baloto en la mitad de los sorteos: debe detectarse.
    rev = h[h["juego"] == "revancha"].copy()
    bal = h[h["juego"] == "baloto"]
    cols = ["b1", "b2", "b3", "b4", "b5", "superbalota"]
    rev.iloc[::2, rev.columns.get_indexer(cols)] = bal.iloc[::2][cols].to_numpy()
    b2 = GameData.from_draws(pd.concat([bal, rev]), "revancha")
    res = independence_between_games(a, b2, 500, np.random.default_rng(3), n_perm_corr=200).summary
    assert res["balotas_en_comun"]["p_valor_permutacion"] < 0.01
    assert res["superbalota_igual"]["p_valor_binomial"] < 0.01
    assert res["balotas_en_comun"]["media_observada"] > 2


def test_multi_hot_common_balls_sanity() -> None:
    a = multi_hot([[1, 2, 3, 4, 5]])
    b = multi_hot([[4, 5, 6, 7, 8]])
    assert common_balls(a, b).tolist() == [2]
