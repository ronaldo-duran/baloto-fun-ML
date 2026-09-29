"""⚠️ LO QUE NO SE DEBE HACER: validación incorrecta, con fines didácticos.

Nada de este módulo se usa en el pipeline. Existe para el notebook 02, que muestra cómo dos
errores clásicos fabrican un modelo que "predice la lotería":

1. Fuga de información: features con una ventana que INCLUYE el sorteo que se quiere predecir
   (el típico `rolling(w).mean()` sin `.shift(1)`), evaluadas con un split aleatorio.
2. Sesgo de selección: probar muchas estrategias y quedarse con la que mejor salió en los
   mismos datos con que se la mide.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from baloto_ml.config import EVAL, EXPECTED_HITS, N_BALLS, N_SUPER, P_SUPER, SEED, EvalConfig
from baloto_ml.evaluation.metrics import evaluate_probabilities, top_k
from baloto_ml.evaluation.stats import hits_variance, rng_for
from baloto_ml.evaluation.walk_forward import walk_forward
from baloto_ml.features.build import window_frequency
from baloto_ml.features.history import past_counts
from baloto_ml.models.base import Forecaster, GameData, Probabilities

TRAP_WINDOWS: tuple[int, ...] = (10, 30, 100)


def rolling_frequency(y: np.ndarray, include_current: bool) -> np.ndarray:
    """(n, k, 3) frecuencias en ventanas de 10/30/100 sorteos.

    `include_current=True` es EL ERROR: la ventana termina en el propio sorteo t, así que la
    feature ya "sabe" si el número salió. Es lo que hace `serie.rolling(w).mean()` sin
    `.shift(1)`. Con `include_current=False` la ventana termina en t-1 (correcto).
    """
    t = np.arange(len(y)) + (1 if include_current else 0)
    counts = past_counts(y)
    return np.stack([window_frequency(counts, t, w, 0) for w in TRAP_WINDOWS], axis=-1)


class RollingLogistic(Forecaster):
    """Logística por balota solo con frecuencias móviles (la superbalota queda uniforme)."""

    name = "logistica_ventanas"

    def __init__(self, leaky: bool) -> None:
        self.leaky = leaky

    def _x(self, data: GameData) -> np.ndarray:
        return rolling_frequency(data.y_balls, include_current=self.leaky)

    def fit(self, data: GameData, train_idx: np.ndarray) -> RollingLogistic:
        x = self._x(data)
        self.models_ = []
        for j in range(N_BALLS):
            scaler = StandardScaler().fit(x[train_idx, j])
            model = LogisticRegression(C=1.0, max_iter=1000)
            model.fit(scaler.transform(x[train_idx, j]), data.y_balls[train_idx, j])
            self.models_.append((scaler, model))
        return self

    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        x = self._x(data)
        balls = np.column_stack(
            [m.predict_proba(s.transform(x[idx, j]))[:, 1] for j, (s, m) in enumerate(self.models_)]
        )
        return Probabilities(balls, np.full((len(idx), N_SUPER), P_SUPER))


def _summary(probs: Probabilities, data: GameData, idx: np.ndarray, key: str) -> dict:
    m = evaluate_probabilities(probs, data.y_balls[idx], data.y_sb[idx], rng_for("trampa", key))
    b = m["balotas"]
    return {
        "n_sorteos_prueba": len(idx),
        "aciertos_top5": b["aciertos_top5_media"],
        "aciertos_top5_ic95": b["aciertos_top5_ic95"],
        "log_loss": b["log_loss"],
        "skill_log_loss": b["skill_log_loss"],
    }


def trap_grid(
    data: GameData, cfg: EvalConfig = EVAL, test_size: float = 0.25, seed: int = SEED
) -> list[dict]:
    """Las 4 combinaciones de {features con fuga, correctas} x {split aleatorio, walk-forward}."""
    rows = []
    targets = np.arange(cfg.warmup, len(data))
    train, test = train_test_split(targets, test_size=test_size, shuffle=True, random_state=seed)
    train, test = np.sort(train), np.sort(test)
    for leaky in (True, False):
        label = "con fuga (ventana incluye el sorteo)" if leaky else "correctas (solo pasado)"
        model = RollingLogistic(leaky).fit(data, train)
        rows.append(
            {"features": label, "validacion": "split aleatorio"}
            | _summary(model.predict(data, test), data, test, f"aleatorio-{leaky}")
        )
        wf = walk_forward(partial(RollingLogistic, leaky), data, cfg)
        rows.append(
            {"features": label, "validacion": "walk-forward"}
            | _summary(wf.probs, data, wf.test_idx, f"wf-{leaky}")
        )
    return rows


def hot_cold_hits(
    y: np.ndarray, idx: np.ndarray, window: int, hot: bool, rng: np.random.Generator
) -> np.ndarray:
    """Aciertos por sorteo de la estrategia 'jugar los 5 más (o menos) frecuentes en la ventana'."""
    freq = window_frequency(past_counts(y), idx, window, 0)
    picks = top_k(freq if hot else -freq, 5, rng)
    return np.take_along_axis(y[idx], picks, axis=1).sum(axis=1)


def selection_bias_demo(
    data: GameData, cfg: EvalConfig = EVAL, windows: range = range(5, 205, 5)
) -> dict:
    """80 estrategias de números 'calientes' y 'fríos': se elige la mejor en la primera mitad
    del periodo de prueba (A) y se la vuelve a medir en la segunda (B), que no vio."""
    test = np.arange(cfg.first_test_index, len(data))
    half = len(test) // 2
    periods = {"A": test[:half], "B": test[half:]}
    rows = []
    for w in windows:
        for hot in (True, False):
            row = {
                "estrategia": f"{'calientes' if hot else 'frios'}_{w}",
                "ventana": w,
                "tipo": "calientes" if hot else "frios",
            }
            for name, idx in periods.items():
                rng = rng_for("trampa-estrategias", w, int(hot), name)
                row[f"aciertos_{name}"] = float(
                    hot_cold_hits(data.y_balls, idx, w, hot, rng).mean()
                )
            rows.append(row)
    best = max(rows, key=lambda r: r["aciertos_A"])
    bands = {}
    for name, idx in periods.items():
        sd = float(np.sqrt(hits_variance() / len(idx)))
        bands[name] = [EXPECTED_HITS - 1.96 * sd, EXPECTED_HITS + 1.96 * sd]
    return {
        "estrategias": rows,
        "mejor_en_A": best,
        "periodos": {
            name: {
                "primer_sorteo": int(data.n_sorteo[idx[0]]),
                "ultimo_sorteo": int(data.n_sorteo[idx[-1]]),
                "n_sorteos": len(idx),
            }
            for name, idx in periods.items()
        },
        "banda95_azar": bands,
    }
