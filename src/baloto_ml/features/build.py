"""Features por número calculadas SOLO con sorteos anteriores al objetivo.

Para el sorteo objetivo t, cada número j (balota 1..43 o superbalota 1..16) recibe:

- `freq_10`, `freq_30`, `freq_100`: fracción de los últimos 10/30/100 sorteos en que salió;
- `sorteos_desde_ultima`: sorteos transcurridos desde su última aparición (1 = salió en el
  anterior; tope en 100);
- `tendencia_15v15`: frecuencia en los últimos 15 sorteos menos la de los 15 anteriores;

y todos comparten el día de la semana del sorteo objetivo (lunes, miércoles o sábado), que se
conoce antes del sorteo. Los números nunca se usan como cantidades: cada uno es una etiqueta.

Garantía anti-fuga (verificada en tests): la fila t depende solo de los sorteos 0..t-1 y de la
fecha de t. Por eso las features de un sorteo no cambian si se modifica el futuro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baloto_ml.features.history import past_counts

WINDOWS: tuple[int, ...] = (10, 30, 100)
TREND_HALF = 15
GAP_CAP = 100
WEEKDAYS: tuple[int, ...] = (0, 2, 5)  # lunes, miércoles, sábado
PER_NUMBER_NAMES: tuple[str, ...] = (
    *(f"freq_{w}" for w in WINDOWS),
    "sorteos_desde_ultima",
    "tendencia_15v15",
)
SHARED_NAMES: tuple[str, ...] = ("dia_lunes", "dia_miercoles", "dia_sabado")


def window_frequency(counts: np.ndarray, t: np.ndarray, start_lag: int, end_lag: int) -> np.ndarray:
    """Frecuencia de cada número en los sorteos [t - start_lag, t - end_lag), recortada en 0.

    `counts` viene de `past_counts` (fila t = conteos de los sorteos 0..t-1).
    """
    hi = np.maximum(t - end_lag, 0)
    lo = np.maximum(t - start_lag, 0)
    width = np.maximum(hi - lo, 1)
    return (counts[hi] - counts[lo]) / width[:, None]


def draws_since_last(y: np.ndarray, t: np.ndarray, cap: int = GAP_CAP) -> np.ndarray:
    """Sorteos desde la última aparición antes de t (1 = salió en t-1; t+1 si nunca salió)."""
    n, k = y.shape
    idx = np.arange(n)[:, None]
    last = np.maximum.accumulate(np.where(y > 0, idx, -1), axis=0)  # último índice <= s
    last_before = np.vstack([np.full((1, k), -1), last])  # fila t: último índice <= t-1
    return np.minimum(t[:, None] - last_before[t], cap)


def number_features(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(len(t), k, 5) features de cada número para los objetivos `t` (0 <= t <= n)."""
    counts = past_counts(y)
    feats = [window_frequency(counts, t, w, 0) for w in WINDOWS]
    feats.append(draws_since_last(y, t))
    recent = window_frequency(counts, t, TREND_HALF, 0)
    older = window_frequency(counts, t, 2 * TREND_HALF, TREND_HALF)
    feats.append(recent - older)
    return np.stack(feats, axis=-1).astype(float)


def weekday_one_hot(weekdays: np.ndarray) -> np.ndarray:
    return (np.asarray(weekdays)[:, None] == np.array(WEEKDAYS)[None, :]).astype(float)


@dataclass(frozen=True)
class NumberFeatures:
    """Features de un bombo: por número (m, k, F) y compartidas por sorteo (m, S)."""

    per_number: np.ndarray
    shared: np.ndarray

    @property
    def n_numbers(self) -> int:
        return self.per_number.shape[1]

    def design(self, idx: np.ndarray, number: int) -> np.ndarray:
        """Matriz (len(idx), F + S) del modelo propio del número `number` (0-based)."""
        return np.hstack([self.per_number[idx, number, :], self.shared[idx]])

    def long_design(self, idx: np.ndarray) -> np.ndarray:
        """Formato largo (len(idx) * k, F + S): una fila por (sorteo, número), sin su identidad."""
        m, k, f = len(idx), self.per_number.shape[1], self.per_number.shape[2]
        return np.hstack(
            [self.per_number[idx].reshape(m * k, f), np.repeat(self.shared[idx], k, axis=0)]
        )


@dataclass(frozen=True)
class FeatureSet:
    balls: NumberFeatures
    sb: NumberFeatures


def build_features(
    y_balls: np.ndarray, y_sb: np.ndarray, target_weekdays: np.ndarray
) -> FeatureSet:
    """Features de los objetivos t = 0..m-1, con m = len(target_weekdays) <= n + 1.

    La fila n (si se pide) es el sorteo siguiente al último conocido: su día de la semana sale
    del calendario y sus features del historial completo.
    """
    m = len(target_weekdays)
    if m > len(y_balls) + 1:
        raise ValueError("Solo se puede construir hasta el sorteo siguiente al último conocido")
    t = np.arange(m)
    shared = weekday_one_hot(target_weekdays)
    return FeatureSet(
        NumberFeatures(number_features(y_balls, t), shared),
        NumberFeatures(number_features(y_sb, t), shared),
    )
