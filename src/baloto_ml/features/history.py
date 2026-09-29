"""Agregados del historial que solo miran hacia atrás."""

from __future__ import annotations

import numpy as np


def past_counts(y: np.ndarray) -> np.ndarray:
    """Conteos acumulados SIN incluir el sorteo actual: la fila t suma los sorteos 0..t-1.

    Devuelve (n + 1, k): la fila 0 es cero y la fila n cuenta todo el historial, lo que permite
    consultar también el sorteo siguiente al último conocido.
    """
    out = np.zeros((y.shape[0] + 1, y.shape[1]), dtype=np.int64)
    np.cumsum(y, axis=0, out=out[1:])
    return out
