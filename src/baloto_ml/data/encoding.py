"""Codificación de sorteos: multi-hot (43) para las balotas y one-hot (16) para la superbalota.

Los números son etiquetas nominales, no cantidades: cada número es una columna independiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from baloto_ml.config import N_BALLS, N_SUPER
from baloto_ml.data.schema import BALL_COLS, SUPER_COL


def multi_hot(balls: ArrayLike, n_numbers: int = N_BALLS) -> np.ndarray:
    """(n, k) con números 1..n_numbers -> matriz (n, n_numbers) de 0/1 con k unos por fila."""
    arr = np.asarray(balls, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba una matriz (n, k); llegó forma {arr.shape}")
    if arr.size and (arr.min() < 1 or arr.max() > n_numbers):
        raise ValueError(f"Números fuera del rango 1..{n_numbers}")
    out = np.zeros((arr.shape[0], n_numbers), dtype=np.uint8)
    rows = np.repeat(np.arange(arr.shape[0]), arr.shape[1])
    out[rows, arr.ravel() - 1] = 1
    if (out.sum(axis=1) != arr.shape[1]).any():
        raise ValueError("Hay números repetidos dentro de una fila")
    return out


def one_hot(values: ArrayLike, n_numbers: int = N_SUPER) -> np.ndarray:
    """(n,) con números 1..n_numbers -> matriz (n, n_numbers) con un 1 por fila."""
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"Se esperaba un vector (n,); llegó forma {arr.shape}")
    return multi_hot(arr[:, None], n_numbers)


def decode_multi_hot(matrix: ArrayLike) -> np.ndarray:
    """Inversa de `multi_hot`: devuelve los números (1-based) de cada fila, ordenados."""
    mat = np.asarray(matrix)
    counts = mat.sum(axis=1)
    if mat.shape[0] and (counts != counts[0]).any():
        raise ValueError("Todas las filas deben tener la misma cantidad de unos")
    k = int(counts[0]) if mat.shape[0] else 0
    return (np.nonzero(mat)[1].reshape(mat.shape[0], k) + 1).astype(np.int64)


def decode_one_hot(matrix: ArrayLike) -> np.ndarray:
    """Inversa de `one_hot`."""
    return decode_multi_hot(matrix)[:, 0]


def encode_draws(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sorteos en el esquema común -> (Y_balotas (n, 43), Y_superbalota (n, 16))."""
    return multi_hot(df[BALL_COLS].to_numpy()), one_hot(df[SUPER_COL].to_numpy())
