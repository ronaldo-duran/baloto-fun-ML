"""Contratos comunes: los datos de un juego, las probabilidades y la interfaz de los modelos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baloto_ml.config import N_BALLS, N_SUPER
from baloto_ml.data.encoding import encode_draws


@dataclass(frozen=True)
class GameData:
    """Historial codificado de UN juego, ordenado por número de sorteo."""

    juego: str
    fechas: np.ndarray  # (n,) datetime64
    n_sorteo: np.ndarray  # (n,)
    y_balls: np.ndarray  # (n, 43) multi-hot
    y_sb: np.ndarray  # (n, 16) one-hot

    @classmethod
    def from_draws(cls, draws: pd.DataFrame, juego: str) -> GameData:
        g = draws[draws["juego"] == juego].sort_values("n_sorteo")
        if g.empty:
            raise ValueError(f"No hay sorteos de {juego}")
        y_balls, y_sb = encode_draws(g)
        return cls(juego, g["fecha"].to_numpy(), g["n_sorteo"].to_numpy(), y_balls, y_sb)

    def __len__(self) -> int:
        return len(self.n_sorteo)

    def head(self, n: int) -> GameData:
        """Los primeros `n` sorteos (útil para comprobar que no hay fuga de información)."""
        return GameData(
            self.juego, self.fechas[:n], self.n_sorteo[:n], self.y_balls[:n], self.y_sb[:n]
        )

    @property
    def weekday(self) -> np.ndarray:
        return pd.DatetimeIndex(self.fechas).dayofweek.to_numpy()


@dataclass(frozen=True)
class Probabilities:
    """Probabilidades por sorteo objetivo: cada balota (m, 43) y la superbalota (m, 16)."""

    balls: np.ndarray
    sb: np.ndarray

    def __post_init__(self) -> None:
        if self.balls.ndim != 2 or self.balls.shape[1] != N_BALLS:
            raise ValueError(f"balls debe ser (m, {N_BALLS}); llegó {self.balls.shape}")
        if self.sb.shape != (self.balls.shape[0], N_SUPER):
            raise ValueError(f"sb debe ser (m, {N_SUPER}); llegó {self.sb.shape}")


class Forecaster(ABC):
    """Modelo probabilístico. Para cada sorteo objetivo t solo puede usar sorteos anteriores a t."""

    name: str = "base"

    def fit(self, data: GameData, train_idx: np.ndarray) -> Forecaster:
        """Entrena con los sorteos objetivo `train_idx` (por defecto no entrena nada)."""
        return self

    @abstractmethod
    def predict(self, data: GameData, idx: np.ndarray) -> Probabilities:
        """Probabilidades para los sorteos `idx`, usando solo información previa a cada uno."""
