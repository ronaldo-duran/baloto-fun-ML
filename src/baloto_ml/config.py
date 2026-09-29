"""Constantes del juego y rutas del proyecto."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Literal

Juego = Literal["baloto", "revancha"]
JUEGOS: tuple[Juego, ...] = ("baloto", "revancha")

N_BALLS = 43  # balotas principales: 1..43
BALLS_PER_DRAW = 5  # 5 balotas distintas por sorteo (un conjunto: el orden no importa)
N_SUPER = 16  # superbalota: 1..16, en otro bombo

P_BALL = BALLS_PER_DRAW / N_BALLS  # probabilidad de que salga una balota dada: 5/43
P_SUPER = 1 / N_SUPER  # probabilidad de acertar la superbalota: 1/16
EXPECTED_HITS = BALLS_PER_DRAW * P_BALL  # aciertos esperados por azar con 5 números: 25/43 ≈ 0.581
JACKPOT_ODDS = comb(N_BALLS, BALLS_PER_DRAW) * N_SUPER  # 1 en 15.401.568

SEED = 42
N_SIMS = 10_000  # simulaciones Monte Carlo por defecto


@dataclass(frozen=True)
class EvalConfig:
    """Ventanas del walk-forward (expanding window), fijadas antes de ver resultados."""

    warmup: int = 100  # primeros sorteos: solo historia (llenan la ventana de 100 de las features)
    min_train: int = 150  # sorteos objetivo del primer entrenamiento
    refit_every: int = 10  # se reentrena cada k sorteos; las features se actualizan en cada uno

    @property
    def first_test_index(self) -> int:
        return self.warmup + self.min_train


EVAL = EvalConfig()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILENAMES: dict[Juego, str] = {
    "baloto": "resultados_baloto.csv",
    "revancha": "resultados_revancha.csv",
}


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto bajo una raíz (inyectable para tests y CI)."""

    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def incoming(self) -> Path:
        return self.data / "incoming"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def processed_draws(self) -> Path:
        return self.processed / "draws.csv"

    @property
    def new_draws(self) -> Path:
        """Sorteos nuevos detectados por `ingest`, pendientes de `validate`."""
        return self.interim / "new_draws.csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def validation_report(self) -> Path:
        return self.reports / "validation.json"

    @property
    def analysis_report(self) -> Path:
        """Uniformidad por juego e independencia Baloto/Revancha."""
        return self.reports / "analysis.json"

    def evaluation_report(self, juego: Juego) -> Path:
        return self.reports / "evaluation" / f"{juego}.json"

    def significance_report(self, juego: Juego) -> Path:
        """Permutación e historiales sintéticos con reentrenamiento (etapa lenta)."""
        return self.reports / "significance" / f"{juego}.json"

    @property
    def controls_report(self) -> Path:
        """Control positivo y sensibilidad a la regularización (solo simulaciones)."""
        return self.reports / "significance" / "controles.json"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def raw_file(self, juego: Juego) -> Path:
        return self.raw / RAW_FILENAMES[juego]


def default_paths() -> Paths:
    """Raíz tomada de `BALOTO_ML_ROOT` o, por defecto, la del repositorio."""
    env = os.environ.get("BALOTO_ML_ROOT")
    return Paths(Path(env).resolve() if env else PROJECT_ROOT)
