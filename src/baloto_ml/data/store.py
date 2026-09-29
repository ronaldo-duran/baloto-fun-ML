"""Lectura y escritura deterministas de sorteos y reportes (LF, UTF-8, orden estable)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baloto_ml.data.schema import empty_draws, read_canonical_csv, to_csv_frame


def draws_to_csv_text(df: pd.DataFrame) -> str:
    """Serialización canónica: misma tabla -> mismo texto en cualquier sistema operativo."""
    return to_csv_frame(df).to_csv(index=False, lineterminator="\n")


def draws_hash(df: pd.DataFrame) -> str:
    """SHA-256 del contenido canónico (independiente de fin de línea y orden de origen)."""
    return hashlib.sha256(draws_to_csv_text(df).encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def read_draws(path: Path) -> pd.DataFrame:
    """Sorteos en el esquema común; tabla vacía si el archivo no existe."""
    return read_canonical_csv(path) if path.exists() else empty_draws()


def write_draws(df: pd.DataFrame, path: Path) -> None:
    write_text(path, draws_to_csv_text(df))


def write_incoming(df: pd.DataFrame, incoming_dir: Path, prefix: str) -> Path | None:
    """Guarda sorteos en data/incoming/ como `<prefijo>_<n_min>-<n_max>_<hash>.csv`."""
    if df.empty:
        return None
    text = draws_to_csv_text(df)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    path = incoming_dir / f"{prefix}_{df['n_sorteo'].min()}-{df['n_sorteo'].max()}_{digest}.csv"
    write_text(path, text)
    return path


def known_last(df: pd.DataFrame) -> dict[str, int]:
    """Último `n_sorteo` conocido por juego."""
    return {str(j): int(g["n_sorteo"].max()) for j, g in df.groupby("juego")}


def to_jsonable(obj: Any, sig_digits: int = 6) -> Any:
    """Convierte tipos de numpy y redondea floats a cifras significativas.

    El redondeo evita que diferencias de 1e-16 entre sistemas operativos cambien los reportes
    versionados cuando los datos no cambiaron (el pipeline debe ser idempotente).
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, sig_digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v, sig_digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), sig_digits)
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return 0.0 if abs(x) < 1e-12 else float(f"{x:.{sig_digits}g}")
    return obj


def write_json(obj: Any, path: Path) -> None:
    write_text(path, json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
