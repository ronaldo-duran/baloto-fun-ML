"""Esquema común de sorteos y parseo de fechas escritas en español."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

import numpy as np
import pandas as pd

BALL_COLS: list[str] = ["b1", "b2", "b3", "b4", "b5"]
SUPER_COL = "superbalota"
COLUMNS: list[str] = ["fecha", "n_sorteo", "juego", *BALL_COLS, SUPER_COL]
KEY: list[str] = ["juego", "n_sorteo"]
INT_COLS: list[str] = ["n_sorteo", *BALL_COLS, SUPER_COL]

MESES: dict[str, int] = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
DIAS: dict[str, int] = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "domingo": 6,
}

_FECHA_RE = re.compile(r"^\s*(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s*$", re.IGNORECASE)


def strip_accents(text: str) -> str:
    """'Sábado' -> 'Sabado' (para comparar nombres de días y meses)."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def parse_fecha_es(text: str) -> date:
    """Convierte '23 de Mayo de 2026' en `date(2026, 5, 23)`."""
    m = _FECHA_RE.match(str(text))
    if m is None:
        raise ValueError(f"Fecha no reconocida: {text!r}")
    day, month_name, year = m.groups()
    month = MESES.get(strip_accents(month_name).lower())
    if month is None:
        raise ValueError(f"Mes no reconocido en {text!r}")
    return date(int(year), month, int(day))


def weekday_from_name(name: str) -> int | None:
    """'Sábado' -> 5; None si no es un día de la semana en español."""
    return DIAS.get(strip_accents(name).strip().lower())


def empty_draws() -> pd.DataFrame:
    """DataFrame vacío con el esquema común y sus tipos."""
    df = pd.DataFrame({c: pd.Series(dtype="int64") for c in COLUMNS})
    df["fecha"] = pd.Series(dtype="datetime64[ns]")
    df["juego"] = pd.Series(dtype=object)
    return df[COLUMNS]


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Copia en el esquema común: tipos fijos, balotas ordenadas por fila, filas por (juego, n).

    Ordenar las 5 balotas de cada fila es seguro porque el orden de extracción no importa:
    el resultado es un conjunto. Los repetidos se conservan para que la validación los detecte.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas del esquema común: {missing}")
    if df.empty:
        return empty_draws()
    out = df[COLUMNS].copy()
    out["fecha"] = pd.to_datetime(out["fecha"]).dt.normalize().astype("datetime64[ns]")
    out["juego"] = out["juego"].astype(str).str.strip().str.lower().astype(object)
    for col in INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
    out[BALL_COLS] = np.sort(out[BALL_COLS].to_numpy(), axis=1)
    return out.sort_values(KEY, kind="stable").reset_index(drop=True)


def concat_draws(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatena tablas de sorteos ignorando las vacías y devuelve el resultado canonizado."""
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return empty_draws()
    return canonicalize(pd.concat(non_empty, ignore_index=True))


def to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Versión serializable: fecha como texto ISO 'YYYY-MM-DD'."""
    out = canonicalize(df)
    out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
    return out


def read_canonical_csv(path) -> pd.DataFrame:
    """Lee un CSV ya en el esquema común (fecha ISO) y lo canoniza."""
    raw = pd.read_csv(path, dtype=str)
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: faltan columnas del esquema común: {missing}")
    if raw.empty:
        return empty_draws()
    raw["fecha"] = pd.to_datetime(raw["fecha"], format="%Y-%m-%d")
    return canonicalize(raw)
