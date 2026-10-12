"""Registros en vivo, APPEND-ONLY: una predicción nunca se modifica ni se borra.

- predictions/predictions_log.csv: lo que el modelo dijo ANTES de cada sorteo.
- predictions/reconciliation_log.csv: esa predicción comparada con el resultado real.

La operación normal de escritura es agregar filas al final. El esquema solo puede crecer con
columnas nuevas al final de cada fila (`add_columns`), sin tocar ninguna celda existente.
`is_append_only` permite verificar en CI que nada de lo anterior cambió.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from baloto_ml.config import N_BALLS, N_SUPER

BALL_PROB_COLS = [f"p_b{j:02d}" for j in range(1, N_BALLS + 1)]
SUPER_PROB_COLS = [f"p_sb{j:02d}" for j in range(1, N_SUPER + 1)]

PREDICTION_COLUMNS = [
    "prediccion_id",
    "creada_utc",
    "juego",
    "n_sorteo",
    "fecha_sorteo",
    "hora_limite_utc",
    "modelo_version",
    "modelo",
    "combinacion",
    "superbalota_sugerida",
    *BALL_PROB_COLS,
    *SUPER_PROB_COLS,
]
# Dónde cayó la combinación ganadora según el modelo ANTES del sorteo (agregadas en octubre de
# 2026; las filas anteriores se completaron con las probabilidades que ya estaban registradas).
RESULT_SCORE_COLUMNS = ["puntaje_resultado", "percentil_resultado"]
RECONCILIATION_COLUMNS = [
    "prediccion_id",
    "conciliada_utc",
    "juego",
    "n_sorteo",
    "fecha_sorteo",
    "fecha_real",
    "modelo_version",
    "combinacion",
    "resultado",
    "superbalota_sugerida",
    "superbalota_real",
    "aciertos",
    "acierto_superbalota",
    "aciertos_esperados",
    "log_loss_balotas",
    "log_loss_constante",
    "log_loss_superbalota",
    "log_loss_superbalota_constante",
    *RESULT_SCORE_COLUMNS,
]


def prediction_id(juego: str, n_sorteo: int) -> str:
    """Una sola predicción por sorteo y juego: la primera registrada es la que vale."""
    return f"{juego}-{int(n_sorteo)}"


def format_combination(numbers: Sequence[int]) -> str:
    return "-".join(f"{int(n):02d}" for n in sorted(numbers))


def parse_combination(text: str) -> list[int]:
    return [int(x) for x in str(text).split("-")]


def read_log(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"combinacion": str, "resultado": str})
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}")
    return df


def append_rows(path: Path, rows: Sequence[dict], columns: list[str]) -> None:
    """Agrega filas al final (crea el archivo con encabezado si no existe). Nunca reescribe."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    text = pd.DataFrame(list(rows), columns=columns).to_csv(
        index=False, header=new_file, lineterminator="\n"
    )
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(text)


def add_columns(path: Path, columns: Sequence[str], values: Sequence[Sequence]) -> None:
    """Agrega columnas AL FINAL de un registro existente: cada fila conserva sus celdas tal
    cual (byte a byte) y gana las nuevas. Es la única forma permitida de cambiar el esquema."""
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n").split("\n")
    if len(lines) - 1 != len(values):
        raise ValueError(f"{path}: {len(lines) - 1} filas y {len(values)} juegos de valores")
    out = [lines[0] + "," + ",".join(columns)]
    for line, vals in zip(lines[1:], values, strict=True):
        if len(vals) != len(columns):
            raise ValueError(f"{path}: se esperaban {len(columns)} valores por fila")
        out.append(line + "," + ",".join(str(v) for v in vals))
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="")


def is_append_only(old: str, new: str) -> bool:
    """True si `new` conserva `old` intacto: solo se agregaron filas al final y, si acaso,
    columnas al final de cada fila. Ninguna celda existente cambió ni desapareció."""
    if new.startswith(old):
        return True
    old_rows = list(csv.reader(io.StringIO(old)))
    new_rows = list(csv.reader(io.StringIO(new)))
    if not old_rows or len(new_rows) < len(old_rows):
        return False
    width, new_width = len(old_rows[0]), len(new_rows[0])
    if new_width <= width:  # sin columnas nuevas solo vale el prefijo exacto
        return False
    return all(
        n[:width] == o and len(n) == new_width for o, n in zip(old_rows, new_rows, strict=False)
    )
