"""Registros en vivo, APPEND-ONLY: una predicción nunca se modifica ni se borra.

- predictions/predictions_log.csv: lo que el modelo dijo ANTES de cada sorteo.
- predictions/reconciliation_log.csv: esa predicción comparada con el resultado real.

La única operación de escritura es agregar filas al final. `is_append_only` permite verificar
en CI que el contenido anterior del archivo es un prefijo exacto del nuevo.
"""

from __future__ import annotations

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


def is_append_only(old: str, new: str) -> bool:
    """True si `new` conserva `old` intacto al principio (solo se agregaron filas)."""
    return new.startswith(old)
