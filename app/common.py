"""Utilidades compartidas por las páginas de la app: carga cacheada, formato y gráficos."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # Streamlit Cloud ejecuta desde el repo sin instalarlo
    sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from baloto_ml.config import BALLS_PER_DRAW, N_BALLS, N_SUPER, Paths  # noqa: E402
from baloto_ml.data.encoding import multi_hot  # noqa: E402
from baloto_ml.data.schema import BALL_COLS  # noqa: E402
from baloto_ml.data.store import read_draws, read_json  # noqa: E402
from baloto_ml.evaluation.significance import combination_scores  # noqa: E402
from baloto_ml.live.log import PREDICTION_COLUMNS, RECONCILIATION_COLUMNS, read_log  # noqa: E402
from baloto_ml.live.predict import NextDraw, next_draw_forecast  # noqa: E402
from baloto_ml.models.registry import ModelRegistry  # noqa: E402

PATHS = Paths(Path(os.environ.get("BALOTO_ML_ROOT") or ROOT))  # la variable permite probarla
DISCLAIMER = (
    "**Experimento educativo.** El Baloto es aleatorio; este modelo no predice resultados. "
    "Juega con responsabilidad."
)
REPO_URL = "https://github.com/ronaldo-duran/baloto-fun-ML"
NOMBRE = {"baloto": "Baloto", "revancha": "Baloto Revancha"}
NOMBRE_MODELO = {"logistica": "regresión logística", "gradient_boosting": "gradient boosting"}

# Paleta validada (legible en tema claro y oscuro). El texto de los gráficos usa el tema.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"


def es(x: float, nd: int = 3) -> str:
    """Número con coma decimal: 0.581 -> '0,581'."""
    return f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def es_int(n: int) -> str:
    """Entero con punto de miles: 15401568 -> '15.401.568'."""
    return f"{int(n):,}".replace(",", ".")


# ------------------------------------------------------------------------------ carga cacheada


@st.cache_data(ttl=3600, show_spinner=False)
def load_draws() -> pd.DataFrame:
    return read_draws(PATHS.processed_draws)


@st.cache_data(ttl=3600, show_spinner=False)
def load_report(relative: str) -> dict | None:
    path = PATHS.root / relative
    return read_json(path) if path.exists() else None


@st.cache_data(ttl=600, show_spinner=False)
def load_logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_log(PATHS.predictions_log, PREDICTION_COLUMNS),
        read_log(PATHS.reconciliation_log, RECONCILIATION_COLUMNS),
    )


def latest_version(juego: str) -> str | None:
    return ModelRegistry(PATHS.models).latest_version(juego)


@st.cache_resource(ttl=3600, show_spinner="Cargando el modelo...")
def load_forecast(juego: str, version: str | None) -> NextDraw:
    """Predicción del modelo vigente para el próximo sorteo (la versión invalida el caché)."""
    return next_draw_forecast(PATHS, juego)


# ------------------------------------------------------------------------------ cálculos


def play_score(nd: NextDraw, balls: list[int], sb: int) -> float:
    """Puntaje del modelo para una jugada: suma de log(p / p_azar) de sus 5 + 1 números."""
    sw, ss = combination_scores(nd.probs.balls, nd.probs.sb)
    return float(sw[0, np.array(balls) - 1].sum() + ss[0, sb - 1])


@st.cache_data(show_spinner=False)
def random_play_scores(_nd: NextDraw, key: tuple, n: int = 10_000, seed: int = 42) -> np.ndarray:
    """Puntajes de `n` jugadas al azar con las mismas probabilidades (key: juego y versión)."""
    rng = np.random.default_rng(seed)
    sw, ss = combination_scores(_nd.probs.balls, _nd.probs.sb)
    picked = np.argpartition(rng.random((n, N_BALLS)), BALLS_PER_DRAW, axis=1)[:, :BALLS_PER_DRAW]
    return sw[0][picked].sum(axis=1) + ss[0][rng.integers(0, N_SUPER, n)]


def backtest(draws: pd.DataFrame, juego: str, balls: list[int], sb: int) -> pd.DataFrame:
    """Aciertos que habría tenido la jugada en cada sorteo histórico del juego."""
    g = draws[draws["juego"] == juego].sort_values("n_sorteo")
    mine = np.zeros(N_BALLS, dtype=np.uint8)
    mine[np.array(balls) - 1] = 1
    hits = (multi_hot(g[BALL_COLS].to_numpy()) * mine).sum(axis=1)
    return pd.DataFrame(
        {
            "n_sorteo": g["n_sorteo"].to_numpy(),
            "fecha": g["fecha"].to_numpy(),
            "aciertos": hits,
            "superbalota": (g["superbalota"].to_numpy() == sb),
        }
    )


# ------------------------------------------------------------------------------ gráficos


def histogram_with_marker(values: np.ndarray, marker: float, x_title: str, marker_label: str):
    """Histograma de una distribución por azar con una regla en el valor del usuario."""
    counts, edges = np.histogram(values, bins=40)
    gap = 0.08 * (edges[1] - edges[0])  # separación entre barras vecinas
    df = pd.DataFrame(
        {"desde": edges[:-1] + gap, "hasta": edges[1:] - gap, "jugadas": counts, "cero": 0}
    )
    bars = (
        alt.Chart(df)
        .mark_rect(color=BLUE, opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("desde:Q", title=x_title, axis=alt.Axis(tickCount=8)),
            x2="hasta:Q",
            y=alt.Y("jugadas:Q", title="Jugadas al azar"),
            y2="cero:Q",
            tooltip=[
                alt.Tooltip("desde:Q", format=".3f", title="desde"),
                alt.Tooltip("hasta:Q", format=".3f", title="hasta"),
                alt.Tooltip("jugadas:Q", title="jugadas"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [marker], "etiqueta": [marker_label]}))
        .mark_rule(color=ORANGE, strokeWidth=3)
        .encode(x="x:Q", tooltip=[alt.Tooltip("etiqueta:N", title="")])
    )
    return (bars + rule).properties(height=260)


def observed_vs_expected(df: pd.DataFrame, x: str, x_title: str, y_title: str):
    """Barras observadas y marcas de lo esperado por azar (df: x, observado, esperado)."""
    base = alt.Chart(df).encode(x=alt.X(f"{x}:O", title=x_title, axis=alt.Axis(labelAngle=0)))
    tooltip = [
        alt.Tooltip(f"{x}:O", title=x_title),
        alt.Tooltip("observado:Q", title="observado"),
        alt.Tooltip("esperado:Q", title="esperado por azar", format=".1f"),
    ]
    bars = base.mark_bar(color=BLUE, size=28).encode(
        y=alt.Y("observado:Q", title=y_title), tooltip=tooltip
    )
    ticks = base.mark_tick(color=ORANGE, thickness=3, size=40).encode(
        y="esperado:Q", tooltip=tooltip
    )
    return (bars + ticks).properties(height=260)


def cumulative_vs_chance(df: pd.DataFrame, x_title: str):
    """Acumulado observado vs. esperado con banda del 95 % (df: i, observado, esperado, lo, hi)."""
    base = alt.Chart(df).encode(x=alt.X("i:Q", title=x_title))
    band = base.mark_area(color=GRAY, opacity=0.25).encode(
        y=alt.Y("lo:Q", title="Aciertos acumulados"), y2="hi:Q"
    )
    long = df.melt(id_vars=["i"], value_vars=["observado", "esperado"], var_name="serie")
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x="i:Q",
            y="value:Q",
            color=alt.Color(
                "serie:N",
                scale=alt.Scale(domain=["observado", "esperado"], range=[BLUE, GRAY]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=[
                alt.Tooltip("i:Q", title="sorteos"),
                alt.Tooltip("serie:N", title="serie"),
                alt.Tooltip("value:Q", title="aciertos", format=".1f"),
            ],
        )
    )
    return (band + lines).properties(height=280)
