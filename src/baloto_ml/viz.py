"""Gráficos estáticos (matplotlib) para notebooks y README. Requiere el grupo `notebooks`.

Paleta validada (slots 1-3 de la paleta de referencia, todas las parejas pasan CVD); el texto
usa tokens de tinta, nunca el color de la serie; grilla fina, sólida y recesiva.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"  # gris neutro: banda de lo esperable por azar
BLUE = "#2a78d6"  # slot 1
ORANGE = "#eb6834"  # slot 2 (énfasis)
AQUA = "#1baf7a"  # slot 3 (contraste < 3:1: siempre con leyenda o etiqueta directa)
DIVERGING = LinearSegmentedColormap.from_list("azul_gris_rojo", [BLUE, BAND, "#e34948"])
# El color sigue a la entidad en todos los gráficos (nunca al orden en que aparece).
MODEL_COLORS = {
    "logistica": BLUE,
    "gradient_boosting": ORANGE,
    "frecuencia": AQUA,
    "constante": MUTED,
}
DISPLAY_NAMES = {
    "logistica": "logística",
    "gradient_boosting": "gradient boosting",
    "frecuencia": "frecuencia",
    "constante": "constante",
}
# Rampa ordinal (un solo tono, claro -> oscuro) para categorías ordenadas como la fuerza del efecto.
ORDINAL_BLUES = ("#86b6ef", "#2a78d6", "#104281")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 9,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "figure.dpi": 100,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    return path


def _note(ax: Axes, text: str, x: float = 0.0, y: float = 1.0) -> None:
    """Subtítulo en tinta secundaria bajo el título."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def frequency_panel(
    ax: Axes, counts: np.ndarray, n_draws: int, p: float, title: str, label: str = "Balota"
) -> int:
    """Barras de frecuencia por número con lo esperado por azar y su banda del 95 %.

    Las barras fuera de la banda van en el color de énfasis; por azar se espera que ~5 % caiga
    fuera. Devuelve cuántas quedaron fuera.
    """
    k = len(counts)
    expected = n_draws * p
    sd = np.sqrt(n_draws * p * (1 - p))
    lo, hi = expected - 1.96 * sd, expected + 1.96 * sd
    outside = (counts < lo) | (counts > hi)
    x = np.arange(1, k + 1)
    ax.axhspan(lo, hi, color=BAND, zorder=0, lw=0)
    ax.bar(x[~outside], counts[~outside], width=0.62, color=BLUE, lw=0, label="dentro de la banda")
    if outside.any():
        ax.bar(
            x[outside], counts[outside], width=0.62, color=ORANGE, lw=0, label="fuera de la banda"
        )
    ax.axhline(expected, color=INK_2, lw=1)
    ax.text(k + 0.9, expected, f"esperado {expected:.1f}", fontsize=8, color=INK_2, va="center")
    ax.set_xlim(0.3, k + 0.7)
    ax.set_ylim(0, max(counts.max(), hi) * 1.22)  # aire arriba para la leyenda
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=6.5 if k > 20 else 8)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(label)
    ax.set_ylabel("Veces que salió")
    ax.set_title(title, pad=18)
    _note(ax, f"banda gris: 95 % por azar para cada número · {outside.sum()} de {k} fuera")
    ax.legend(loc="upper right", ncols=2)
    return int(outside.sum())


def null_panel(
    ax: Axes,
    simulated: np.ndarray,
    observed: float,
    title: str,
    xlabel: str,
    curves: Sequence[tuple[np.ndarray, np.ndarray, str, str]] = (),
    bins: int = 60,
) -> None:
    """Distribución por azar (histograma) con el valor observado y curvas teóricas opcionales."""
    ax.hist(
        simulated,
        bins=bins,
        density=True,
        color=BLUE,
        alpha=0.35,
        edgecolor=SURFACE,
        linewidth=0.6,
        label="simulación bajo azar",
    )
    for xs, ys, name, color in curves:
        ax.plot(xs, ys, color=color, lw=2, label=name)
    pct = 100 * np.mean(simulated <= observed)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.32)  # franja libre arriba para la etiqueta y la leyenda
    ax.axvline(observed, color=INK, lw=1.5)
    ax.text(
        observed,
        top * 1.3,
        f" observado {observed:.3g}\n percentil {pct:.0f}",
        fontsize=8,
        color=INK,
        va="top",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidad")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, pad=8)
    lo, hi = ax.get_xlim()
    ax.legend(loc="upper left" if observed > (lo + hi) / 2 else "upper right")


def common_balls_panel(ax: Axes, observed: np.ndarray, expected: np.ndarray, title: str) -> None:
    """Balotas en común por fecha: barras observadas y marcas de lo esperado (hipergeométrica)."""
    k = np.arange(len(observed))
    ax.bar(k, observed, width=0.55, color=BLUE, lw=0, label="observado")
    ax.scatter(
        k,
        expected,
        marker="_",
        s=500,
        linewidths=2,
        color=INK,
        zorder=3,
        label="esperado si son independientes",
    )
    pad = 0.015 * max(observed.max(), expected.max())
    for xi, v, e in zip(k, observed, expected, strict=True):
        ax.text(xi, max(v, e) + pad, f"{int(v)}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(k)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Balotas en común entre Baloto y Revancha el mismo día")
    ax.set_ylabel("Sorteos")
    ax.set_title(title, pad=8)
    ax.legend(loc="upper right")


def corr_heatmap(ax: Axes, corr: np.ndarray, title: str, vmax: float = 0.2):
    """Correlaciones entre indicadores (balota j de un juego vs balota k del otro)."""
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-vmax, vmax=vmax, origin="lower")
    ticks = [0, 4, 9, 14, 19, 24, 29, 34, 39, 42]
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_yticks(ticks, [str(t + 1) for t in ticks])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Balota de Revancha")
    ax.set_ylabel("Balota de Baloto")
    ax.set_title(title, pad=8)
    return im


def interval_dots(
    ax: Axes,
    labels: Sequence[str],
    means: Sequence[float],
    intervals: Sequence[Sequence[float]],
    reference: float,
    band: Sequence[float] | None,
    title: str,
    xlabel: str,
    colors: Sequence[str] | None = None,
    note: str | None = None,
    fmt: str = "{:.3f}",
) -> None:
    """Media e IC95 % por fila frente a una referencia (y, si hay, su banda por azar)."""
    y = np.arange(len(labels))[::-1]
    colors = colors or [BLUE] * len(labels)
    if band is not None:
        ax.axvspan(band[0], band[1], color=BAND, lw=0, zorder=0)
    ax.axvline(reference, color=INK_2, lw=1)
    for yi, m, (lo, hi), c in zip(y, means, intervals, colors, strict=True):
        ax.plot([lo, hi], [yi, yi], color=INK_2, lw=2, solid_capstyle="round")
        ax.scatter([m], [yi], s=70, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m, yi + 0.16, fmt.format(m), fontsize=8, color=INK, ha="center", va="bottom")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2, labelsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=18)
    default = f"línea: azar ({fmt.format(reference)})" + (
        " · banda gris: 95 % por azar" if band is not None else ""
    )
    _note(ax, note or default)


def selection_scatter(
    ax: Axes,
    a: np.ndarray,
    b: np.ndarray,
    best: int,
    band: Sequence[float],
    reference: float,
    title: str,
) -> None:
    """Estrategias: rendimiento en el periodo A (donde se eligen) vs. B (donde se verifican)."""
    ax.axvspan(band[0], band[1], color=BAND, lw=0, zorder=0)  # solo A: ahí se elige
    ax.axvline(reference, color=INK_2, lw=1)
    ax.axhline(reference, color=INK_2, lw=1)
    mask = np.ones(len(a), dtype=bool)
    mask[best] = False
    ax.scatter(a[mask], b[mask], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               label="cada estrategia", zorder=3)  # fmt: skip
    ax.scatter([a[best]], [b[best]], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2,
               label="la mejor en A", zorder=4)  # fmt: skip
    ax.annotate(
        f"mejor en A: {a[best]:.3f}\nen B: {b[best]:.3f}",
        (a[best], b[best]),
        xytext=(0.98, 0.97),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8, "shrinkB": 6},
    )
    ax.grid(axis="x")
    ax.set_xlabel("Aciertos por sorteo en A (donde se elige)")
    ax.set_ylabel("Aciertos por sorteo en B (datos nuevos)")
    ax.set_title(title, pad=18)
    _note(ax, f"líneas: azar ({reference:.3f}) · banda gris: 95 % por azar en A")
    ax.legend(loc="lower left")


def percentile_hist(ax: Axes, percentiles: np.ndarray, title: str) -> None:
    """Percentil de las combinaciones ganadoras según el modelo: sin señal debe ser plano."""
    n = len(percentiles)
    counts, edges = np.histogram(percentiles, bins=10, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=0.08, color=BLUE, lw=0, label="combinaciones ganadoras")
    ax.axhline(n / 10, color=INK_2, lw=1)
    ax.text(1.01, n / 10, "si el modelo no\nsabe nada", fontsize=8, color=INK_2, va="center",
            transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v * 100)}" for v in np.linspace(0, 1, 6)])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Percentil de la combinación ganadora entre 2000 jugadas al azar")
    ax.set_ylabel("Sorteos")
    ax.set_title(title, pad=8)


def line_panel(
    ax: Axes,
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    colors: dict[str, str],
    title: str,
    xlabel: str,
    ylabel: str,
    reference: float | None = None,
    logx: bool = False,
    xticklabels: Sequence[str] | None = None,
) -> None:
    """Varias series con marcadores; la leyenda identifica cada una (nunca solo el color)."""
    if reference is not None:
        ax.axhline(reference, color=INK_2, lw=1)
    for name, ys in series.items():
        ax.plot(x, ys, color=colors[name], lw=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)  # fmt: skip
    if logx:
        ax.set_xscale("log")
    if xticklabels is not None:
        ax.set_xticks(list(x), list(xticklabels))
        ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.legend(loc="best")


def reliability_panel(
    ax: Axes, curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], p0: float, title: str
) -> None:
    """Calibración: probabilidad predicha (x) vs. frecuencia observada (y) por deciles."""
    lo_x = min(c[0].min() for c in curves.values())
    hi_x = max(c[0].max() for c in curves.values())
    grid = np.array([lo_x, hi_x])
    ax.plot(grid, grid, color=AXIS, lw=1, label="calibración perfecta")
    ax.axhline(p0, color=INK_2, lw=1, label=f"azar ({p0:.3f})")
    for name, (pred, obs, half) in curves.items():
        ax.errorbar(pred, obs, yerr=half, color=MODEL_COLORS.get(name, BLUE), lw=2, marker="o",
                    markersize=6, markeredgecolor=SURFACE, capsize=0,
                    label=DISPLAY_NAMES.get(name, name))  # fmt: skip
    ax.set_xlabel("Probabilidad que el modelo le daba a la balota (promedio del decil)")
    ax.set_ylabel("Fracción de veces que salió")
    ax.set_title(title, pad=8)
    ax.legend(loc="upper left")
