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
    band: Sequence[float],
    title: str,
    xlabel: str,
) -> None:
    """Media e IC95 % por modelo frente a lo esperado por azar y su banda."""
    y = np.arange(len(labels))[::-1]
    ax.axvspan(band[0], band[1], color=BAND, lw=0, zorder=0)
    ax.axvline(reference, color=INK_2, lw=1)
    for yi, m, (lo, hi) in zip(y, means, intervals, strict=True):
        ax.plot([lo, hi], [yi, yi], color=INK_2, lw=2, solid_capstyle="round")
        ax.scatter([m], [yi], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m, yi + 0.16, f"{m:.3f}", fontsize=8, color=INK, ha="center", va="bottom")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2, labelsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=18)
    _note(ax, f"línea: azar ({reference:.3f}) · banda gris: 95 % por azar")
