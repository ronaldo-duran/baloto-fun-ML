"""Resumen en Markdown del estado del pipeline (para el resumen de GitHub Actions).

Uso: uv run python scripts/pipeline_summary.py >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

from baloto_ml.config import EXPECTED_HITS, JUEGOS, default_paths
from baloto_ml.data.store import read_draws, read_json
from baloto_ml.models.registry import ModelRegistry


def main() -> None:
    paths = default_paths()
    draws = read_draws(paths.processed_draws)
    registry = ModelRegistry(paths.models)
    live = read_json(paths.live_summary) if paths.live_summary.exists() else {"juegos": {}}
    print("## Pipeline de reentrenamiento\n")
    print("> Experimento educativo. El Baloto es aleatorio; este modelo no predice resultados.\n")
    print("| Juego | Último sorteo | Modelo vigente | Predicciones | Aciertos (azar) | Pendiente |")
    print("|---|---|---|---|---|---|")
    for juego in JUEGOS:
        g = draws[draws["juego"] == juego]
        last = f"{int(g['n_sorteo'].max())} ({g['fecha'].max():%Y-%m-%d})" if len(g) else "-"
        s = live["juegos"].get(juego, {})
        hits = (
            f"{s['aciertos']} ({s['conciliadas'] * EXPECTED_HITS:.1f})"
            if s.get("conciliadas")
            else "-"
        )
        pending = ", ".join(
            f"{p['n_sorteo']} ({p['fecha_sorteo']}): "
            f"{p['combinacion']} + {p['superbalota_sugerida']}"
            for p in s.get("pendientes", [])
        )
        print(
            f"| {juego} | {last} | `{registry.latest_version(juego) or '-'}` | "
            f"{s.get('predicciones', 0)} | {hits} | {pending or '-'} |"
        )


if __name__ == "__main__":
    main()
