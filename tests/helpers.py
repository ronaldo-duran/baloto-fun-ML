"""Utilidades de tests: formato original de los CSV, páginas HTML sintéticas y HTTP falso."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]  # fmt: skip
DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

ROBOTS_OK = "User-agent: *\nDisallow: /admin-baloto/\nDisallow: /api/\n"
BASE = "https://www.baloto.com"


def fecha_es(ts) -> str:
    return f"{ts.day:02d} de {MESES_ES[ts.month - 1]} de {ts.year}"


def write_raw_csv(df: pd.DataFrame, path: Path) -> None:
    """Sorteos de UN juego en el formato original (Kaggle), del más reciente al más antiguo."""
    g = df.sort_values("n_sorteo", ascending=False)
    out = pd.DataFrame(
        {
            "Date": [fecha_es(d) for d in g["fecha"]],
            "C1": g["b1"].to_numpy(),
            "C2": g["b2"].to_numpy(),
            "C3": g["b3"].to_numpy(),
            "C4": g["b4"].to_numpy(),
            "C5": g["b5"].to_numpy(),
            "SB": g["superbalota"].to_numpy(),
            "#Sorteo": g["n_sorteo"].to_numpy(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def render_results_page(
    n: int, fecha_txt: str, dia: str, balls: Sequence[int], sb: int, *, thousands: bool = True
) -> str:
    """Página mínima con la estructura de baloto.com, con señuelos que el parser debe ignorar."""
    n_txt = f"{n:,}".replace(",", ".") if thousands else str(n)
    yellow = "".join(
        f'<div class="col-2"><div class="text-center"><div class="yellow-ball gotham-medium">'
        f"\n{b:02d}\n</div></div></div>"
        for b in balls
    )
    return f"""<html><head><style>.container-balls-results {{ margin: 0 }}</style></head>
<body>
<nav><a href="/resultados">PRÓXIMO SORTEO</a></nav>
<div class="mt-2 lh30 border-left-blue ps-3">
  <div class="gotham-medium dark-blue fs-5"><strong>SORTEO {n_txt}</strong></div>
  <div class="gotham-medium dark-blue fs-5 text-uppercase"><strong>{dia}</strong></div>
  <div class="gotham-medium dark-blue">{fecha_txt}</div>
</div>
<div class="text-center my-3 gotham-medium dark-blue fs-6">
  ACUMULADO DEL SORTEO: $38.800 MILLONES
</div>
<div class="container-balls-results"><div class="row">{yellow}
  <div class="col-2"><div class="text-center"><div class="red-ball gotham-medium">
{sb:02d}
</div></div></div>
</div></div>
<div class="ultimos">
  <div class="yellow-ball-results">01</div><div class="pink-ball-results">02</div>
</div>
<div class="gotham-book white-color">PRÓXIMO SORTEO</div><div>30 de Septiembre de 2026</div>
</body></html>"""


def page_for_row(row) -> str:
    """Página sintética para una fila del esquema común."""
    return render_results_page(
        int(row.n_sorteo),
        fecha_es(row.fecha),
        DIAS_ES[row.fecha.dayofweek],
        [row.b1, row.b2, row.b3, row.b4, row.b5],
        int(row.superbalota),
    )


class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeSession:
    """Sesión HTTP falsa: `routes` asocia URL -> respuesta, excepción o lista (una por llamada).

    Cualquier URL sin ruta responde 302 (así se comporta baloto.com con sorteos inexistentes).
    """

    def __init__(self, routes: dict) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        r = self.routes.get(url, FakeResponse(302))
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r

    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]


def site_routes(draws: pd.DataFrame, robots: str = ROBOTS_OK) -> dict:
    """Rutas de un baloto.com falso que publica exactamente `draws`."""
    routes: dict = {f"{BASE}/robots.txt": FakeResponse(200, robots)}
    for row in draws.itertuples():
        routes[f"{BASE}/resultados-{row.juego}/{row.n_sorteo}"] = FakeResponse(
            200, page_for_row(row)
        )
    return routes
