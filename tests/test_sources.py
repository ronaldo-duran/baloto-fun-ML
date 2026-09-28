from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests
from helpers import (
    BASE,
    FakeResponse,
    FakeSession,
    render_results_page,
    site_routes,
    write_raw_csv,
)

from baloto_ml.data.schema import COLUMNS
from baloto_ml.data.sources import (
    ManualCSVSource,
    PageParseError,
    RawHistorySource,
    WebSource,
    WebSourceBlockedError,
    WebSourceError,
    parse_results_page,
)
from baloto_ml.data.store import write_draws


def make_source(session: FakeSession, **kwargs) -> tuple[WebSource, list[float]]:
    sleeps: list[float] = []
    src = WebSource(session=session, sleep=sleeps.append, **kwargs)
    return src, sleeps


# ---------------------------------------------------------------- histórico y CSV manuales


def test_raw_history_source_maps_original_format(history: pd.DataFrame, tmp_path) -> None:
    files = {}
    for juego in ("baloto", "revancha"):
        files[juego] = tmp_path / f"{juego}.csv"
        write_raw_csv(history[history["juego"] == juego], files[juego])
    out = RawHistorySource(files).fetch()
    pd.testing.assert_frame_equal(out, history)


def test_raw_history_source_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("fecha,b1\n2026-01-01,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formato original"):
        RawHistorySource({"baloto": path}).fetch()


def test_manual_csv_source_reads_all_files_and_dedupes(history: pd.DataFrame, tmp_path) -> None:
    write_draws(history.iloc[:10], tmp_path / "a.csv")
    write_draws(history.iloc[5:20], tmp_path / "b.csv")  # 5 filas repetidas entre archivos
    out = ManualCSVSource(tmp_path).fetch()
    assert len(out) == 20
    assert list(out.columns) == COLUMNS


def test_manual_csv_source_requires_juego(tmp_path) -> None:
    (tmp_path / "x.csv").write_text(
        "fecha,n_sorteo,b1,b2,b3,b4,b5,superbalota\n2026-05-25,2661,1,2,3,4,5,6\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="juego"):
        ManualCSVSource(tmp_path).fetch()


# ---------------------------------------------------------------- parser de baloto.com


def test_parse_results_page() -> None:
    html = render_results_page(2660, "23 de Mayo de 2026", "Sábado", [23, 27, 29, 30, 38], 8)
    draw = parse_results_page(html)
    assert draw.n_sorteo == 2660  # "SORTEO 2.660" con punto de miles
    assert draw.fecha == date(2026, 5, 23)  # no la fecha del "próximo sorteo"
    assert draw.balls == (23, 27, 29, 30, 38)
    assert draw.superbalota == 8
    assert draw.weekday_name == "Sábado"


def test_parse_results_page_without_thousands_separator() -> None:
    html = render_results_page(
        999, "02 de Junio de 2025", "Lunes", [1, 2, 3, 4, 5], 16, thousands=False
    )
    assert parse_results_page(html).n_sorteo == 999


def test_parse_results_page_detects_layout_changes() -> None:
    with pytest.raises(PageParseError, match="bloque de balotas"):
        parse_results_page("<html><body>SORTEO 2.660</body></html>")
    html = render_results_page(2660, "23 de Mayo de 2026", "Sábado", [23, 27, 29, 30], 8)
    with pytest.raises(PageParseError, match="Se esperaban 5"):
        parse_results_page(html)


# ---------------------------------------------------------------- WebSource


def test_web_source_fetches_until_redirect(history: pd.DataFrame) -> None:
    published = history[history["n_sorteo"].isin([1058, 1059])]
    session = FakeSession(site_routes(published))
    src, sleeps = make_source(session, delay_s=0.5)
    out = src.fetch({"baloto": 1057, "revancha": 1057})

    pd.testing.assert_frame_equal(out, published.reset_index(drop=True))
    assert session.urls() == [
        f"{BASE}/robots.txt",
        f"{BASE}/resultados-baloto/1058",
        f"{BASE}/resultados-baloto/1059",
        f"{BASE}/resultados-baloto/1060",  # 302: fin
        f"{BASE}/resultados-revancha/1058",
        f"{BASE}/resultados-revancha/1059",
        f"{BASE}/resultados-revancha/1060",
    ]
    for _, kwargs in session.calls:
        assert kwargs["allow_redirects"] is False
        assert "baloto-fun-ML" in kwargs["headers"]["User-Agent"]
    assert sleeps == [0.5] * (len(session.calls) - 1)  # pausa entre todas las peticiones


def test_web_source_stops_on_block_without_retrying(history: pd.DataFrame) -> None:
    routes = site_routes(history.iloc[:0])
    routes[f"{BASE}/resultados-baloto/1060"] = FakeResponse(403)
    session = FakeSession(routes)
    src, _ = make_source(session)
    with pytest.raises(WebSourceBlockedError):
        src.fetch({"baloto": 1059, "revancha": 1059})
    assert session.urls().count(f"{BASE}/resultados-baloto/1060") == 1


def test_web_source_rejects_page_with_other_draw_number(history: pd.DataFrame) -> None:
    row = history[(history["juego"] == "baloto") & (history["n_sorteo"] == 1059)]
    routes = site_routes(row)
    routes[f"{BASE}/resultados-baloto/1060"] = routes[f"{BASE}/resultados-baloto/1059"]
    src, _ = make_source(FakeSession(routes), juegos=("baloto",))
    with pytest.raises(PageParseError, match="muestra el sorteo 1059"):
        src.fetch({"baloto": 1059})


def test_web_source_respects_robots(history: pd.DataFrame) -> None:
    session = FakeSession(site_routes(history, robots="User-agent: *\nDisallow: /\n"))
    src, _ = make_source(session)
    with pytest.raises(WebSourceError, match="robots.txt"):
        src.fetch({"baloto": 1000, "revancha": 1000})
    assert session.urls() == [f"{BASE}/robots.txt"]


def test_web_source_retries_transient_errors(history: pd.DataFrame) -> None:
    row = history[(history["juego"] == "baloto") & (history["n_sorteo"] == 1059)]
    routes = site_routes(row)
    url = f"{BASE}/resultados-baloto/1059"
    routes[url] = [requests.ConnectionError("caída"), FakeResponse(503), routes[url]]
    src, sleeps = make_source(FakeSession(routes), juegos=("baloto",), delay_s=0, backoff_s=1)
    out = src.fetch({"baloto": 1058})
    assert out["n_sorteo"].tolist() == [1059]
    assert [s for s in sleeps if s > 0] == [1, 2]  # backoff exponencial


def test_web_source_gives_up_after_retries() -> None:
    url = f"{BASE}/resultados-baloto/5"
    routes = {f"{BASE}/robots.txt": FakeResponse(200, ""), url: FakeResponse(500)}
    src, _ = make_source(FakeSession(routes), juegos=("baloto",), retries=3)
    with pytest.raises(WebSourceError, match="tras 3 intentos"):
        src.fetch({"baloto": 4})


def test_web_source_requires_known_last() -> None:
    src, _ = make_source(FakeSession({}))
    with pytest.raises(ValueError):
        src.fetch(None)
