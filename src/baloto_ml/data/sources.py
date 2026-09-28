"""Fuentes de sorteos: histórico original, CSV manuales y resultados publicados en baloto.com."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from baloto_ml import __version__
from baloto_ml.config import BALLS_PER_DRAW, JUEGOS
from baloto_ml.data.schema import (
    BALL_COLS,
    COLUMNS,
    SUPER_COL,
    canonicalize,
    concat_draws,
    empty_draws,
    parse_fecha_es,
    read_canonical_csv,
    weekday_from_name,
)

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Fuente de sorteos. `fetch` devuelve filas en el esquema común (puede venir vacía)."""

    name: str = "base"

    @abstractmethod
    def fetch(self, known_last: Mapping[str, int] | None = None) -> pd.DataFrame:
        """`known_last`: último `n_sorteo` conocido por juego (para fuentes incrementales)."""


class RawHistorySource(DataSource):
    """Histórico original (formato Kaggle: Date, C1..C5, SB, #Sorteo), un archivo por juego."""

    name = "raw"
    COLUMN_MAP = {
        "Date": "fecha",
        "C1": "b1",
        "C2": "b2",
        "C3": "b3",
        "C4": "b4",
        "C5": "b5",
        "SB": "superbalota",
        "#Sorteo": "n_sorteo",
    }

    def __init__(self, files: Mapping[str, Path]) -> None:
        self.files = dict(files)

    def fetch(self, known_last: Mapping[str, int] | None = None) -> pd.DataFrame:
        frames = []
        for juego, path in self.files.items():
            if not path.exists():
                logger.warning("No existe el histórico de %s: %s", juego, path)
                continue
            raw = pd.read_csv(path, dtype=str)
            missing = [c for c in self.COLUMN_MAP if c not in raw.columns]
            if missing:
                raise ValueError(f"{path}: faltan columnas del formato original: {missing}")
            df = raw.rename(columns=self.COLUMN_MAP)[list(self.COLUMN_MAP.values())].copy()
            df["fecha"] = df["fecha"].map(parse_fecha_es)
            df["n_sorteo"] = df["n_sorteo"].str.replace(".", "", regex=False)
            df["juego"] = juego
            frames.append(df)
            logger.debug("%s: %d sorteos leídos de %s", juego, len(df), path.name)
        return concat_draws(*frames)


class ManualCSVSource(DataSource):
    """Sorteos nuevos en `data/incoming/*.csv`, ya en el esquema común (con columna `juego`)."""

    name = "manual"

    def __init__(self, incoming_dir: Path, pattern: str = "*.csv") -> None:
        self.incoming_dir = incoming_dir
        self.pattern = pattern

    def files(self) -> list[Path]:
        return sorted(p for p in self.incoming_dir.glob(self.pattern) if p.is_file())

    def fetch(self, known_last: Mapping[str, int] | None = None) -> pd.DataFrame:
        frames = [read_canonical_csv(p) for p in self.files()]
        df = concat_draws(*frames)
        return canonicalize(df.drop_duplicates(subset=COLUMNS))


# --------------------------------------------------------------------------------------------
# Web: baloto.com publica una página por sorteo en /resultados-<juego>/<n_sorteo>.
# --------------------------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://www.baloto.com"
RESULT_PATHS: dict[str, str] = {
    "baloto": "/resultados-baloto/{n}",
    "revancha": "/resultados-revancha/{n}",
}
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
BLOCKED_CODES = frozenset({401, 403, 429})

_SORTEO_RE = re.compile(r"SORTEO\s+(\d{1,3}(?:\.\d{3})+|\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b", re.IGNORECASE)
_DAY_RE = re.compile(r"\b(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)\b", re.I)


class WebSourceError(RuntimeError):
    """No se pudieron leer resultados de la web."""


class WebSourceBlockedError(WebSourceError):
    """El sitio rechazó el acceso (401/403/429). No se insiste ni se intenta eludir."""


class PageParseError(WebSourceError):
    """La página no tiene la estructura esperada (¿cambió el sitio?)."""


@dataclass(frozen=True)
class ParsedDraw:
    n_sorteo: int
    fecha: date
    balls: tuple[int, ...]
    superbalota: int
    weekday_name: str | None = None


def parse_results_page(html: str) -> ParsedDraw:
    """Extrae número, fecha, balotas y superbalota de una página de resultados de baloto.com."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.container-balls-results")
    if container is None:
        raise PageParseError("No se encontró el bloque de balotas (div.container-balls-results)")
    try:
        balls = tuple(int(el.get_text(strip=True)) for el in container.select("div.yellow-ball"))
        reds = [int(el.get_text(strip=True)) for el in container.select("div.red-ball")]
    except ValueError as exc:
        raise PageParseError(f"Balota con texto no numérico: {exc}") from exc
    if len(balls) != BALLS_PER_DRAW or len(reds) != 1:
        raise PageParseError(
            f"Se esperaban {BALLS_PER_DRAW} balotas y 1 superbalota; hay {len(balls)} y {len(reds)}"
        )

    label = soup.find(string=_SORTEO_RE)
    if label is None:
        raise PageParseError("No se encontró el número de sorteo")
    n_sorteo = int(_SORTEO_RE.search(label).group(1).replace(".", ""))

    # La fecha está en el mismo bloque de cabecera que "SORTEO N": se sube hasta encontrarla.
    header = label.parent
    while header is not None and not _DATE_RE.search(header.get_text(" ", strip=True)):
        header = header.parent
    if header is None:
        raise PageParseError("No se encontró la fecha del sorteo")
    header_text = header.get_text(" ", strip=True)
    fecha = parse_fecha_es(_DATE_RE.search(header_text).group(1))
    day = _DAY_RE.search(header_text)
    return ParsedDraw(n_sorteo, fecha, balls, reds[0], day.group(1) if day else None)


def default_user_agent() -> str:
    return (
        f"baloto-fun-ML/{__version__} "
        "(proyecto educativo; +https://github.com/ronaldo-duran/baloto-fun-ML)"
    )


class WebSource(DataSource):
    """Lee de baloto.com los sorteos posteriores al último conocido, de forma respetuosa.

    - Respeta robots.txt (hoy solo prohíbe /admin-baloto/ y /api/; no se usa la API).
    - Se identifica con un User-Agent honesto y no intenta eludir bloqueos: ante 401/403/429
      se detiene con `WebSourceBlockedError` (p. ej., IPs de datacenter como Kaggle o CI).
    - Espera `delay_s` entre peticiones y reintenta solo errores transitorios (red, 5xx).
    - NO sigue redirecciones: un sorteo inexistente redirige a /resultados, que muestra el
      último sorteo; seguirla guardaría ese resultado otra vez. La redirección marca el final.
    """

    name = "web"

    def __init__(
        self,
        juegos: Sequence[str] = JUEGOS,
        *,
        base_url: str = DEFAULT_BASE_URL,
        delay_s: float = 2.5,
        timeout_s: float = 20.0,
        retries: int = 3,
        backoff_s: float = 5.0,
        max_draws_per_game: int = 200,
        user_agent: str | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        respect_robots: bool = True,
    ) -> None:
        self.juegos = tuple(juegos)
        self.base_url = base_url.rstrip("/")
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_draws_per_game = max_draws_per_game
        self.user_agent = user_agent or default_user_agent()
        self.session = session or requests.Session()
        self.respect_robots = respect_robots
        self._sleep = sleep
        self._headers = {"User-Agent": self.user_agent, "Accept-Language": "es-CO,es;q=0.9"}
        self._requests_made = 0

    def url_for(self, juego: str, n_sorteo: int) -> str:
        return self.base_url + RESULT_PATHS[juego].format(n=n_sorteo)

    def fetch(self, known_last: Mapping[str, int] | None = None) -> pd.DataFrame:
        if known_last is None:
            raise ValueError("WebSource necesita el último n_sorteo conocido de cada juego")
        if self.respect_robots:
            self.check_robots()
        rows = []
        for juego in self.juegos:
            if juego not in known_last:
                raise ValueError(f"Falta el último sorteo conocido de {juego}")
            n = int(known_last[juego]) + 1
            fetched = 0
            while fetched < self.max_draws_per_game:
                draw = self.fetch_draw(juego, n)
                if draw is None:
                    break
                rows.append(
                    {
                        "fecha": draw.fecha,
                        "n_sorteo": draw.n_sorteo,
                        "juego": juego,
                        **dict(zip(BALL_COLS, draw.balls, strict=True)),
                        SUPER_COL: draw.superbalota,
                    }
                )
                fetched += 1
                n += 1
            else:
                logger.warning(
                    "%s: se alcanzó el máximo de %d sorteos por ejecución",
                    juego,
                    self.max_draws_per_game,
                )
            logger.info("%s: %d sorteos nuevos en la web", juego, fetched)
        return canonicalize(pd.DataFrame(rows, columns=COLUMNS)) if rows else empty_draws()

    def fetch_draw(self, juego: str, n_sorteo: int) -> ParsedDraw | None:
        """Resultado de un sorteo, o None si aún no existe (redirección o 404)."""
        url = self.url_for(juego, n_sorteo)
        resp = self._get(url)
        if resp.status_code in REDIRECT_CODES or resp.status_code == 404:
            logger.debug("%s: sorteo %d aún no publicado (%d)", juego, n_sorteo, resp.status_code)
            return None
        if resp.status_code in BLOCKED_CODES:
            raise WebSourceBlockedError(
                f"{url} respondió {resp.status_code}: el sitio rechaza el acceso automatizado "
                "desde esta red. No se insiste; usa la fuente manual (data/incoming/)."
            )
        if resp.status_code != 200:
            raise WebSourceError(f"{url} respondió {resp.status_code}")
        draw = parse_results_page(resp.text)
        if draw.n_sorteo != n_sorteo:
            raise PageParseError(f"{url} muestra el sorteo {draw.n_sorteo}, no el {n_sorteo}")
        if draw.weekday_name is not None:
            weekday = weekday_from_name(draw.weekday_name)
            if weekday is not None and weekday != draw.fecha.weekday():
                logger.warning(
                    "%s %d: la página dice '%s' pero %s no cae ese día",
                    juego,
                    n_sorteo,
                    draw.weekday_name,
                    draw.fecha,
                )
        return draw

    def check_robots(self) -> None:
        """Verifica en robots.txt que las páginas de resultados se pueden leer."""
        robots_url = f"{self.base_url}/robots.txt"
        resp = self._get(robots_url)
        if resp.status_code == 200:
            parser = RobotFileParser()
            parser.parse(resp.text.splitlines())
            for juego in self.juegos:
                url = self.url_for(juego, 1)
                if not parser.can_fetch(self.user_agent, url):
                    raise WebSourceError(f"robots.txt no permite leer {url}")
        elif resp.status_code in BLOCKED_CODES:
            raise WebSourceBlockedError(f"{robots_url} respondió {resp.status_code}")
        elif 400 <= resp.status_code < 500:
            logger.info("Sin robots.txt (%d): se asume permitido", resp.status_code)
        else:
            raise WebSourceError(f"{robots_url} respondió {resp.status_code}")

    def _get(self, url: str) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            if self._requests_made:
                self._sleep(self.delay_s)
            self._requests_made += 1
            try:
                resp = self.session.get(
                    url, headers=self._headers, timeout=self.timeout_s, allow_redirects=False
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("%s: %s (intento %d/%d)", url, exc, attempt, self.retries)
            else:
                if resp.status_code < 500:
                    return resp
                last_exc = WebSourceError(f"{url} respondió {resp.status_code}")
                logger.warning(
                    "%s: %d (intento %d/%d)", url, resp.status_code, attempt, self.retries
                )
            if attempt < self.retries:
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
        raise WebSourceError(f"No se pudo leer {url} tras {self.retries} intentos") from last_exc
