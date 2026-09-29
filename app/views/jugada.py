"""Página 1: el usuario prueba una jugada y ve que no se distingue del azar."""

import common as c
import numpy as np
import pandas as pd
import streamlit as st

from baloto_ml.config import EXPECTED_HITS, JACKPOT_ODDS, JUEGOS, P_SUPER
from baloto_ml.evaluation.stats import hits_pmf

st.title("Prueba tu jugada")
st.write(
    'Escoge 5 balotas y una superbalota. Verás qué tanto le "gusta" tu jugada al modelo, '
    "cómo se compara con 10 000 jugadas al azar y cuántos aciertos habría tenido en cada sorteo "
    "de la historia."
)


def _random_play() -> None:
    rng = np.random.default_rng()
    st.session_state["balotas"] = sorted(
        int(x) for x in rng.choice(np.arange(1, 44), 5, replace=False)
    )
    st.session_state["superbalota"] = int(rng.integers(1, 17))


if "balotas" not in st.session_state:
    _random_play()

left, right = st.columns([3, 2], vertical_alignment="bottom")
with left:
    juego = st.segmented_control(
        "Juego", JUEGOS, format_func=c.NOMBRE.get, default="baloto", required=True
    )
with right:
    st.button("Jugada al azar", icon=":material/casino:", on_click=_random_play)

col_b, col_s = st.columns([3, 1])
balls = col_b.multiselect(
    "Tus 5 balotas (del 1 al 43, sin repetir)",
    options=list(range(1, 44)),
    max_selections=5,
    key="balotas",
)
sb = col_s.selectbox("Superbalota (1 a 16)", list(range(1, 17)), key="superbalota")
if len(balls) != 5:
    st.info("Elige exactamente 5 balotas para continuar.")
    st.stop()
balls = sorted(balls)

# ----------------------------------------------------------------- 1. lo que "opina" el modelo
st.header("1. Lo que opina el modelo", divider="gray")
version = c.latest_version(juego)
if version is None:
    st.error("No hay un modelo registrado para este juego todavía.")
    st.stop()
nd = c.load_forecast(juego, version)
score = c.play_score(nd, balls, sb)
scores = c.random_play_scores(nd, (juego, version))
pct = 100 * float(np.mean(scores <= score))

m1, m2, m3 = st.columns(3)
m1.metric(
    "Puntaje del modelo",
    f"{score:+.3f}".replace(".", ","),
    help="Suma de log(p / p_azar) de tus 6 números según las probabilidades del modelo de "
    f"producción para el sorteo {nd.n_sorteo}. 0 = lo mismo que el azar.",
)
m2.metric("Percentil entre 10 000 jugadas al azar", f"{pct:.0f}")
m3.metric(
    "Tu probabilidad real del premio mayor",
    f"1 en {c.es_int(JACKPOT_ODDS)}",
    help="Igual para cualquier combinación: el sorteo es aleatorio.",
)
st.altair_chart(
    c.histogram_with_marker(scores, score, "Puntaje del modelo", "tu jugada"), width="stretch"
)
if not nd.up_to_date:
    st.warning(
        "El modelo vigente no está al día con el último sorteo; sus probabilidades son de antes."
    )
st.caption(
    f"Modelo `{nd.version}` ({c.NOMBRE_MODELO.get(nd.model, nd.model)}), probabilidades "
    f"para el sorteo {nd.n_sorteo} del {nd.fecha:%d/%m/%Y}."
)

ev = c.load_report(f"reports/evaluation/{juego}.json")
pg = ev["modelos"][ev["modelo_produccion"]].get("percentil_ganadoras") if ev else None
if pg:
    pcts = np.array(pg["percentiles"]) * 100
    media = 100 * pg["media"]
    st.markdown(
        f"""**¿Y eso sirve de algo? No.** Un percentil alto solo dice que el modelo "cree" en tu
jugada. Si esa creencia valiera algo, las combinaciones que de verdad ganaron habrían tenido,
antes del sorteo, puntajes altos. En los {len(pcts)} sorteos de prueba, la combinación ganadora
cayó en promedio en el **percentil {media:.0f}** del puntaje del modelo. Es lo mismo que se
obtendría eligiendo al azar (50), y su distribución es plana:"""
    )
    counts, edges = np.histogram(pcts, bins=10, range=(0, 100))
    dist = pd.DataFrame(
        {
            "percentil": [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:], strict=True)],
            "observado": counts,
            "esperado": len(pcts) / 10,
        }  # fmt: skip
    )
    st.altair_chart(
        c.observed_vs_expected(
            dist, "percentil", "Percentil de la combinación ganadora", "Sorteos"
        ),
        width="stretch",
    )

# ------------------------------------------------------------------------------- 2. backtest
st.header("2. ¿Cuánto habría acertado en la historia?", divider="gray")
draws = c.load_draws()
bt = c.backtest(draws, juego, balls, sb)
n = len(bt)
pmf = hits_pmf()
b1, b2, b3 = st.columns(3)
b1.metric(
    "Aciertos promedio por sorteo",
    c.es(bt["aciertos"].mean()),
    delta=f"azar: {c.es(EXPECTED_HITS)}",
    delta_color="off",
)
b2.metric(
    "Sorteos con 3 o más aciertos",
    c.es_int((bt["aciertos"] >= 3).sum()),
    delta=f"azar: {c.es(n * pmf[3:].sum(), 1)}",
    delta_color="off",
)
b3.metric(
    "Veces que acertó la superbalota",
    c.es_int(bt["superbalota"].sum()),
    delta=f"azar: {c.es(n * P_SUPER, 1)}",
    delta_color="off",
)

dist = pd.DataFrame(
    {
        "aciertos": range(6),
        "observado": np.bincount(bt["aciertos"], minlength=6),
        "esperado": n * pmf,
    }
)
cum = pd.DataFrame({"i": np.arange(1, n + 1), "observado": bt["aciertos"].cumsum().to_numpy()})
cum["esperado"] = cum["i"] * EXPECTED_HITS
sd = np.sqrt(cum["i"] * 5 * (5 / 43) * (38 / 43) * (38 / 42))
cum["lo"], cum["hi"] = cum["esperado"] - 1.96 * sd, cum["esperado"] + 1.96 * sd
g1, g2 = st.columns(2)
g1.altair_chart(
    c.observed_vs_expected(dist, "aciertos", "Balotas acertadas en un sorteo", "Sorteos"),
    width="stretch",
)
g2.altair_chart(
    c.cumulative_vs_chance(cum, f"Sorteos de {c.NOMBRE[juego]} desde {bt['fecha'].min():%Y}"),
    width="stretch",
)
st.caption(
    f"Barras: tu jugada en los {c.es_int(n)} sorteos de {c.NOMBRE[juego]} del "
    f"{bt['fecha'].min():%d/%m/%Y} al {bt['fecha'].max():%d/%m/%Y}. Marcas naranjas y banda gris: "
    "lo esperado por azar para cualquier combinación."
)

cats = []
for k in range(5, -1, -1):
    for with_sb in (True, False):
        if k <= 2 and not with_sb:
            continue  # sin premio
        mask = (bt["aciertos"] == k) & (bt["superbalota"] == with_sb)
        cats.append(
            {
                "categoría": f"{k} balotas" + (" + superbalota" if with_sb else ""),
                "veces": int(mask.sum()),
                "esperado por azar": n * pmf[k] * (P_SUPER if with_sb else 1 - P_SUPER),
            }
        )
with st.expander("Categorías de premio del Baloto (sin montos)"):
    st.dataframe(
        pd.DataFrame(cats),
        hide_index=True,
        column_config={"esperado por azar": st.column_config.NumberColumn(format="%.2f")},
    )
st.info(
    "Cualquier combinación, incluida la que el modelo sugiere, tiene exactamente la misma "
    f"probabilidad: 1 en {c.es_int(JACKPOT_ODDS)} para el premio mayor.",
    icon=":material/info:",
)
