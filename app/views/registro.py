"""Página 2: registro de predicciones en vivo (aciertos acumulados del modelo vs. azar)."""

import common as c
import numpy as np
import pandas as pd
import streamlit as st

from baloto_ml.config import EXPECTED_HITS, JACKPOT_ODDS, JUEGOS, P_SUPER
from baloto_ml.data.calendar import COT
from baloto_ml.evaluation.stats import hits_variance
from baloto_ml.live.replay import summarize_percentiles

st.title("Registro en vivo")
st.write(
    "Antes de cada sorteo, el pipeline guarda la predicción del modelo: las probabilidades de "
    "cada número y la combinación sugerida. Después del sorteo la compara con el resultado real. "
    "Nada de esto se puede maquillar después: el registro solo crece y su historial está en git."
)


def _local(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(COT).dt.strftime("%d/%m/%Y %H:%M")


preds, rec = c.load_logs()
if preds.empty:
    st.info(
        "Todavía no hay predicciones registradas. La primera se guarda cuando el pipeline procesa "
        "un sorteo nuevo antes de que se juegue el siguiente.",
        icon=":material/hourglass_empty:",
    )
else:
    pending = preds[~preds["prediccion_id"].isin(rec["prediccion_id"])].sort_values("n_sorteo")
    if not pending.empty:
        st.subheader("Predicciones pendientes")
        st.dataframe(
            pd.DataFrame(
                {
                    "juego": pending["juego"].map(c.NOMBRE),
                    "sorteo": pending["n_sorteo"],
                    "fecha del sorteo": pending["fecha_sorteo"],
                    "combinación sugerida": pending["combinacion"],
                    "superbalota": pending["superbalota_sugerida"],
                    "registrada (hora Colombia)": _local(pending["creada_utc"]),
                    "modelo": pending["modelo_version"],
                }
            ),
            hide_index=True,
        )

    for juego in JUEGOS:
        r = rec[rec["juego"] == juego].sort_values("n_sorteo")
        st.subheader(c.NOMBRE[juego], divider="gray")
        if r.empty:
            st.caption("Aún no hay predicciones de este juego con resultado.")
            continue
        n = len(r)
        hits, sb_hits = int(r["aciertos"].sum()), int(r["acierto_superbalota"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sorteos con resultado", c.es_int(n))
        m2.metric(
            "Aciertos acumulados (5 balotas)",
            c.es_int(hits),
            delta=f"azar: {c.es(n * EXPECTED_HITS, 1)}",
            delta_color="off",
        )
        m3.metric(
            "Superbalotas acertadas",
            c.es_int(sb_hits),
            delta=f"azar: {c.es(n * P_SUPER, 2)}",
            delta_color="off",
        )
        m4.metric(
            "Percentil medio del resultado",
            f"{r['percentil_resultado'].mean():.0f}",
            delta="azar: 50",
            delta_color="off",
            help="Dónde cayó, en promedio, la combinación ganadora entre todas las jugadas "
            "posibles ordenadas por el puntaje que les daba el modelo antes del sorteo.",
        )
        i = np.arange(1, n + 1)
        cum = pd.DataFrame(
            {"i": i, "observado": r["aciertos"].cumsum().to_numpy(), "esperado": i * EXPECTED_HITS}
        )
        sd = np.sqrt(i * hits_variance())
        cum["lo"], cum["hi"] = cum["esperado"] - 1.96 * sd, cum["esperado"] + 1.96 * sd
        st.altair_chart(c.cumulative_vs_chance(cum, "Sorteos con resultado"), width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "sorteo": r["n_sorteo"],
                    "fecha": r["fecha_real"],
                    "sugerida": r["combinacion"] + " + " + r["superbalota_sugerida"].astype(str),
                    "resultado": r["resultado"] + " + " + r["superbalota_real"].astype(str),
                    "aciertos": r["aciertos"],
                    "superbalota": r["acierto_superbalota"].map({1: "sí", 0: "no"}),
                    "puntaje del resultado": r["puntaje_resultado"],
                    "percentil del resultado": r["percentil_resultado"],
                }
            ).iloc[::-1],
            hide_index=True,
            column_config={
                "puntaje del resultado": st.column_config.NumberColumn(format="%+.3f"),
                "percentil del resultado": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "**Puntaje y percentil del resultado:** lo que el modelo opinaba, antes del sorteo, de "
            "la combinación que salió. El puntaje es la suma de log(p / p_azar) de sus 5 balotas y "
            "su superbalota (0 = igual que el azar). El percentil la ubica entre las "
            f"{c.es_int(JACKPOT_ODDS)} jugadas posibles: 100 sería la favorita del modelo. Si el "
            "modelo no sabe nada, es un número al azar entre 0 y 100: la mitad de las veces cae "
            "entre 25 y 75, y una de cada diez por encima de 90."
        )

st.subheader("Simulación: dónde cayó cada ganadora en el último año", divider="gray")
sim = c.load_replay()
if sim.empty:
    st.caption("Todavía no hay simulación: corre `baloto-ml replay`.")
else:
    st.write(
        "El registro en vivo tiene pocos sorteos todavía, y con pocos datos es fácil ver patrones "
        "que no existen. Por eso se repitió el mismo procedimiento con todos los sorteos desde el "
        f"{pd.Timestamp(sim['fecha'].min()):%d/%m/%Y}: antes de cada uno se reentrenó el modelo "
        "solo con los sorteos anteriores y se calculó el puntaje y el percentil de la combinación "
        "que después salió. En los sorteos que también están en el registro en vivo da los mismos "
        "números. El pipeline agrega cada sorteo nuevo."
    )
    for juego, tab in zip(JUEGOS, st.tabs([c.NOMBRE[j] for j in JUEGOS]), strict=True):
        g = sim[sim["juego"] == juego].sort_values("n_sorteo")
        if g.empty:
            tab.caption("Sin sorteos simulados de este juego.")
            continue
        s = summarize_percentiles(g["percentil_resultado"].to_numpy())
        n, middle = s["n"], round(s["entre_25_y_75"] * s["n"])
        live_ids = set(rec.loc[rec["juego"] == juego, "n_sorteo"])
        g = g.assign(
            origen=np.where(g["n_sorteo"].isin(live_ids), "registro en vivo", "simulación")
        )
        k1, k2, k3, k4 = tab.columns(4)
        k1.metric("Sorteos simulados", c.es_int(n))
        k2.metric("Percentil medio", f"{s['media']:.0f}", delta="azar: 50", delta_color="off")
        k3.metric(
            "Cayeron entre 25 y 75",
            f"{100 * s['entre_25_y_75']:.0f} %",
            delta="azar: 50 %",
            delta_color="off",
        )
        k4.metric(
            "Cayeron por encima de 90",
            f"{100 * s['sobre_90']:.0f} %",
            delta="azar: 10 %",
            delta_color="off",
        )
        tab.altair_chart(c.percentile_timeline(g), width="stretch")
        counts = np.array(s["deciles"])
        edges = np.arange(0, 101, 10)
        dist = pd.DataFrame(
            {
                "percentil": [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:], strict=True)],
                "observado": counts,
                "esperado": n / 10,
            }
        )
        tab.altair_chart(
            c.observed_vs_expected(
                dist, "percentil", "Percentil de la combinación ganadora", "Sorteos"
            ),
            width="stretch",
        )
        sin_senal = s["p_valor_binomial_mitad"] >= 0.05 and s["p_valor_ks_uniforme"] >= 0.05
        tab.markdown(
            f"""**Cómo leerlo.** Cada punto es un sorteo: arriba, las combinaciones que al modelo le
gustaban; abajo, las que no. La franja gris es la mitad (25 a 75), donde por azar cae la mitad de
los sorteos. En las barras, cada decena de percentiles debería tener unos {c.es(n / 10, 1)} sorteos
(marcas naranjas).

Entre 25 y 75 cayeron **{middle} de {n}**; por azar se esperan {c.es(n / 2, 1)}. Prueba binomial:
p = {c.es(s["p_valor_binomial_mitad"], 3)}. Toda la distribución contra el azar
(Kolmogorov-Smirnov): p = {c.es(s["p_valor_ks_uniforme"], 2)}. """
            + (
                "Con p ≥ 0,05, la diferencia es del tamaño que el azar produce solo: **no hay "
                "evidencia de que las ganadoras caigan más en un lado que en otro.**"
                if sin_senal
                else "Un p < 0,05 aparece por azar una de cada veinte veces; antes de creerlo hay "
                "que ver si se repite en sorteos nuevos y en el otro juego."
            )
        )
        with tab.expander(f"Los {c.es_int(n)} sorteos simulados"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "sorteo": g["n_sorteo"],
                        "fecha": g["fecha"],
                        "resultado": g["resultado"] + " + " + g["superbalota"].astype(str),
                        "puntaje": g["puntaje_resultado"],
                        "percentil": g["percentil_resultado"],
                        "en el registro en vivo": g["origen"]
                        .eq("registro en vivo")
                        .map({True: "sí", False: "no"}),
                    }
                ).iloc[::-1],
                hide_index=True,
                column_config={
                    "puntaje": st.column_config.NumberColumn(format="%+.3f"),
                    "percentil": st.column_config.NumberColumn(format="%.1f"),
                },
            )

st.subheader("Reglas del registro", divider="gray")
st.markdown(
    f"""
- Hay **una sola predicción por sorteo**, registrada antes de las 8:00 p. m. (hora de Colombia)
  del día del sorteo, con un modelo entrenado con todos los resultados conocidos.
- Si los datos están atrasados, el "próximo" sorteo ya ocurrió y **no se registra nada**: sería
  predecir el pasado.
- Los archivos
  [`predictions_log.csv`]({c.REPO_URL}/blob/main/predictions/predictions_log.csv) y
  [`reconciliation_log.csv`]({c.REPO_URL}/blob/main/predictions/reconciliation_log.csv) son
  **append-only**: el CI verifica en cada actualización que ninguna celda anterior cambió (solo
  se agregan filas, o columnas nuevas al final). El historial de git hace de sello de tiempo.
- La referencia es el azar: **{c.es(EXPECTED_HITS)} aciertos por sorteo** jugando 5 números y
  1 superbalota de cada 16. La banda gris marca lo esperable por azar con 95 % de probabilidad.
"""
)
