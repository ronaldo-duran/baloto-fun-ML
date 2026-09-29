"""Página 2: registro de predicciones en vivo (aciertos acumulados del modelo vs. azar)."""

import common as c
import numpy as np
import pandas as pd
import streamlit as st

from baloto_ml.config import EXPECTED_HITS, JUEGOS, P_SUPER
from baloto_ml.data.calendar import COT
from baloto_ml.evaluation.stats import hits_variance

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
        m1, m2, m3 = st.columns(3)
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
                }
            ).iloc[::-1],
            hide_index=True,
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
  **append-only**: el CI verifica en cada actualización que las filas anteriores no cambiaron. El
  historial de git hace de sello de tiempo.
- La referencia es el azar: **{c.es(EXPECTED_HITS)} aciertos por sorteo** jugando 5 números y
  1 superbalota de cada 16. La banda gris marca lo esperable por azar con 95 % de probabilidad.
"""
)
