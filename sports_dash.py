# sports_dash.py – Streamlit dashboard (v6.2) – incluye goles 1er tiempo
# Run: streamlit run sports_dash.py

"""
Cambios v6.2
-------------
• **Exp_1H**: Nueva columna con goles esperados **totales en el 1.º tiempo** (ambos equipos).  
  - Aproximación estándar: 46 % de los goles del partido ocurren antes del descanso.  
• Glosario actualizado para explicar Exp_1H.
• Para bloques UEFA/WNBA la columna muestra “—” (sin modelo detallado aún).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, re, os
from datetime import date, datetime, timezone

st.set_page_config(page_title="Daily Sports Predictions", layout="centered")

# -----------------------------------------------------------------------------
# Parámetros y utilidades
# -----------------------------------------------------------------------------

TODAY = st.sidebar.date_input("Fecha de análisis", value=date.today())
DATE_STR = TODAY.strftime("%Y-%m-%d")

@st.cache_data(ttl=4*60*60)
def fetch_json(url):
    try:
        return requests.get(url,timeout=15).json()
    except Exception:
        return {}

FIRST_HALF_FACTOR = 0.46  # proporción promedio de goles anotados en 1H (FIFA stats)

# -----------------------------------------------------------------------------
# Backup CONMEBOL data con tarjetas totales
# -----------------------------------------------------------------------------
STATIC_CONMEBOL = pd.DataFrame({
    "Team": ["Argentina","Ecuador","Paraguay","Brazil","Colombia","Uruguay","Venezuela","Bolivia","Peru","Chile"],
    "MP": [15]*10,
    "GF": [27,13,13,20,18,17,15,14,6,9],
    "GA": [8,5,9,16,14,12,17,32,17,22],
    "Cards": [2.6,2.8,3.0,2.9,3.1,2.7,3.2,3.4,3.3,3.1],
    "COR": [5.2,4.8,4.9,6.1,5.4,5.0,4.7,3.9,4.2,4.5]
})

FIXTURES_CONMEBOL = [
    {"Kickoff":"18:00","Home":"Argentina","Away":"Colombia"},
    {"Kickoff":"18:00","Home":"Brazil","Away":"Paraguay"},
    {"Kickoff":"18:00","Home":"Bolivia","Away":"Chile"},
    {"Kickoff":"18:00","Home":"Peru","Away":"Ecuador"},
]

# -----------------------------------------------------------------------------
# Poisson – CONMEBOL con Exp_1H
# -----------------------------------------------------------------------------

def conmebol_predictions():
    df = STATIC_CONMEBOL.copy().set_index("Team")
    alpha=1.5
    league_avg=((df["GF"]+alpha).sum()/(df["MP"]+alpha).sum())/2
    df["atk"]=((df["GF"]+alpha)/(df["MP"]+alpha))/league_avg
    df["def"]=((df["GA"]+alpha)/(df["MP"]+alpha))/league_avg

    def pmf(l,k):
        return math.exp(-l)*(l**k)/math.factorial(k)

    rows=[]
    for fx in FIXTURES_CONMEBOL:
        h,a=fx['Home'],fx['Away']
        lam_h=league_avg*df.at[h,'atk']*df.at[a,'def']*1.12
        lam_a=league_avg*df.at[a,'atk']*df.at[h,'def']
        lam_tot=lam_h+lam_a
        exp_1h=round(lam_tot*FIRST_HALF_FACTOR,2)
        p_home=p_draw=p_away=p_o25=0
        for i in range(7):
            for j in range(7):
                p=pmf(lam_h,i)*pmf(lam_a,j)
                if i>j: p_home+=p
                elif i==j: p_draw+=p
                else: p_away+=p
                if i+j>2: p_o25+=p
        rows.append({
            "Kickoff":fx['Kickoff'],
            "Home":h,"Away":a,
            "Exp_H":round(lam_h,2),"Exp_A":round(lam_a,2),
            "Exp_1H":exp_1h,
            "P(H)":round(p_home,3),"P(D)":round(p_draw,3),"P(A)":round(p_away,3),
            "P(>2.5)":round(p_o25,3),
            "Avg_Cards":round((df.at[h,'Cards']+df.at[a,'Cards'])/2,2),
            "Avg_COR":round((df.at[h,'COR']+df.at[a,'COR'])/2,1)
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# UEFA / WNBA placeholders incluyen columna Exp_1H="—"
# -----------------------------------------------------------------------------

def uefa_predictions():
    return pd.DataFrame([{"Kickoff":"—","Home":"Finland","Away":"Poland","Exp_1H":"—","P(H)":0.17,"P(D)":0.28,"P(A)":0.55,"Avg_Cards":"—","Avg_COR":"—"}])

def wnba_predictions():
    return pd.DataFrame([{"Kickoff":"—","Home":"Atlanta Dream","Away":"Indiana Fever","Exp_1H":"—","P(Home)":0.64,"P(Away)":0.36}])

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

MODS={"Fútbol – Sudamérica":conmebol_predictions,
      "Fútbol – Europa (UEFA)":uefa_predictions,
      "Basketball – WNBA":wnba_predictions}
sel=st.sidebar.radio("Bloque:",list(MODS.keys()))

if st.sidebar.button("🔄 Refrescar cuotas ahora"):
    st.cache_data.clear(); st.rerun()

st.title(f"Predicciones • {DATE_STR}")

df=MODS[sel]()
if df.empty:
    st.warning("Sin datos disponibles.")
else:
    st.dataframe(df,use_container_width=True,hide_index=True)
    if sel.startswith("Fútbol"):
        st.markdown("""**Kickoff:** hora local UTC. Campos '—' indican dato no disponible.""")
        st.markdown(
            """
            **Glosario columnas**  
            • **Exp_H / Exp_A**: Goles esperados de local y visita.  
            • **Exp_1H**: Goles esperados **totales en el 1.º tiempo** (≈46 % del total).  
            • **P(H) / P(D) / P(A)**: Probabilidades de 1X2.  
            • **P(>2.5)**: Probabilidad de 3+ goles en el partido.  
            • **Avg_Cards**: Tarjetas totales (amarillas + rojas) promedio.  
            • **Avg_COR**: Córners promedio.  
            """
        )

st.caption("© 2025 – Demo v6.2: incluye goles esperados al descanso.")
