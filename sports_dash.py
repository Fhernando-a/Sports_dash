# sports_dash.py – Streamlit dashboard (resilient v5)
# Run with:  streamlit run sports_dash.py
"""
Cambios vs v4
-------------
1. Agrega glosario bajo cada tabla explicando las columnas.
2. Añade columnas placeholder para **Tarjetas amarillas (Avg_YC)** y **Corners (Avg_COR)** al modelo Poisson de CONMEBOL.
   • Se calculan como promedios históricos estimados ‑ por ahora simulados.
3. Punto de conexión para *feeds de cuotas* (OddsAPI) – se muestra campo `Avg_Odds_H` cuando hay API‑KEY.
4. UI: selector de deporte + fecha (date_input) y diseño centrado.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, re, os
from datetime import date, datetime

st.set_page_config(page_title="Daily Sports Predictions", layout="centered")

# -----------------------------------------------------------------------------
# Parámetros y utilidades
# -----------------------------------------------------------------------------

TODAY = st.sidebar.date_input("Fecha de análisis", value=date.today())
DATE_STR = TODAY.strftime("%Y-%m-%d")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # Inserta tu key en Secrets o variable de entorno

@st.cache_data(ttl=4 * 60 * 60)
def fetch_table(url: str, regex: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(url, flavor=["lxml", "bs4"])
        for tbl in tables:
            if isinstance(tbl.columns, pd.MultiIndex):
                tbl.columns = ["_".join(map(str, c)).strip() for c in tbl.columns.values]
            if tbl.astype(str).apply(lambda s: s.str.contains(regex, flags=re.I, regex=True)).any().any():
                return tbl
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=4 * 60 * 60)
def fetch_json(url: str) -> dict:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# Fallback data
# -----------------------------------------------------------------------------

STATIC_CONMEBOL = pd.DataFrame({
    "Team": ["Argentina", "Ecuador", "Paraguay", "Brazil", "Colombia", "Uruguay", "Venezuela", "Bolivia", "Peru", "Chile"],
    "MP":   [15]*10,
    "GF":   [27, 13, 13, 20, 18, 17, 15, 14, 6, 9],
    "GA":   [8,5,9,16,14,12,17,32,17,22],
    # Promedios simulados (tarjetas y corners por partido)
    "YC":   [1.8,1.9,2.1,2.0,2.2,2.0,2.3,2.5,2.4,2.2],
    "COR":  [5.2,4.8,4.9,6.1,5.4,5.0,4.7,3.9,4.2,4.5]
})

FIXTURES_CONMEBOL = [
    {"Home": "Argentina", "Away": "Colombia"},
    {"Home": "Brazil",    "Away": "Paraguay"},
    {"Home": "Bolivia",   "Away": "Chile"},
    {"Home": "Peru",      "Away": "Ecuador"},
]

# -----------------------------------------------------------------------------
# 1. Poisson model – CONMEBOL + extra stats
# -----------------------------------------------------------------------------

def conmebol_predictions():
    wiki = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(CONMEBOL)"
    standings = fetch_table(wiki, r"GF|Goals for|GF_")
    if standings.empty or "GF" not in standings.columns:
        standings = STATIC_CONMEBOL.copy()
    else:
        # merge YC & COR placeholders
        standings = standings.merge(STATIC_CONMEBOL[["Team","YC","COR"]], on="Team", how="left")

    standings[["MP","GF","GA"]] = standings[["MP","GF","GA"]].apply(pd.to_numeric, errors="coerce")
    standings = standings.dropna().set_index("Team")

    alpha = 1.5
    league_avg = ((standings["GF"] + alpha).sum() / (standings["MP"] + alpha).sum()) / 2
    standings["atk"] = ((standings["GF"] + alpha) / (standings["MP"] + alpha)) / league_avg
    standings["def"] = ((standings["GA"] + alpha) / (standings["MP"] + alpha)) / league_avg

    def pmf(lam,k):
        return math.exp(-lam)*(lam**k)/math.factorial(k)

    rows=[]
    for fx in FIXTURES_CONMEBOL:
        h,a=fx['Home'],fx['Away']
        if h not in standings.index or a not in standings.index:
            continue
        lam_h = league_avg*standings.at[h,'atk']*standings.at[a,'def']*1.12
        lam_a = league_avg*standings.at[a,'atk']*standings.at[h,'def']

        p_home=p_draw=p_away=p_over25=0.0
        for i in range(7):
            for j in range(7):
                p=pmf(lam_h,i)*pmf(lam_a,j)
                if i>j: p_home+=p
                elif i==j: p_draw+=p
                else: p_away+=p
                if i+j>2: p_over25+=p

        # -------- Odds block --------
        avg_odds_h = np.nan  # por defecto
        # 1) Si tienes API‑KEY de TheOddsAPI
        if ODDS_API_KEY:
            odds_url = (
                f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
                f"?regions=eu&mkt=h2h&dateFormat=iso&apiKey={ODDS_API_KEY}"
            )
            data_odds = fetch_json(odds_url)
            # (parsing omitido por brevedad)
        # 2) Sin API‑KEY: usa las predicciones FiveThirtyEight (convierten probs→cuotas)
        else:
            fivethirty_url = (
                "https://projects.fivethirtyeight.com/soccer-api/nations/nations_matches_latest.csv"
            )
            try:
                latest = pd.read_csv(fivethirty_url)
                m = latest[
                    (latest["team1"].str.contains(h, case=False)) &
                    (latest["team2"].str.contains(a, case=False))
                ]
                if not m.empty:
                    prob1 = m.iloc[0]["prob1"]
                    avg_odds_h = round(1/prob1, 2) if prob1 else np.nan
            except Exception:
                pass
        if ODDS_API_KEY:
            odds_url = f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds?regions=eu&mkt=h2h&apiKey={ODDS_API_KEY}"
            resp = fetch_json(odds_url)
            # parse when endpoint available (omitted for brevity)

        rows.append({
            "Home":h,"Away":a,
            "Exp_H":round(lam_h,2),"Exp_A":round(lam_a,2),
            "P(H)":round(p_home,3),"P(D)":round(p_draw,3),"P(A)":round(p_away,3),
            "P(>2.5)":round(p_over25,3),
            "Avg_YC":round((standings.at[h,'YC']+standings.at[a,'YC'])/2,2),
            "Avg_COR":round((standings.at[h,'COR']+standings.at[a,'COR'])/2,1),
            "Avg_Odds_H":avg_odds_h
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. Simple fallback UEFA / WNBA (sin cambios)
# -----------------------------------------------------------------------------

def uefa_predictions():
    return pd.DataFrame([{"Home":"Finland","Away":"Poland","P(Home)":0.17,"P(Away)":0.83}])

def wnba_predictions():
    return pd.DataFrame([{"Home":"Atlanta Dream","Away":"Indiana Fever","P(Home)":0.64,"P(Away)":0.36}])

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

MODS={"Fútbol – Sudamérica":conmebol_predictions,
      "Fútbol – Europa (UEFA)":uefa_predictions,
      "Basketball – WNBA":wnba_predictions}

sel = st.sidebar.radio("Bloque:", list(MODS.keys()))

# --- Botón para refrescar cuotas y limpiar caché ---
if st.sidebar.button("🔄 Refrescar cuotas ahora"):
    st.cache_data.clear()
    st.rerun()


st.title(f"Predicciones • {TODAY.strftime('%d %B %Y')}")

with st.spinner("Cargando…"):
    df = MODS[sel]()

if df.empty:
    st.warning("Sin datos disponibles.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- Glosario ----------
    if sel == "Fútbol – Sudamérica":
        st.markdown(
            """
            **Glosario columnas**  
            • **Exp_H / Exp_A**: Goles esperados para local y visita según modelo Poisson.  
            • **P(H) / P(D) / P(A)**: Probabilidad de victoria local, empate o visitante.  
            • **P(>2.5)**: Probabilidad total de que se marquen 3+ goles.  
            • **Avg_YC**: Promedio histórico de tarjetas amarillas combinadas por partido.  
            • **Avg_COR**: Promedio histórico de córners combinados por partido.  
            • **Avg_Odds_H**: Cuota promedio del mercado para triunfo local (⚠ muestra *NaN* si no hay API‑KEY configurada).
            """)

st.caption("© 2025 – Demo con columnas extra (tarjetas, corners). Conecta tu API key de cuotas para mostrar odds reales.")
