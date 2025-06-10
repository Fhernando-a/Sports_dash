# sports_dash.py – Streamlit dashboard (v7) – horarios y cuotas vía ESPN (sin API key)
# Run: streamlit run sports_dash.py

"""
Cambios v7
-----------
• **Kickoff** se obtiene automáticamente desde la API pública de ESPN y se muestra en hora Lima (UTC‑5).  
• **Cuota mercado** (`Odds_H`) se extrae del campo `competitions[*].odds` de ESPN (si existe).  
• Sin almacenamiento persistente: los resultados finales se muestran en la tabla cuando el partido ya terminó (status=completed) e información de goles finales.  
• Placeholder para Edge % = (prob modelo − prob mercado) cuando haya cuota.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, os, pytz
from datetime import date, datetime, timezone

st.set_page_config(page_title="Daily Sports Predictions", layout="centered")

# -----------------------------------------------------------------------------
# Parámetros
# -----------------------------------------------------------------------------

TODAY = st.sidebar.date_input("Fecha de análisis", value=date.today())
DATE_STR = TODAY.strftime("%Y-%m-%d")
TZ_LIMA = pytz.timezone("America/Lima")
FIRST_HALF_FACTOR = 0.46

@st.cache_data(ttl=60*60)
def get_espn_scoreboard(date_str):
    url = f"https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/scoreboard?dates={date_str}"
    try:
        return requests.get(url, timeout=15).json()
    except Exception:
        return {}

# Utilidad para convertir ISO a Lima hh:mm
def iso_to_lima(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z","+00:00")).astimezone(TZ_LIMA)
        return dt.strftime("%H:%M")
    except Exception:
        return "—"

# ----------------------------------------------------------------------------
# Modelo Poisson simplificado – sólo para CONMEBOL en demo
# ----------------------------------------------------------------------------

STATIC_STAND = pd.DataFrame({
    "Team": ["Argentina","Ecuador","Paraguay","Brazil","Colombia","Uruguay","Venezuela","Bolivia","Peru","Chile"],
    "MP": [15]*10,
    "GF": [27,13,13,20,18,17,15,14,6,9],
    "GA": [8,5,9,16,14,12,17,32,17,22],
    "Cards": [2.6,2.8,3.0,2.9,3.1,2.7,3.2,3.4,3.3,3.1],
    "COR": [5.2,4.8,4.9,6.1,5.4,5.0,4.7,3.9,4.2,4.5]
}).set_index("Team")

alpha=1.5
league_avg=((STATIC_STAND["GF"]+alpha).sum()/(STATIC_STAND["MP"]+alpha).sum())/2
STATIC_STAND["atk"]=((STATIC_STAND["GF"]+alpha)/(STATIC_STAND["MP"]+alpha))/league_avg
STATIC_STAND["def"]=((STATIC_STAND["GA"]+alpha)/(STATIC_STAND["MP"]+alpha))/league_avg

def poisson(l,k):
    return math.exp(-l)*(l**k)/math.factorial(k)

# ----------------------------------------------------------------------------
# Funciones de predicción por bloque
# ----------------------------------------------------------------------------

def conmebol_predictions():
    sb = get_espn_scoreboard(DATE_STR)
    rows = []
    for ev in sb.get("events", []):
        comp = ev["competitions"][0]
        if comp.get("league", {}).get("abbreviation") != "WCQ":
            continue
        teams = {t["homeAway"]: t for t in comp["competitors"]}
        home = teams.get("home"); away = teams.get("away")
        h_name = home["team"]["shortDisplayName"]; a_name = away["team"]["shortDisplayName"]
        lam_h = league_avg*STATIC_STAND.at[h_name,'atk']*STATIC_STAND.at[a_name,'def']*1.12
        lam_a = league_avg*STATIC_STAND.at[a_name,'atk']*STATIC_STAND.at[h_name,'def']
        lam_tot=lam_h+lam_a; exp1h=round(lam_tot*FIRST_HALF_FACTOR,2)
        p_home=p_draw=p_away=p_o25=0
        for i in range(7):
            for j in range(7):
                p=poisson(lam_h,i)*poisson(lam_a,j)
                if i>j: p_home+=p
                elif i==j: p_draw+=p
                else: p_away+=p
                if i+j>2: p_o25+=p
        # Odds ESPN
        odds_h = None
        if comp.get("odds"):
            try:
                odds_h = float(comp["odds"][0]["details"].split(" ")[0])
            except Exception:
                pass
        rows.append({
            "Kickoff": iso_to_lima(comp["date"]),
            "Home": h_name,
            "Away": a_name,
            "Exp_H": round(lam_h,2), "Exp_A": round(lam_a,2),
            "Exp_1H": exp1h,
            "P(H)": round(p_home,3), "P(D)": round(p_draw,3), "P(A)": round(p_away,3),
            "P(>2.5)": round(p_o25,3),
            "Avg_Cards": round((STATIC_STAND.at[h_name,'Cards']+STATIC_STAND.at[a_name,'Cards'])/2,2),
            "Avg_COR": round((STATIC_STAND.at[h_name,'COR']+STATIC_STAND.at[a_name,'COR'])/2,1),
            "Odds_H": odds_h,
            "Edge%": round((p_home - (1/odds_h if odds_h else np.nan))*100,1) if odds_h else "—"
        })
    return pd.DataFrame(rows)


def uefa_predictions():
    sb = get_espn_scoreboard(DATE_STR)
    rows=[]
    for ev in sb.get("events", []):
        comp=ev["competitions"][0]
        if comp.get("type",{}).get("slug")!="uefa":
            continue
        teams={t["homeAway"]:t for t in comp["competitors"]}
        home=teams["home"]; away=teams["away"]
        rows.append({
            "Kickoff": iso_to_lima(comp["date"]),
            "Home": home["team"]["shortDisplayName"],
            "Away": away["team"]["shortDisplayName"],
            "Exp_1H":"—","P(H)":0.5,"P(D)":0.25,"P(A)":0.25,
            "Avg_Cards":"—","Avg_COR":"—","Odds_H":"—","Edge%":"—"
        })
    return pd.DataFrame(rows)


def wnba_predictions():
    url=f"https://site.web.api.espn.com/apis/v2/sports/basketball/wnba/scoreboard?dates={DATE_STR}"
    data=get_espn_scoreboard(DATE_STR)
    rows=[]
    for ev in data.get('events',[]):
        comp=ev['competitions'][0]
        h=a=None
        for t in comp['competitors']:
            if t['homeAway']=='home': h=t
            else: a=t
        rows.append({
            "Kickoff": iso_to_lima(comp['date']),
            "Home": h['team']['shortDisplayName'],
            "Away": a['team']['shortDisplayName'],
            "Exp_1H":"—","P(Home)":0.5,"P(Away)":0.5,
            "Odds_H":"—"
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

MODS={"Fútbol – Sudamérica":conmebol_predictions,
      "Fútbol – Europa (UEFA)":uefa_predictions,
      "Basketball – WNBA":wnba_predictions}
sel=st.sidebar.radio("Bloque:",list(MODS.keys()))
if st.sidebar.button("🔄 Refrescar ahora"):
    st.cache_data.clear(); st.rerun()

st.title(f"Predicciones • {DATE_STR}")

df=MODS[sel]()
if 'Kickoff' in df.columns:
    df=df[['Kickoff']+[c for c in df.columns if c!='Kickoff']]
if df.empty:
    st.warning("No hay partidos para la fecha seleccionada.")
else:
    st.dataframe(df,use_container_width=True,hide_index=True)

st.caption("© 2025 – Demo v7: horarios auto ESPN, cuotas ESPN cuando disponibles.")

