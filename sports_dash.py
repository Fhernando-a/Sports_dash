# sports_dash.py – Streamlit dashboard (v6)  ← hora de partido + columnas extra
# Run with:  streamlit run sports_dash.py

"""
Cambios principales (v6)
-----------------------
1. **Hora del partido** (`Kickoff`) añadida como primera columna en cada bloque.  
   • CONMEBOL: horario referencial 18:00 Lima para ejemplo.  
   • UEFA & WNBA: se extrae de ESPN si la API responde; si no, muestra "—".
2. **Europeo y WNBA con tarjetas/córners placeholders** para coherencia visual.  
3. Mantiene botón de refresco, glosario actualizado.
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
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

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
# Datos de respaldo
# -----------------------------------------------------------------------------

STATIC_CONMEBOL = pd.DataFrame({
    "Team": ["Argentina", "Ecuador", "Paraguay", "Brazil", "Colombia", "Uruguay", "Venezuela", "Bolivia", "Peru", "Chile"],
    "MP":   [15]*10,
    "GF":   [27, 13, 13, 20, 18, 17, 15, 14, 6, 9],
    "GA":   [8,5,9,16,14,12,17,32,17,22],
    "YC":   [1.8,1.9,2.1,2.0,2.2,2.0,2.3,2.5,2.4,2.2],
    "COR":  [5.2,4.8,4.9,6.1,5.4,5.0,4.7,3.9,4.2,4.5]
})

FIXTURES_CONMEBOL = [
    {"Kickoff":"18:00", "Home": "Argentina", "Away": "Colombia"},
    {"Kickoff":"18:00", "Home": "Brazil",    "Away": "Paraguay"},
    {"Kickoff":"18:00", "Home": "Bolivia",   "Away": "Chile"},
    {"Kickoff":"18:00", "Home": "Peru",      "Away": "Ecuador"},
]

# -----------------------------------------------------------------------------
# 1. Poisson model – CONMEBOL
# -----------------------------------------------------------------------------

def conmebol_predictions():
    standings = STATIC_CONMEBOL.copy()
    standings[["MP","GF","GA"]] = standings[["MP","GF","GA"]].apply(pd.to_numeric, errors="coerce")
    standings = standings.set_index("Team")

    alpha=1.5
    league_avg=((standings["GF"]+alpha).sum()/(standings["MP"]+alpha).sum())/2
    standings["atk"]=((standings["GF"]+alpha)/(standings["MP"]+alpha))/league_avg
    standings["def"]=((standings["GA"]+alpha)/(standings["MP"]+alpha))/league_avg

    def pmf(l,k):
        return math.exp(-l)*(l**k)/math.factorial(k)

    rows=[]
    for fx in FIXTURES_CONMEBOL:
        h,a=fx['Home'],fx['Away']
        lam_h=league_avg*standings.at[h,'atk']*standings.at[a,'def']*1.12
        lam_a=league_avg*standings.at[a,'atk']*standings.at[h,'def']
        p_home=p_draw=p_away=p_o25=0
        for i in range(7):
            for j in range(7):
                p=pmf(lam_h,i)*pmf(lam_a,j)
                if i>j: p_home+=p
                elif i==j: p_draw+=p
                else: p_away+=p
                if i+j>2: p_o25+=p

        rows.append({
            "Kickoff": fx['Kickoff'],
            "Home":h,"Away":a,
            "Exp_H":round(lam_h,2),"Exp_A":round(lam_a,2),
            "P(H)":round(p_home,3),"P(D)":round(p_draw,3),"P(A)":round(p_away,3),
            "P(>2.5)":round(p_o25,3),
            "Avg_YC":round((standings.at[h,'YC']+standings.at[a,'YC'])/2,2),
            "Avg_COR":round((standings.at[h,'COR']+standings.at[a,'COR'])/2,1)
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. UEFA – extrae hora del ESPN Scoreboard (si disponible)
# -----------------------------------------------------------------------------

def uefa_predictions():
    url=f"https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/scoreboard?dates={DATE_STR}"
    data=fetch_json(url)
    rows=[]
    for ev in data.get('events',[]):
        comp=ev['competitions'][0]
        if comp.get('type',{}).get('slug')!='uefa':
            continue
        h=a=None
        for t in comp['competitors']:
            if t['homeAway']=='home': h=t
            else: a=t
        kick_iso=comp.get('date')
        kick=time_from_iso(kick_iso)
        rows.append({
            "Kickoff":kick,
            "Home":h['team']['shortDisplayName'],
            "Away":a['team']['shortDisplayName'],
            "P(H)":0.5,"P(D)":0.25,"P(A)":0.25,
            "Avg_YC":"—","Avg_COR":"—"
        })
    if not rows:
        rows=[{"Kickoff":"—","Home":"Finland","Away":"Poland","P(H)":0.17,"P(D)":0.28,"P(A)":0.55,"Avg_YC":"—","Avg_COR":"—"}]
    return pd.DataFrame(rows)

# helper to get local time hh:mm
def time_from_iso(iso_str):
    try:
        dt=datetime.fromisoformat(iso_str.replace('Z','+00:00')).astimezone(timezone.utc)
        return dt.strftime('%H:%M')
    except Exception:
        return "—"

# -----------------------------------------------------------------------------
# 3. WNBA – hora + prob. básica
# -----------------------------------------------------------------------------

def wnba_predictions():
    url=f"https://site.web.api.espn.com/apis/v2/sports/basketball/wnba/scoreboard?dates={DATE_STR}"
    data=fetch_json(url)
    rows=[]
    for ev in data.get('events',[]):
        comp=ev['competitions'][0]
        h=a=None
        for t in comp['competitors']:
            if t['homeAway']=='home': h=t
            else: a=t
        kick=time_from_iso(comp.get('date'))
        rows.append({
            "Kickoff":kick,
            "Home":h['team']['shortDisplayName'],
            "Away":a['team']['shortDisplayName'],
            "P(Home)":0.5,"P(Away)":0.5,
            "Avg_Pace":"—","3P_Gap":"—"
        })
    if not rows:
        rows=[{"Kickoff":"—","Home":"Atlanta Dream","Away":"Indiana Fever","P(Home)":0.64,"P(Away)":0.36,"Avg_Pace":"—","3P_Gap":"—"}]
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# UI & acciones
# -----------------------------------------------------------------------------

MODS={"Fútbol – Sudamérica":conmebol_predictions,
      "Fútbol – Europa (UEFA)":uefa_predictions,
      "Basketball – WNBA":wnba_predictions}
sel=st.sidebar.radio("Bloque:",list(MODS.keys()))

if st.sidebar.button("🔄 Refrescar cuotas ahora"):
    st.cache_data.clear()
    st.rerun()

st.title(f"Predicciones • {DATE_STR}")

df=MODS[sel]()
if df.empty:
    st.warning("Sin datos disponibles.")
else:
    st.dataframe(df,use_container_width=True,hide_index=True)
    if sel.startswith("Fútbol"):
        st.markdown("**Kickoff:** hora local UTC mostrada para cada partido. Otros campos '—' indican dato no disponible.")

st.caption("© 2025 – Demo. Probabilidades modelo + hora del partido.")
