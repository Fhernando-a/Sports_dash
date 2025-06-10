# sports_dash.py – Streamlit dashboard (resilient v3)
# Run with:  streamlit run sports_dash.py

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, re
from datetime import date, datetime

st.set_page_config(page_title="Daily Sports Predictions", layout="wide")
TODAY = date.today().strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@st.cache_data(ttl=4*60*60)
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

@st.cache_data(ttl=4*60*60)
def fetch_json(url: str) -> dict:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# FALLBACK STATIC DATA  (garantiza que siempre se muestre algo)
# -----------------------------------------------------------------------------

STATIC_CONMEBOL = pd.DataFrame({
    "Team": ["Argentina", "Ecuador", "Paraguay", "Brazil", "Colombia", "Uruguay", "Venezuela", "Bolivia", "Peru", "Chile"],
    "MP":   [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
    "GF":   [27, 13, 13, 20, 18, 17, 15, 14, 6, 9],
    "GA":   [ 8,  5,  9, 16, 14, 12, 17, 32,17,22]
})

FIXTURES_CONMEBOL = [
    {"Home": "Argentina", "Away": "Colombia"},
    {"Home": "Brazil",    "Away": "Paraguay"},
    {"Home": "Bolivia",   "Away": "Chile"},
    {"Home": "Peru",      "Away": "Ecuador"},
]

# -----------------------------------------------------------------------------
# 1. Poisson model – CONMEBOL
# -----------------------------------------------------------------------------

def conmebol_predictions():
    wiki = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(CONMEBOL)"
    standings = fetch_table(wiki, r"GF|Goals for|GF_")
    if standings.empty or 'GF' not in standings.columns:
        standings = STATIC_CONMEBOL.copy()

    standings[["MP","GF","GA"]] = standings[["MP","GF","GA"]].apply(pd.to_numeric, errors="coerce")
    standings = standings.dropna()
    standings = standings.set_index("Team")

    alpha = 1.5
    league_avg = ((standings["GF"]+alpha).sum() / (standings["MP"]+alpha).sum()) / 2
    standings["atk"] = ((standings["GF"]+alpha)/(standings["MP"]+alpha)) / league_avg
    standings["def"] = ((standings["GA"]+alpha)/(standings["MP"]+alpha)) / league_avg

    def pmf(l,k):
        return math.exp(-l)*(l**k)/math.factorial(k)

    rows=[]
    for fx in FIXTURES_CONMEBOL:
        h,a = fx['Home'], fx['Away']
        if h not in standings.index or a not in standings.index:
            continue
        λh = league_avg*standings.at[h,'atk']*standings.at[a,'def']*1.12
        λa = league_avg*standings.at[a,'atk']*standings.at[h,'def']
        ph = p_draw = pa = po25 = 0.0
        for i in range(7):
            for j in range(7):
                p=pmf(λh,i)*pmf(λa,j)
                if i>j: ph+=p
                elif i == j:
                p_draw += p
                else: pa+=p
                if i+j>2: po25+=p
        rows.append({"Home":h,"Away":a,"Exp_H":round(λh,2),"Exp_A":round(λa,2),
                     "P(H)":round(ph,3),"P(D)": round(p_draw, 3),"P(A)":round(pa,3),"P(>2.5)":round(po25,3)})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. UEFA (simple fallback -> static sample if API empty)
# -----------------------------------------------------------------------------

def uefa_predictions():
    url=f"https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/scoreboard?dates={TODAY}"
    data=fetch_json(url)
    ev=data.get('events',[])
    if not ev:
        return pd.DataFrame([{"Home":"Finland","Away":"Poland","P(Home)":0.17,"P(Away)":0.83}])
    rows=[]
    for e in ev:
        c=e['competitions'][0]
        if c.get('type',{}).get('slug')!='uefa':
            continue
        h,a=c['competitors']
        home=h if h['homeAway']=='home' else a
        away=a if h['homeAway']=='home' else h
        rows.append({"Home":home['team']['shortDisplayName'],"Away":away['team']['shortDisplayName'],
                     "P(Home)":0.5,"P(Away)":0.5})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 3. WNBA (fallback sample)
# -----------------------------------------------------------------------------

def wnba_predictions():
    url=f"https://site.web.api.espn.com/apis/v2/sports/basketball/wnba/scoreboard?dates={TODAY}"
    data=fetch_json(url)
    ev=data.get('events',[])
    if not ev:
        return pd.DataFrame([{"Home":"Atlanta Dream","Away":"Indiana Fever","P(Home)":0.64,"P(Away)":0.36}])
    rows=[]
    for e in ev:
        c=e['competitions'][0]
        h=next(t for t in c['competitors'] if t['homeAway']=='home')
        a=next(t for t in c['competitors'] if t['homeAway']=='away')
        rows.append({"Home":h['team']['shortDisplayName'],"Away":a['team']['shortDisplayName'],
                     "P(Home)":0.5,"P(Away)":0.5})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

MODS={"Fútbol – Sudamérica":conmebol_predictions,
      "Fútbol – Europa (UEFA)":uefa_predictions,
      "Basketball – WNBA":wnba_predictions}

sel=st.sidebar.radio("Bloque:",list(MODS.keys()))

st.title(f"Predicciones • {datetime.today().strftime('%d %B %Y')}")

df=MODS[sel]()
if df.empty:
    st.warning("Sin datos disponibles (ni en vivo ni fallback).")
else:
    st.dataframe(df,use_container_width=True,hide_index=True)

st.caption("©2025 – Ejemplo con fallback estático si las fuentes en vivo no responden.")
