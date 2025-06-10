# sports_dash.py – Streamlit dashboard with live web‑fetched data (robust parsing)
# Run with:  streamlit run sports_dash.py

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, re
from datetime import date, datetime

st.set_page_config(page_title="Daily Sports Predictions", layout="wide")
TODAY = date.today().strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------
# Helpers for web fetching & caching
# -----------------------------------------------------------------------------

@st.cache_data(ttl=4 * 60 * 60)
def fetch_table(url: str, regex: str) -> pd.DataFrame:
    """Return first HTML table in *url* whose text matches *regex* (case‑insensitive)."""
    try:
        tables = pd.read_html(url, flavor=["lxml", "bs4"])
        for tb in tables:
            if tb.astype(str).apply(lambda s: s.str.contains(regex, flags=re.I)).any().any():
                return tb
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Error fetching {url}: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl=4 * 60 * 60)
def fetch_json(url: str) -> dict:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Error fetching {url}: {exc}")
        return {}

# -----------------------------------------------------------------------------
# 1. FOOTBALL – CONMEBOL (Poisson hierarchical)
# -----------------------------------------------------------------------------

def conmebol_predictions():
    wiki = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(CONMEBOL)"
    raw = fetch_table(wiki, r"Team|Equipo")
    if raw.empty:
        st.error("No se pudo leer la tabla de posiciones CONMEBOL.")
        return pd.DataFrame()

    # Detect column names robustamente
    def col_like(df, *cands):
        for c in cands:
            matches = [col for col in df.columns if col.lower() == c.lower()]
            if matches:
                return matches[0]
        return None

    team_c = col_like(raw, "Team", "Teams", "Equipo")
    mp_c   = col_like(raw, "Pld", "MP", "Played", "J")
    gf_c   = col_like(raw, "GF", "F", "Goals for")
    ga_c   = col_like(raw, "GA", "A", "Goals against")

    if None in (team_c, mp_c, gf_c, ga_c):
        st.error("No se encontraron todas las columnas necesarias en la tabla CONMEBOL.")
        return pd.DataFrame()

    standings = raw[[team_c, mp_c, gf_c, ga_c]].rename(columns={
        team_c: "Team", mp_c: "MP", gf_c: "GF", ga_c: "GA"})

    standings[["MP", "GF", "GA"]] = standings[["MP", "GF", "GA"]].apply(pd.to_numeric, errors="coerce")
    standings = standings.dropna()

    # Intentar extraer fixtures de hoy; si falla usar manual
    fx_raw = fetch_table(wiki, TODAY.split("-")[2] + r"\s+" + datetime.today().strftime("%B"))
    fixtures = []
    if not fx_raw.empty and fx_raw.shape[1] >= 3:
        for _, r in fx_raw.iterrows():
            fixtures.append({"Home": str(r[0]).strip(), "Away": str(r[2]).strip()})
    if not fixtures:
        fixtures = [
            {"Home": "Argentina", "Away": "Colombia"},
            {"Home": "Brazil", "Away": "Paraguay"},
            {"Home": "Bolivia", "Away": "Chile"},
            {"Home": "Peru", "Away": "Ecuador"},
        ]

    # Poisson model
    alpha = 1.5
    league_avg = ((standings["GF"] + alpha).sum() / (standings["MP"] + alpha).sum()) / 2
    standings = standings.set_index("Team")
    standings["attack"]  = ((standings["GF"] + alpha) / (standings["MP"] + alpha)) / league_avg
    standings["defence"] = ((standings["GA"] + alpha) / (standings["MP"] + alpha)) / league_avg

    def pmf(lmbd, k):
        return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)

    rows = []
    for fx in fixtures:
        h, a = fx["Home"], fx["Away"]
        if h not in standings.index or a not in standings.index:
            continue
        λh = league_avg * standings.loc[h, "attack"] * standings.loc[a, "defence"] * 1.12
        λa = league_avg * standings.loc[a, "attack"] * standings.loc[h, "defence"]
        ph = pd = pa = po25 = 0.0
        for i in range(7):
            for j in range(7):
                p = pmf(λh, i) * pmf(λa, j)
                if i > j: ph += p
                elif i == j: pd += p
                else: pa += p
                if i + j > 2: po25 += p
        rows.append({"Home": h, "Away": a, "Exp_H": round(λh,2), "Exp_A": round(λa,2),
                     "P(H)": round(ph,3), "P(D)": round(pd,3), "P(A)": round(pa,3), "P(>2.5)": round(po25,3)})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 2. FOOTBALL – UEFA (Elo‑logistic)
# -----------------------------------------------------------------------------

def uefa_predictions():
    espn = f"https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/scoreboard?dates={TODAY}"
    data = fetch_json(espn)
    events = data.get("events", [])
    if not events:
        return pd.DataFrame()

    elo_csv = "https://raw.githubusercontent.com/tadhg-ohiggins/fifa-elo-ratings/main/data/fifa_elo_latest.csv"
    elo_df = fetch_table(elo_csv, "elo")

    def elo(t):
        if elo_df.empty: return 1500
        row = elo_df[elo_df["team"] == t]
        return float(row.iloc[0]["elo"]) if not row.empty else 1500

    rec = []
    for ev in events:
        comp = ev["competitions"][0]
        if comp.get("type", {}).get("slug") != "uefa":
            continue
        h, a = comp["competitors"]
        home = h if h["homeAway"] == "home" else a
        away = a if h["homeAway"] == "home" else h
        th, ta = home["team"]["shortDisplayName"], away["team"]["shortDisplayName"]
        ph = 1 / (1 + 10 ** (-(elo(th)+60-elo(ta))/400))
        rec.append({"Home": th, "Away": ta, "P(Home)": round(ph,3), "P(Away)": round(1-ph,3)})
    return pd.DataFrame(rec)

# -----------------------------------------------------------------------------
# 3. BASKET – WNBA (win‑ratio)
# -----------------------------------------------------------------------------

def wnba_predictions():
    url = f"https://site.web.api.espn.com/apis/v2/sports/basketball/wnba/scoreboard?dates={TODAY}"
    data = fetch_json(url)
    events = data.get("events", [])
    out = []
    for ev in events:
        comp = ev["competitions"][0]
        h = next(t for t in comp["competitors"] if t["homeAway"]=="home")
        a = next(t for t in comp["competitors"] if t["homeAway"]=="away")
        def wp(t):
            w,l = map(int, t.get("records",[{"summary":"0-0"}])[0]["summary"].split("-"))
            return w/(w+l+1e-6)
        ph = wp(h)/(wp(h)+wp(a)+1e-6)
        out.append({"Home": h["team"]["shortDisplayName"], "Away": a["team"]["shortDisplayName"],
                    "P(Home)": round(ph,3), "P(Away)": round(1-ph,3)})
    return pd.DataFrame(out)

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------------------------------------------

MODS = {"Fútbol – Sudamérica": conmebol_predictions,
        "Fútbol – Europa (UEFA)": uefa_predictions,
        "Basketball – WNBA": wnba_predictions}

st.sidebar.title("⚽🏀 Predicciones deportivas (en vivo)")
choice = st.sidebar.radio("Selecciona bloque:", list(MODS.keys()))

st.title(f"Predicciones • {datetime.today().strftime('%d %B %Y')}")

with st.spinner("Cargando datos …"):
    df = MODS[choice]()

if df.empty:
    st.warning("No se pudieron obtener datos en vivo para el bloque seleccionado.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "P(H)" in df.columns and (df["P(H)"] >= 0.55).any():
        st.subheader("Posibles edges (P(H) ≥ 55 %)")
        st.table(df[df["P(H)"] >= 0.55][["Home","Away","P(H)"]])

st.caption("© 2025 – Modelos: Poisson (CONMEBOL), Elo (UEFA), win‑ratio (WNBA). Datos Wikipedia & ESPN.")
