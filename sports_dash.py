# sports_dash.py – Streamlit dashboard with live web‑fetched data
# Run with:  streamlit run sports_dash.py
# NOTE: Requires internet access from the host running Streamlit.
# If you are behind a firewall proxy, set env vars HTTP_PROXY / HTTPS_PROXY.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, datetime
from functools import lru_cache

st.set_page_config(page_title="Daily Sports Predictions", layout="wide")
TODAY = date.today().strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------
# 1. Utility helpers
# -----------------------------------------------------------------------------

@st.cache_data(ttl=4 * 60 * 60)  # refresh every 4 h
def fetch_table(url: str, match: str) -> pd.DataFrame:
    """Loads the first HTML table containing *match* regex from *url*."""
    try:
        tbls = pd.read_html(url, match=match)
        return tbls[0]
    except Exception as err:
        st.error(f"Error fetching {url}: {err}")
        return pd.DataFrame()

@st.cache_data(ttl=4 * 60 * 60)
def fetch_json(url: str) -> dict:
    """Simple GET to JSON endpoint with caching."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as err:
        st.error(f"Error fetching {url}: {err}")
        return {}

# -----------------------------------------------------------------------------
# 2. FOOTBALL – CONMEBOL predictions (Poisson hierarchical)
# -----------------------------------------------------------------------------

def conmebol_predictions(match_day: str = TODAY):
    wiki_url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_qualification_(CONMEBOL)"
    standings = fetch_table(wiki_url, "Team")
    if standings.empty:
        return pd.DataFrame()

    # Clean & rename columns
    standings = standings[["Team", "Pld", "GF", "GA"]]
    standings.columns = ["Team", "MP", "GF", "GA"]

    # Convert numeric
    for col in ("MP", "GF", "GA"):
        standings[col] = pd.to_numeric(standings[col], errors="coerce")

    # Fixtures today (scrape the Results section – limited rows)
    fixtures_all = fetch_table(wiki_url, "\d{1,2}\s+June\s+2025")
    if fixtures_all.empty:
        st.warning("No fixtures scraped; falling back to manual list.")
        fixtures = [
            {"Home": "Argentina", "Away": "Colombia"},
            {"Home": "Brazil", "Away": "Paraguay"},
            {"Home": "Bolivia", "Away": "Chile"},
            {"Home": "Peru", "Away": "Ecuador"},
        ]
    else:
        fixtures = []
        for _, row in fixtures_all.iterrows():
            h, a = str(row[0]), str(row[2])  # table format: Home, score, Away
            fixtures.append({"Home": h.replace("\xa0", " "), "Away": a.replace("\xa0", " ")})

    alpha = 1.5  # Laplace smoothing
    league_GF = (standings["GF"] + alpha).sum()
    league_MP = (standings["MP"] + alpha).sum()
    league_avg = (league_GF / league_MP) / 2  # goals per team‑match

    standings = standings.set_index("Team")
    standings["attack"] = ((standings["GF"] + alpha) / (standings["MP"] + alpha)) / league_avg
    standings["defence"] = ((standings["GA"] + alpha) / (standings["MP"] + alpha)) / league_avg

    def poisson_pmf(lmbd, k):
        return np.exp(-lmbd) * (lmbd ** k) / math.factorial(k)

    rows = []
    for fx in fixtures:
        h, a = fx["Home"], fx["Away"]
        if h not in standings.index or a not in standings.index:
            continue
        lh = league_avg * standings.loc[h, "attack"] * standings.loc[a, "defence"] * 1.12
        la = league_avg * standings.loc[a, "attack"] * standings.loc[h, "defence"]
        # Outcome probs via truncated Poisson convolution
        p_home = p_draw = p_away = p_over25 = 0.0
        for i in range(7):
            for j in range(7):
                p = poisson_pmf(lh, i) * poisson_pmf(la, j)
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p
                if i + j > 2:
                    p_over25 += p
        rows.append({
            "Home": h, "Away": a,
            "ExpGoals_H": round(lh, 2), "ExpGoals_A": round(la, 2),
            "P(Home)": round(p_home, 3), "P(Draw)": round(p_draw, 3),
            "P(Away)": round(p_away, 3), "P(>2.5)": round(p_over25, 3),
        })

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 3. FOOTBALL – UEFA predictions (Elo logistic)
# -----------------------------------------------------------------------------

def uefa_predictions(date_str: str = TODAY):
    # ESPN scoreboard often lists all UEFA qualifiers under soccer/fifa.world
    espn_url = f"https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/scoreboard?dates={date_str}"
    data = fetch_json(espn_url)
    events = data.get("events", [])
    if not events:
        st.warning("No UEFA events fetched; sample data will be shown.")
        sample = pd.DataFrame([
            {"Home": "Finland", "Away": "Poland", "P(Home)": 0.17, "P(Away)": 0.83},
            {"Home": "Netherlands", "Away": "Malta", "P(Home)": 0.991, "P(Away)": 0.009},
        ])
        return sample

    # Fetch Elo ratings table (static CSV hosted on GitHub)
    elo_url = "https://raw.githubusercontent.com/tadhg-ohiggins/fifa-elo-ratings/main/data/fifa_elo_latest.csv"
    elo_df = fetch_table(elo_url, "Team")
    if elo_df.empty:
        st.warning("Elo ratings unavailable; falling back to equal strength.")

    def elo(team):
        if elo_df.empty:
            return 1500
        row = elo_df[elo_df["team"] == team]
        if row.empty:
            return 1500
        return float(row.iloc[0]["elo"])

    rows = []
    for ev in events:
        comp = ev["competitions"][0]
        if comp.get("type", {}).get("slug") != "uefa":
            continue  # skip non‑UEFA matches
        h, a = comp["competitors"]
        home = h if h["homeAway"] == "home" else a
        away = a if h["homeAway"] == "home" else h
        home_team = home["team"]["shortDisplayName"]
        away_team = away["team"]["shortDisplayName"]
        ph = 1 / (1 + 10 ** (-(elo(home_team) + 60 - elo(away_team)) / 400))
        rows.append({"Home": home_team, "Away": away_team, "P(Home)": round(ph, 3), "P(Away)": round(1 - ph, 3)})

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 4. BASKETBALL – WNBA predictions (record‑based quick model)
# -----------------------------------------------------------------------------

def wnba_predictions(date_str: str = TODAY):
    url = f"https://site.web.api.espn.com/apis/v2/sports/basketball/wnba/scoreboard?dates={date_str}"
    data = fetch_json(url)
    events = data.get("events", [])
    rows = []
    for ev in events:
        comp = ev["competitions"][0]
        comp_team = comp["competitors"]
        home = next(t for t in comp_team if t["homeAway"] == "home")
        away = next(t for t in comp_team if t["homeAway"] == "away")
        def win_pct(team_json):
            rec = team_json.get("records", [{"summary": "0-0"}])[0]["summary"]
            w, l = map(int, rec.split("-"))
            return w / (w + l + 1e-9)
        wp_h, wp_a = win_pct(home), win_pct(away)
        ph = wp_h / (wp_h + wp_a + 1e-9)
        rows.append({
            "Home": home["team"]["displayName"],
            "Away": away["team"]["displayName"],
            "P(Home)": round(ph, 3),
            "P(Away)": round(1 - ph, 3),
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI
# -----------------------------------------------------------------------------

SPORT_FUNCS = {
    "Fútbol – Sudamérica": conmebol_predictions,
    "Fútbol – Europa (UEFA)": uefa_predictions,
    "Basketball – WNBA": wnba_predictions,
}

st.sidebar.title("⚽🏀 Daily Sports Predictions (Live)")
sel = st.sidebar.radio("Selecciona deporte/ámbito:", list(SPORT_FUNCS.keys()))

st.title(f"Predicciones • {datetime.today().strftime('%d %B %Y')}")

with st.spinner("Cargando datos en vivo…"):
    df = SPORT_FUNCS[sel]()

if df.empty:
    st.error("No se pudieron obtener datos en vivo.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)
    # Edge detector ≥55 %
    if "P(Home)" in df.columns:
        edge_df = df[df["P(Home)"] >= 0.55]
        if not edge_df.empty:
            st.subheader("Edges potenciales (P(Home) ≥ 55 %)")
            st.table(edge_df[["Home", "Away", "P(Home)"]])
        else:
            st.info("Sin edges claros ≥55 % para partidos listados.")

st.caption("© 2025 – Datos en vivo vía Wikipedia & ESPN. Modelo Poisson jerárquico (CONMEBOL), Elo simple (UEFA) y record‑ratio (WNBA). Sustituye por tus feeds/API comerciales para mayor precisión.")
