import streamlit as st
import json
import sqlite3
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
from datetime import datetime, date
import pickle

# Import predict module (handles model reloading internally)
import predict

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLES
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  section[data-testid="stSidebar"] {
    background: #0d1f0f;
    border-right: 1px solid #1e3a21;
  }
  section[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
  .main { background: #0a0a0a; color: #e8e8e8; }
  .metric-card {
    background: #111;
    border: 1px solid #1e3a21;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    text-align: center;
  }
  .metric-card .label { font-size: 0.70rem; letter-spacing: 0.1em; text-transform: uppercase; color: #4caf50; margin-bottom: 0.35rem; }
  .metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; color: #e8e8e8; }
  .prob-bar {
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem;
  }
  .prob-bar .label { font-size: 0.78rem; color: #aaa; width: 90px; text-align: right; }
  .prob-bar .bg { flex: 1; background: #1a1a1a; border-radius: 3px; height: 8px; }
  .prob-bar .fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #1b5e20, #4caf50); }
  .prob-bar .pct { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #c8e6c9; width: 42px; }
  .value-badge {
    display: inline-block;
    background: #1b5e20;
    border: 1px solid #4caf50;
    color: #a5d6a7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    margin: 0.15rem;
  }
  .section-head {
    font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #4caf50; border-bottom: 1px solid #1e3a21; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
  }
  .match-header { font-size: 1.6rem; font-weight: 700; color: #fff; }
  .vs-badge { font-size: 0.75rem; background: #1e3a21; color: #4caf50; padding: 0.15rem 0.5rem; border-radius: 4px; }
  .log-box { background: #050f06; border: 1px solid #1e3a21; border-radius: 6px; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #81c784; white-space: pre-wrap; max-height: 320px; overflow-y: auto; }
  .stButton > button { background: #1b5e20; color: #c8e6c9; border: 1px solid #2e7d32; border-radius: 6px; font-size: 0.82rem; font-weight: 500; padding: 0.45rem 1.2rem; }
  .stButton > button:hover { background: #2e7d32; border-color: #4caf50; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & PATHS
# ═══════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
BANKROLL_PATH = DATA_DIR / "bankroll.json"

# ═══════════════════════════════════════════════════════════════
#  DATA LOADERS (cached by file mtime to force invalidation)
# ═══════════════════════════════════════════════════════════════
def get_model_mtime():
    model_path = DATA_DIR / "model.pkl"
    return model_path.stat().st_mtime if model_path.exists() else 0

@st.cache_data(show_spinner=False)
def load_schedule():
    path = DATA_DIR / "enriched_schedule.json"
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_teams(_mtime):
    """Get sorted team list from predict module's model.
    _mtime argument forces cache invalidation when model.pkl changes."""
    params = predict.get_team_params()
    return sorted(params.keys()) if params else []

@st.cache_data(show_spinner=False)
def load_db_stats():
    db = DATA_DIR / "matches.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    total = pd.read_sql("SELECT COUNT(*) as n FROM matches", conn)['n'][0]
    by_year = pd.read_sql("SELECT substr(date,1,4) as yr, COUNT(*) as n FROM matches GROUP BY yr ORDER BY yr", conn)
    conn.close()
    return {'total': total, 'by_year': by_year}

def load_team_news_dict():
    path = DATA_DIR / "team_news.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_team_news_dict(data):
    with open(DATA_DIR / "team_news.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_bankroll():
    if BANKROLL_PATH.exists():
        with open(BANKROLL_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {'bankroll': 18000, 'bets': []}

def save_bankroll(data):
    with open(BANKROLL_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# ── Display helpers ─────────────────────────────────────────────
def prob_bar(label, pct, color="linear-gradient(90deg,#1b5e20,#4caf50)"):
    st.markdown(f"""
    <div class="prob-bar">
      <div class="label">{label}</div>
      <div class="bg"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>
      <div class="pct">{pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {f"<div style='font-size:0.72rem;color:#555'>{sub}</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)

def render_probs_detail(probs):
    """Show Over/Under, BTTS, and most likely scores – exactly like predict.py."""
    st.markdown("**Over / Under Goals**")
    for label, key in [("Over 1.5", "over_1.5"), ("Under 1.5", "under_1.5"),
                       ("Over 2.5", "over_2.5"), ("Under 2.5", "under_2.5"),
                       ("Over 3.5", "over_3.5"), ("Under 3.5", "under_3.5")]:
        prob_bar(label, probs[key] * 100)

    st.markdown("**Both Teams to Score**")
    prob_bar("BTTS Yes", probs["btts"] * 100)
    prob_bar("BTTS No", probs["btts_no"] * 100)

    if probs.get("top_scores"):
        st.markdown("**Most Likely Scores**")
        for score, prob_val in probs["top_scores"]:
            prob_bar(score, prob_val * 100, color="#2e7d32")

# ── Results parser (INSERT OR IGNORE) ───────────────────────────
def process_results_text(text):
    """Parse lines like 'TeamA 2-1 TeamB' and insert into DB (no overwrite)."""
    import re
    schedule = load_schedule()
    schedule_lookup = {}
    for m in schedule:
        schedule_lookup[(m['team1'], m['team2'])] = m['date']
        schedule_lookup[(m['team2'], m['team1'])] = m['date']

    conn = sqlite3.connect(str(DATA_DIR / "matches.db"))
    pattern = r"^(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+)$"
    added = []
    for line in text.strip().splitlines():
        m = re.match(pattern, line.strip())
        if not m:
            continue
        team1_raw, g1, g2, team2_raw = m.groups()
        home_team, away_team = team1_raw.strip(), team2_raw.strip()
        home_goals, away_goals = int(g1), int(g2)
        match_date = schedule_lookup.get((home_team, away_team))
        if not match_date:
            match_date = datetime.now().strftime('%Y-%m-%d')
        competition = 'World Cup 2026' if match_date in schedule_lookup.values() else 'Friendly'
        conn.execute("""
            INSERT OR IGNORE INTO matches
            (date, home_team, away_team, home_score, away_score, competition, season, venue, neutral)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (match_date, home_team, away_team, home_goals, away_goals, competition, int(match_date[:4]), "auto", 1))
        conn.commit()
        added.append(f"{home_team} {home_goals}-{away_goals} {away_team} ({match_date})")
    conn.close()
    return added

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ WC 2026")
    page = st.radio("Navigation", [
        "Match Predictor",
        "Bet Slip",
        "Simulator",
        "Model Stats",
        "Team News",
        "Update Results",
        "Bankroll",
        "Train Model"
    ], label_visibility="collapsed")
    st.markdown("---")
    db_stats = load_db_stats()
    if db_stats:
        st.markdown(f"**DB:** {db_stats['total']:,} matches")
    model_path = DATA_DIR / "model.pkl"
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        st.markdown(f"**Model trained**  \n{mtime.strftime('%d %b %Y, %H:%M')}")
        st.markdown("---")
        if st.button("🔄 Refresh DB stats"):
            load_db_stats.clear()
            st.rerun()
        train_params = predict.get_train_params()
        if train_params:
            params_str = f"α={train_params.get('alpha', '?')}  HL={train_params.get('half_life', '?')}d"
            if 'prior_strength' in train_params:
                params_str += f"  PS={train_params.get('prior_strength', '?')}"
            st.caption(params_str)
    
    # ── Debug panel ───────────────────────────────────────────────
    with st.expander("🔍 Debug model"):
        if st.button("Show first 5 team parameters"):
            params = predict.get_team_params()
            if params:
                st.write({k: params[k] for k in list(params.keys())[:5]})
            else:
                st.write("No model loaded.")
        if st.button("Force reload model now"):
            predict.clear_model_cache()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

# ═══════════════════════════════════════════════════════════════
#  PAGE: MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
if page == "Match Predictor":
    st.markdown('<div class="match-header">Match Predictor</div>', unsafe_allow_html=True)
    model_mtime = get_model_mtime()
    teams = load_teams(model_mtime)
    schedule = load_schedule()
    if not teams:
        st.error("No model found. Train first.")
        st.stop()

    col1, col2, col3 = st.columns([5,1,5])
    with col1:
        team1 = st.selectbox("Home / Team 1", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        st.markdown("<div style='text-align:center;padding-top:1.9rem;color:#4caf50;font-weight:700'>VS</div>", unsafe_allow_html=True)
    with col3:
        team2 = st.selectbox("Away / Team 2", teams, index=1)
    if team1 == team2:
        st.warning("Pick two different teams.")
        st.stop()

    with st.expander("Match options & odds", expanded=False):
        oc1, oc2 = st.columns(2)
        with oc1:
            venue_choice = st.selectbox("Venue", ["Neutral", "Team 1 is home"])
            neutral = venue_choice == "Neutral"
        with oc2:
            match_date = st.date_input("Match date (optional)", value=None)
        rest1, rest2, travel1, travel2 = 0, 0, 0, 0
        auto_loaded = False
        if match_date and schedule:
            date_str = match_date.strftime('%Y-%m-%d')
            match_info = next((m for m in schedule if m['team1'] == team1 and m['team2'] == team2 and m['date'] == date_str), None)
            if match_info:
                rest1 = match_info.get('team1_rest', 0)
                rest2 = match_info.get('team2_rest', 0)
                travel1 = match_info.get('team1_travel', 0)
                travel2 = match_info.get('team2_travel', 0)
                auto_loaded = True
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            rest1 = st.number_input(f"{team1} rest days", 0,30, rest1, disabled=auto_loaded)
        with rc2:
            travel1 = st.number_input(f"{team1} travel km", 0,20000, travel1, disabled=auto_loaded)
        with rc3:
            rest2 = st.number_input(f"{team2} rest days", 0,30, rest2, disabled=auto_loaded)
        with rc4:
            travel2 = st.number_input(f"{team2} travel km", 0,20000, travel2, disabled=auto_loaded)

        st.markdown("**Bookmaker odds** (leave 0 to skip)")
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        o1    = oc1.number_input(f"{team1} win", 0.0,50.0,0.0,0.05,format="%.2f")
        oX    = oc2.number_input("Draw",         0.0,50.0,0.0,0.05,format="%.2f")
        o2    = oc3.number_input(f"{team2} win",  0.0,50.0,0.0,0.05,format="%.2f")
        o_o25 = oc4.number_input("Over 2.5",     0.0,20.0,0.0,0.05,format="%.2f")
        o_u25 = oc5.number_input("Under 2.5",    0.0,20.0,0.0,0.05,format="%.2f")

    if st.button("⚡ Run Prediction"):
        news = load_team_news_dict()
        att_mod1 = news.get(team1, {}).get('att_mod', 0.0)
        def_mod1 = news.get(team1, {}).get('def_mod', 0.0)
        att_mod2 = news.get(team2, {}).get('att_mod', 0.0)
        def_mod2 = news.get(team2, {}).get('def_mod', 0.0)

        with st.spinner("Computing..."):
            probs = predict.outcome_probs(
                team1, team2,
                neutral=neutral,
                rest1=rest1 or None, rest2=rest2 or None,
                travel1=travel1 or None, travel2=travel2 or None,
                match_date=str(match_date) if match_date else None,
                att_mod1=att_mod1, def_mod1=def_mod1,
                att_mod2=att_mod2, def_mod2=def_mod2
            )
        section("Expected Goals")
        c1,_,c2 = st.columns([4,1,4])
        with c1: metric_card(team1, f"{probs['lambda_home']:.2f}")
        with c2: metric_card(team2, f"{probs['lambda_away']:.2f}")

        section("Match Result")
        c1,c2,c3 = st.columns(3)
        with c1: metric_card(f"{team1} Win", f"{probs['home_win']*100:.1f}%")
        with c2: metric_card("Draw", f"{probs['draw']*100:.1f}%")
        with c3: metric_card(f"{team2} Win", f"{probs['away_win']*100:.1f}%")

        # ── Detailed probabilities (same as predict.py CLI) ──
        section("Detailed Probabilities")
        render_probs_detail(probs)

        section("Value Betting")
        markets = [
            (f"{team1} Win", o1, probs['home_win']),
            ("Draw", oX, probs['draw']),
            (f"{team2} Win", o2, probs['away_win']),
            ("Over 2.5", o_o25, probs['over_2.5']),
            ("Under 2.5", o_u25, probs['under_2.5']),
        ]
        any_odds = any(o>0 for _,o,_ in markets)
        if not any_odds:
            st.markdown('<span style="color:#666">Enter odds to see value.</span>', unsafe_allow_html=True)
        else:
            value_found = False
            for bet_name, odds_val, prob in markets:
                if odds_val > 0:
                    stake = predict.kelly_criterion(prob, odds_val)
                    if stake > 0:
                        value_found = True
                        edge = (prob * (1 + (odds_val-1)*(1-predict.TAX_RATE)) - 1) * 100
                        st.markdown(f'<span class="value-badge">✅ {bet_name} @ {odds_val:.2f} | edge {edge:+.1f}% | Kelly {stake*100:.1f}%</span>', unsafe_allow_html=True)
            if not value_found:
                st.markdown('<span style="color:#666">No value bets at these odds (post-tax).</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: BET SLIP
# ═══════════════════════════════════════════════════════════════
elif page == "Bet Slip":
    st.markdown('<div class="match-header">Daily Bet Slip</div>', unsafe_allow_html=True)
    schedule = load_schedule()
    if not schedule:
        st.error("No schedule found.")
        st.stop()

    dates_dict = {}
    for m in schedule:
        dates_dict.setdefault(m['date'], []).append(m)
    today_str = date.today().strftime('%Y-%m-%d')
    available = sorted(dates_dict.keys())
    upcoming = [d for d in available if d >= today_str]
    default = upcoming[0] if upcoming else available[-1]
    sel_date = st.selectbox("Matchday", available, index=available.index(default),
                            format_func=lambda d: datetime.strptime(d,'%Y-%m-%d').strftime('%a %d %b %Y'))
    matches = dates_dict[sel_date]
    news = load_team_news_dict()

    HOST_NATIONS = {'Mexico','United States','Canada'}

    section(f"{len(matches)} matches · {sel_date}")
    all_value = []
    for match in matches:
        t1, t2 = match['team1'], match['team2']
        r1 = match.get('team1_rest')
        r2 = match.get('team2_rest')
        tr1 = match.get('team1_travel',0)
        tr2 = match.get('team2_travel',0)

        home_team = t1 if t1 in HOST_NATIONS else (t2 if t2 in HOST_NATIONS else None)
        swapped = False
        calc_t1, calc_t2 = t1, t2
        calc_r1, calc_r2 = r1, r2
        calc_tr1, calc_tr2 = tr1, tr2
        if home_team == t2:
            calc_t1, calc_t2 = t2, t1
            calc_r1, calc_r2 = r2, r1
            calc_tr1, calc_tr2 = tr2, tr1
            swapped = True

        att_mod1 = news.get(calc_t1, {}).get('att_mod', 0.0)
        def_mod1 = news.get(calc_t1, {}).get('def_mod', 0.0)
        att_mod2 = news.get(calc_t2, {}).get('att_mod', 0.0)
        def_mod2 = news.get(calc_t2, {}).get('def_mod', 0.0)

        with st.expander(f"⚽ {t1} vs {t2} {'🏠' if home_team else '🌐'}"):
            probs = predict.outcome_probs(
                calc_t1, calc_t2,
                neutral=(home_team is None),
                rest1=calc_r1, rest2=calc_r2,
                travel1=calc_tr1, travel2=calc_tr2,
                match_date=sel_date,
                att_mod1=att_mod1, def_mod1=def_mod1,
                att_mod2=att_mod2, def_mod2=def_mod2
            )
            if swapped:
                probs['home_win'], probs['away_win'] = probs['away_win'], probs['home_win']
                probs['lambda_home'], probs['lambda_away'] = probs['lambda_away'], probs['lambda_home']

            # ── Expected Goals ──
            section("Expected Goals")
            eg_col1, _, eg_col2 = st.columns([4, 1, 4])
            with eg_col1:
                metric_card(t1, f"{probs['lambda_home']:.2f}")
            with eg_col2:
                metric_card(t2, f"{probs['lambda_away']:.2f}")

            # ── 1X2 ──
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(f"{t1} Win", f"{probs['home_win']*100:.1f}%")
            with c2:
                metric_card("Draw", f"{probs['draw']*100:.1f}%")
            with c3:
                metric_card(f"{t2} Win", f"{probs['away_win']*100:.1f}%")

            # ── Full probability breakdown (same as CLI) ──
            st.markdown("---")
            render_probs_detail(probs)

            st.markdown("**Enter odds:**")
            oc1,oc2,oc3,oc4,oc5 = st.columns(5)
            slip_o1  = oc1.number_input(f"{t1} W", 0.0,50.0,0.0, key=f"o1_{t1}{t2}", format="%.2f")
            slip_oX  = oc2.number_input("Draw", 0.0,50.0,0.0, key=f"oX_{t1}{t2}", format="%.2f")
            slip_o2  = oc3.number_input(f"{t2} W", 0.0,50.0,0.0, key=f"o2_{t1}{t2}", format="%.2f")
            slip_o25 = oc4.number_input("Ov 2.5", 0.0,20.0,0.0, key=f"o25_{t1}{t2}", format="%.2f")
            slip_u25 = oc5.number_input("Un 2.5", 0.0,20.0,0.0, key=f"u25_{t1}{t2}", format="%.2f")

            markets = [
                (f"{t1} Win", slip_o1,  probs['home_win']),
                ("Draw",       slip_oX,  probs['draw']),
                (f"{t2} Win",  slip_o2,  probs['away_win']),
                ("Over 2.5",   slip_o25, probs['over_2.5']),
                ("Under 2.5",  slip_u25, probs['under_2.5']),
            ]
            for bet_name, odds_val, prob in markets:
                if odds_val > 0:
                    stake = predict.kelly_criterion(prob, odds_val)
                    if stake > 0:
                        edge = (prob * (1 + (odds_val-1)*(1-predict.TAX_RATE)) - 1) * 100
                        st.markdown(f'<span class="value-badge">✅ {bet_name} @ {odds_val:.2f} | edge {edge:+.1f}% | Kelly {stake*100:.1f}%</span>', unsafe_allow_html=True)
                        all_value.append({'Match':f"{t1} vs {t2}", 'Bet':bet_name, 'Odds':odds_val, 'Model %':f"{prob*100:.1f}%", 'Edge':f"{edge:+.1f}%", 'Kelly':f"{stake*100:.1f}%"})
    if all_value:
        section("Value Bet Summary")
        st.dataframe(pd.DataFrame(all_value), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif page == "Simulator":
    st.markdown('<div class="match-header">Tournament Simulator</div>', unsafe_allow_html=True)
    if st.button("▶ Run Simulation (10,000 trials)"):
        with st.spinner("Simulating... this may take 10-20 seconds."):
            result = subprocess.run([sys.executable, "simulate.py"], capture_output=True, text=True, cwd=Path.cwd())
            output = result.stdout + result.stderr
        if result.returncode == 0:
            st.success("Simulation completed.")
            lines = output.splitlines()
            table_start = next((i for i, line in enumerate(lines) if "Win Grp" in line), None)
            if table_start is not None:
                import re
                data_lines = lines[table_start+2:]
                rows = []
                for line in data_lines:
                    line = line.strip()
                    if not line or 'R32 =' in line:
                        continue
                    parts = re.split(r'\s{2,}', line)
                    if len(parts) >= 9:
                        team = parts[0].strip()
                        vals = []
                        for p in parts[1:9]:
                            val_str = p.strip().replace('%','')
                            try:
                                vals.append(float(val_str))
                            except ValueError:
                                vals.append(0.0)
                        rows.append([team] + vals)
                if rows:
                    cols = ['Team','Win Grp','Top2','R32','R16','QF','SF','Final','Champ']
                    df_sim = pd.DataFrame(rows, columns=cols)
                    st.dataframe(df_sim.style.format({c:'{:.1f}%' for c in cols[1:]}), use_container_width=True)
                else:
                    st.text(output)
        else:
            st.error("Simulation failed.")
            st.code(output)

# ═══════════════════════════════════════════════════════════════
#  PAGE: MODEL STATS
# ═══════════════════════════════════════════════════════════════
elif page == "Model Stats":
    st.markdown('<div class="match-header">Model Stats</div>', unsafe_allow_html=True)
    params = predict.get_team_params()
    if not params:
        st.error("No model found.")
        st.stop()

    db_stats = load_db_stats()
    if db_stats:
        section("Training Database")
        c1,c2 = st.columns(2)
        c1.metric("Total Matches", f"{db_stats['total']:,}")
        c2.metric("Years", f"{db_stats['by_year']['yr'].iloc[0]}-{db_stats['by_year']['yr'].iloc[-1]}")

    train_params = predict.get_train_params()
    if train_params:
        section("Training Configuration")
        st.write(f"**Half-life:** {train_params.get('half_life', '?')} days")
        st.write(f"**Regularisation α:** {train_params.get('alpha', '?')}")
        if 'prior_strength' in train_params:
            st.write(f"**Elo prior strength:** {train_params.get('prior_strength', '?')}")
        st.write(f"**Trained on:** {train_params.get('trained_on', '?')}")

    section("Team Rankings")
    df = pd.DataFrame([{'Team':t, 'Attack':v['att'], 'Defence':v['def'], 'Overall':v['att']-v['def']}
                       for t,v in params.items()]).sort_values('Overall', ascending=False).reset_index(drop=True)
    st.dataframe(df.style.format({'Attack':'{:.3f}','Defence':'{:.3f}','Overall':'{:.3f}'}), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: TEAM NEWS
# ═══════════════════════════════════════════════════════════════
elif page == "Team News":
    st.markdown('<div class="match-header">Team News (Manual Adjustments)</div>', unsafe_allow_html=True)
    news = load_team_news_dict()
    model_mtime = get_model_mtime()
    teams = load_teams(model_mtime)
    if not teams:
        st.error("No model.")
        st.stop()

    selected_team = st.selectbox("Select team", teams)
    current = news.get(selected_team, {'att_mod':0.0, 'def_mod':0.0})
    col1, col2 = st.columns(2)
    with col1:
        att = st.number_input(f"Attack modifier ({selected_team})", -2.0, 2.0, current.get('att_mod',0.0), 0.05)
    with col2:
        df = st.number_input(f"Defence modifier ({selected_team})", -2.0, 2.0, current.get('def_mod',0.0), 0.05)
    if st.button("Save"):
        news[selected_team] = {'att_mod':att, 'def_mod':df}
        save_team_news_dict(news)
        st.success(f"Saved adjustments for {selected_team}.")
    if news:
        section("Current Adjustments")
        st.json(news)

# ═══════════════════════════════════════════════════════════════
#  PAGE: UPDATE RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "Update Results":
    st.markdown('<div class="match-header">Update Results</div>', unsafe_allow_html=True)
    st.markdown("Paste scores, one per line: `TeamA 2-1 TeamB`")
    text = st.text_area("Results", height=200)
    if st.button("Add to database"):
        if text.strip():
            added = process_results_text(text)
            if added:
                st.success(f"Added {len(added)} matches:")
                for a in added:
                    st.write(f"- {a}")
                load_db_stats.clear()
                st.cache_data.clear()
            else:
                st.warning("No valid lines found.")
        else:
            st.warning("Paste results first.")

# ═══════════════════════════════════════════════════════════════
#  PAGE: BANKROLL
# ═══════════════════════════════════════════════════════════════
elif page == "Bankroll":
    st.markdown('<div class="match-header">Bankroll Tracker</div>', unsafe_allow_html=True)
    data = load_bankroll()
    bankroll = data['bankroll']
    bets = data['bets']

    col1,col2 = st.columns(2)
    with col1:
        metric_card("Current Bankroll", f"{bankroll:,.2f} MKD")
    with col2:
        profit = bankroll - 18000
        col = "green" if profit >=0 else "red"
        st.markdown(f"**Total Profit:** <span style='color:{col}'>{profit:+,.2f} MKD</span>", unsafe_allow_html=True)

    with st.expander("Add new bet"):
        with st.form("bet_form"):
            date = st.date_input("Date")
            match = st.text_input("Match (e.g. Mexico vs South Africa)")
            bet_type = st.text_input("Bet type (Home/Draw/Away/Over 2.5 etc.)")
            odds = st.number_input("Odds", 1.0, 50.0, 2.0, 0.01)
            kelly_pct = st.number_input("Kelly % (from bet slip)", 0.0, 20.0, 2.0, 0.1) / 100
            result = st.selectbox("Result", ["Win", "Loss"])
            submitted = st.form_submit_button("Add Bet")
            if submitted:
                stake = bankroll * kelly_pct
                if result == "Win":
                    profit_after_tax = stake * (odds - 1) * 0.85
                else:
                    profit_after_tax = -stake
                new_bankroll = bankroll + profit_after_tax
                bets.append({
                    'date': str(date),
                    'match': match,
                    'bet_type': bet_type,
                    'odds': odds,
                    'kelly_pct': kelly_pct,
                    'stake': stake,
                    'result': result,
                    'profit': profit_after_tax,
                    'bankroll': new_bankroll
                })
                data['bankroll'] = new_bankroll
                data['bets'] = bets
                save_bankroll(data)
                st.success(f"Bet added. New bankroll: {new_bankroll:,.2f} MKD")
                st.rerun()

    if bets:
        section("Bet History")
        df_bets = pd.DataFrame(bets)
        df_bets['date'] = pd.to_datetime(df_bets['date'])
        df_bets = df_bets.sort_values('date', ascending=False)
        st.dataframe(df_bets.style.format({'stake':'{:,.2f}','profit':'{:,.2f}','bankroll':'{:,.2f}','odds':'{:.2f}','kelly_pct':'{:.1%}'}),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: TRAIN MODEL
# ═══════════════════════════════════════════════════════════════
elif page == "Train Model":
    st.markdown('<div class="match-header">Train Model</div>', unsafe_allow_html=True)
    db_stats = load_db_stats()
    if db_stats:
        section("Current Database")
        c1,c2 = st.columns(2)
        c1.metric("Matches in DB", f"{db_stats['total']:,}")
        c2.metric("Date Range", f"{db_stats['by_year']['yr'].iloc[0]}-{db_stats['by_year']['yr'].iloc[-1]}")

    section("Training Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        half_life = st.slider("Half-life (days)", 90, 730, 600, 30)
    with col2:
        alpha = st.slider("Regularisation α", 0.1, 10.0, 1.5, 0.1)
    with col3:
        prior_strength = st.slider("Elo prior strength", 0.0, 3.0, 1.5, 0.1,
                                   help="0 = ignore Elo; 1.5 = strong anchor. Higher values keep minnows realistic.")

    if st.button("🚀 Train now"):
        with st.spinner("Training..."):
            result = subprocess.run(
                [sys.executable, "train.py",
                 f"--half-life={half_life}",
                 f"--alpha={alpha}",
                 f"--prior-strength={prior_strength}"],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            output = result.stdout + result.stderr
        if result.returncode == 0:
            st.success("Model trained successfully.")
            predict.clear_model_cache()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Training failed.")
        st.markdown('<div class="log-box">' + output + '</div>', unsafe_allow_html=True)