#!/usr/bin/env python3
import sqlite3
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from datetime import datetime
from pathlib import Path
import argparse
import pickle
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

DATA_DIR   = Path("data")
DB_PATH    = DATA_DIR / "matches.db"
MODEL_PATH = DATA_DIR / "model.pkl"

REFERENCE_DATE = datetime(2026, 6, 11)

# Competition weights (simple, constant)
COMPETITION_WEIGHTS = {
    'FIFA World Cup':               3.0,
    'UEFA Euro':                    2.5,
    'Copa America':                 2.5,
    'Copa América':                 2.5,
    'AFC Asian Cup':                2.0,
    'Africa Cup of Nations':        2.0,
    'African Cup of Nations':       2.0,
    'Gold Cup':                     1.8,
    'UEFA Nations League':          1.5,
    'CONCACAF Nations League':      1.5,
    'UEFA Euro qualification':      1.4,
    'World Cup qualification':      1.3,
    'FIFA World Cup qualification': 1.3,
    'AFC qualification':            1.3,
    'CAF qualification':            1.3,
    'CONMEBOL qualification':       1.3,
    'CONCACAF qualification':       1.3,
    'Confederations Cup':           1.2,
    'Friendly':                     0.5,
}

# ── Elo upset-dampening parameters ───────────────────────────────────────────
ELO_SURPRISE_FLOOR   = 0.35   # minimum weight for a massive upset result
ELO_SURPRISE_THRESH  = 1.5    # goal-diff error before penalty kicks in
ELO_SURPRISE_SLOPE   = 0.18   # how quickly weight falls beyond threshold
ELO_SCALE            = 350    # Elo points → expected goal-diff scale


def get_competition_weight(competition: str) -> float:
    if not competition:
        return 1.0
    comp_lower = competition.lower()
    for key, w in COMPETITION_WEIGHTS.items():
        if key.lower() in comp_lower:
            return w
    return 1.0


def load_name_map() -> dict:
    path = DATA_DIR / "team_name_map.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_elo() -> dict:
    """Load global_elo.json, applying the team name map so keys match training data."""
    path = DATA_DIR / "global_elo.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    name_map = load_name_map()
    elo = {}
    for team, rating in raw.items():
        canonical = name_map.get(team, team)
        elo[canonical] = rating
        elo[team]      = rating   # keep original too
    return elo


def elo_surprise_weight(home_elo: float, away_elo: float,
                        home_score: float, away_score: float) -> float:
    """
    Down-weights matches where the scoreline is a large upset relative to Elo.

    Logic:
      - Compute the Elo-implied expected goal difference.
      - Compute the actual goal difference.
      - If the discrepancy is small  → weight ≈ 1.0  (normal result, full credit)
      - If the discrepancy is large  → weight → ELO_SURPRISE_FLOOR  (heavy upset,
        e.g. Cape Verde 0-0 Spain, less credit to the defensive parameters)

    This stops elite teams being punished by a single freak clean-sheet and stops
    minnows earning an inflated defensive rating from one heroic stand.
    """
    expected_gd = (home_elo - away_elo) / ELO_SCALE
    actual_gd   = home_score - away_score
    error       = abs(actual_gd - expected_gd)
    excess      = max(0.0, error - ELO_SURPRISE_THRESH)
    weight      = 1.0 / (1.0 + ELO_SURPRISE_SLOPE * excess)
    return max(ELO_SURPRISE_FLOOR, weight)


def load_matches(half_life_days: int) -> pd.DataFrame:
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("""
        SELECT date, home_team, away_team,
               home_score, away_score, neutral, competition
        FROM matches
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        print("No matches found in database.")
        sys.exit(1)

    df['date']        = pd.to_datetime(df['date'])
    df['competition'] = df['competition'].fillna('Unknown')
    df['neutral']     = df['neutral'].fillna(1).astype(bool)
    df = df[df['date'] >= '2014-01-01']

    name_map = load_name_map()
    df['home_team'] = df['home_team'].replace(name_map)
    df['away_team'] = df['away_team'].replace(name_map)

    qualified_path = DATA_DIR / "qualified_teams.json"
    if qualified_path.exists():
        with open(qualified_path, encoding="utf-8") as f:
            qualified = set(json.load(f))
        before = len(df)
        df = df[df['home_team'].isin(qualified) & df['away_team'].isin(qualified)]
        print(f"Qualified filter: {before} → {len(df)} matches "
              f"({len(qualified)} WC 2026 teams)")
    else:
        print("Warning: qualified_teams.json not found — training on all teams.")

    if df.empty:
        print("No matches remain after filtering.")
        sys.exit(1)

    # ── Time-decay weight ─────────────────────────────────────────────────────
    ref        = pd.Timestamp(REFERENCE_DATE)
    days_ago   = (ref - df['date']).dt.days.clip(lower=0)
    df['tw']   = np.exp(-days_ago * np.log(2) / half_life_days)
    df['tw']  /= df['tw'].mean()

    # ── Competition weight ────────────────────────────────────────────────────
    df['cw'] = df['competition'].apply(get_competition_weight)

    # ── Elo surprise weight ───────────────────────────────────────────────────
    elo = load_elo()
    DEFAULT_ELO = 1500
    df['elo_w'] = df.apply(
        lambda r: elo_surprise_weight(
            elo.get(r['home_team'], DEFAULT_ELO),
            elo.get(r['away_team'], DEFAULT_ELO),
            r['home_score'],
            r['away_score'],
        ),
        axis=1,
    )

    # Log how many matches were meaningfully dampened
    dampened = (df['elo_w'] < 0.80).sum()
    print(f"  Elo surprise dampening: {dampened} matches weighted < 0.80")

    # ── Combined weight ───────────────────────────────────────────────────────
    df['weight']  = df['tw'] * df['cw'] * df['elo_w']
    df['weight'] /= df['weight'].mean()

    return df


def train_ridge_model(df, alpha):
    teams     = sorted(set(df['home_team']) | set(df['away_team']))
    team2idx  = {t: i for i, t in enumerate(teams)}
    n_teams   = len(teams)
    n_matches = len(df)

    X             = np.zeros((n_matches * 2, n_teams * 2 + 1))
    y             = np.zeros(n_matches * 2)
    sample_weight = np.zeros(n_matches * 2)

    for i, (_, row) in enumerate(df.iterrows()):
        h_idx = team2idx[row['home_team']]
        a_idx = team2idx[row['away_team']]
        w     = row['weight']

        # Home goals: att_home + def_away + home_adv
        X[i*2,     h_idx]             = 1
        X[i*2,     n_teams + a_idx]   = 1
        X[i*2,     -1]                = 1
        y[i*2]                        = np.log(max(row['home_score'], 0.1))
        sample_weight[i*2]            = w

        # Away goals: att_away + def_home
        X[i*2+1,   a_idx]             = 1
        X[i*2+1,   n_teams + h_idx]   = 1
        y[i*2+1]                      = np.log(max(row['away_score'], 0.1))
        sample_weight[i*2+1]          = w

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y, sample_weight=sample_weight)

    coef     = model.coef_
    att      = coef[:n_teams].copy()
    def_     = coef[n_teams:2*n_teams].copy()
    home_adv = coef[-1]

    # Centre parameters so mean att = mean def = 0
    att_mean = np.mean(att)
    def_mean = np.mean(def_)
    att     -= att_mean
    def_    -= def_mean
    home_adv += att_mean + def_mean

    team_params = {
        t: {'att': float(att[i]), 'def': float(def_[i])}
        for i, t in enumerate(teams)
    }
    return team_params, home_adv, teams


def train_model(half_life: int = 600, alpha: float = 2.5) -> dict:
    print("=" * 55)
    print("  WC 2026 Predictor — Ridge + Elo Surprise Weighting")
    print("=" * 55)
    print(f"  half-life={half_life}d | alpha={alpha} | "
          f"surprise_floor={ELO_SURPRISE_FLOOR} | "
          f"surprise_slope={ELO_SURPRISE_SLOPE}")

    df = load_matches(half_life)
    print(f"  Training on {len(df)} matches "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    team_params, home_adv, teams = train_ridge_model(df, alpha)

    model = {
        'team_params': team_params,
        'home_adv':    home_adv,
        'teams':       teams,
        'train_params': {
            'half_life':         half_life,
            'alpha':             alpha,
            'surprise_floor':    ELO_SURPRISE_FLOOR,
            'surprise_slope':    ELO_SURPRISE_SLOPE,
            'trained_on':        datetime.now().isoformat(),
        },
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n  Model saved → {MODEL_PATH}")
    print(f"  home_adv={home_adv:.3f}")

    top_att = sorted(team_params.items(), key=lambda x: x[1]['att'], reverse=True)[:10]
    print("\n  Top 10 attacking teams:")
    for team, p in top_att:
        print(f"    {team:<28s}  att={p['att']:+.3f}")

    # Strong defence is negative def (harder to score against)
    top_def = sorted(team_params.items(), key=lambda x: x[1]['def'])[:10]
    print("\n  Top 10 defensive teams (lower def = better):")
    for team, p in top_def:
        print(f"    {team:<28s}  def={p['def']:+.3f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC 2026 Ridge model with Elo surprise weighting")
    parser.add_argument("--half-life",       type=int,   default=600,
                        help="Time-decay half-life in days (default: 600)")
    parser.add_argument("--alpha",           type=float, default=2.5,
                        help="Ridge regularisation (default: 2.5)")
    parser.add_argument("--surprise-floor",  type=float, default=ELO_SURPRISE_FLOOR,
                        help=f"Min weight for giant-killing matches (default: {ELO_SURPRISE_FLOOR})")
    parser.add_argument("--surprise-slope",  type=float, default=ELO_SURPRISE_SLOPE,
                        help=f"How fast weight falls for upsets (default: {ELO_SURPRISE_SLOPE})")
    args = parser.parse_args()

    ELO_SURPRISE_FLOOR  = args.surprise_floor
    ELO_SURPRISE_SLOPE  = args.surprise_slope
    train_model(half_life=args.half_life, alpha=args.alpha)