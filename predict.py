import pickle
import json
import numpy as np
from pathlib import Path
import scipy.stats as stats
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# ── Constants ─────────────────────────────────────────────────────────────────
TAX_RATE       = 0.15        # 15% tax on net winnings
ELO_BLEND      = 0.35        # how strongly Elo adjusts lambdas
ELO_GOAL_RATIO = 0.5 / 100  # Elo points → goal difference scale
KELLY_FRACTION = 0.25        # quarter-Kelly to limit variance
MAX_GOALS      = 10          # grid size — higher = less truncation error

# Dixon-Coles low-score correlation correction
# Negative ρ means 0-0 / 1-1 are slightly more common than pure Poisson predicts.
# Typical fitted values lie in [-0.20, -0.05]; -0.13 is a widely cited default.
DIXON_COLES_RHO = -0.13

DATA_DIR = Path("data")

# ── File-level caches (invalidate on mtime change, no restart needed) ─────────
_MODEL_CACHE = {"mtime": None, "data": None}
_NEWS_CACHE  = {"mtime": None, "data": None}
_ELO_CACHE   = {"mtime": None, "data": None}
_MAP_CACHE   = {"mtime": None, "data": None}


def clear_model_cache():
    """Clear all in-memory caches so files are reloaded."""
    global _MODEL_CACHE, _NEWS_CACHE, _ELO_CACHE, _MAP_CACHE
    _MODEL_CACHE = {"mtime": None, "data": None}
    _NEWS_CACHE  = {"mtime": None, "data": None}
    _ELO_CACHE   = {"mtime": None, "data": None}
    _MAP_CACHE   = {"mtime": None, "data": None}


def _load_name_map() -> dict:
    path = DATA_DIR / "team_name_map.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _MAP_CACHE["mtime"] != mtime:
        with open(path, encoding="utf-8") as f:
            _MAP_CACHE["data"] = json.load(f)
        _MAP_CACHE["mtime"] = mtime
    return _MAP_CACHE["data"]


def _resolve(team: str) -> str:
    return _load_name_map().get(team, team)


def _load_model() -> dict:
    path = DATA_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError("model.pkl not found — run train.py first.")
    mtime = path.stat().st_mtime
    if _MODEL_CACHE["mtime"] != mtime:
        with open(path, "rb") as f:
            _MODEL_CACHE["data"] = pickle.load(f)
        _MODEL_CACHE["mtime"] = mtime
    return _MODEL_CACHE["data"]


def _load_team_news() -> dict:
    path = DATA_DIR / "team_news.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _NEWS_CACHE["mtime"] != mtime:
        with open(path, encoding="utf-8") as f:
            _NEWS_CACHE["data"] = json.load(f)
        _NEWS_CACHE["mtime"] = mtime
    return _NEWS_CACHE["data"]


def _load_global_elo() -> dict:
    path = DATA_DIR / "global_elo.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _ELO_CACHE["mtime"] != mtime:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        elo = {}
        for team, rating in raw.items():
            elo[team]           = rating
            elo[_resolve(team)] = rating
        _ELO_CACHE["data"]  = elo
        _ELO_CACHE["mtime"] = mtime
    return _ELO_CACHE["data"]


def _load_schedule() -> list:
    path = DATA_DIR / "enriched_schedule.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_heat_factors():
    path = DATA_DIR / "heat_factors.json"
    if not path.exists():
        return {}, {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.loads(f.read().strip())
        return data.get("venues", {}), data.get("team_tolerance", {})
    except Exception as e:
        print(f"  Warning: could not load heat_factors.json: {e}", file=sys.stderr)
        return {}, {}


# ── Public helpers ─────────────────────────────────────────────────────────────
def get_team_params() -> dict:
    return _load_model().get("team_params", {})

def get_train_params() -> dict:
    return _load_model().get("train_params", {})


# ── Dixon-Coles correction ─────────────────────────────────────────────────────
def _dixon_coles_tau(i: int, j: int,
                     lambda1: float, lambda2: float,
                     rho: float = DIXON_COLES_RHO) -> float:
    """
    Correction factor τ(i, j) for the four low-scoring cells.

    The original paper (Dixon & Coles 1997) showed that a standard
    independent Poisson model under-predicts 0-0 and 1-1 and slightly
    over-predicts 1-0 and 0-1.  τ adjusts the joint probability grid:

        P_DC(i, j) ∝ τ(i, j) · Poisson(i; λ₁) · Poisson(j; λ₂)

    Only four cells are affected; all others return 1.0.
    ρ < 0 → 0-0 and 1-1 become MORE likely (negative correlation).
    """
    if   i == 0 and j == 0:
        return 1.0 - lambda1 * lambda2 * rho
    elif i == 1 and j == 0:
        return 1.0 + lambda2 * rho
    elif i == 0 and j == 1:
        return 1.0 + lambda1 * rho
    elif i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _build_score_grid(lambda1: float, lambda2: float,
                      rho: float = DIXON_COLES_RHO) -> np.ndarray:
    """
    Build the joint score probability matrix with Dixon-Coles correction.
    Rows = home goals, columns = away goals.  Renormalised after truncation.
    """
    g        = np.arange(MAX_GOALS + 1)
    home_pmf = stats.poisson.pmf(g, lambda1)
    away_pmf = stats.poisson.pmf(g, lambda2)
    joint    = np.outer(home_pmf, away_pmf)

    # Apply Dixon-Coles τ to the 2×2 low-score corner
    for i in range(min(2, MAX_GOALS + 1)):
        for j in range(min(2, MAX_GOALS + 1)):
            joint[i, j] *= _dixon_coles_tau(i, j, lambda1, lambda2, rho)

    joint /= joint.sum()   # renormalise (truncation + DC correction)
    return joint


# ── Core prediction ────────────────────────────────────────────────────────────
def outcome_probs(team1: str, team2: str,
                  neutral: bool = True,
                  rest1=None,   rest2=None,
                  travel1=None, travel2=None,
                  match_date: str = None,
                  att_mod1: float = 0.0, def_mod1: float = 0.0,
                  att_mod2: float = 0.0, def_mod2: float = 0.0,
                  rho: float = DIXON_COLES_RHO) -> dict:
    """
    Compute match outcome probabilities using a Poisson + Dixon-Coles model.

    Parameters
    ----------
    team1, team2  : team names (must match model keys)
    neutral       : True = no home advantage applied
    rest1, rest2  : days since last match (None = ignore)
    travel1/2     : km travelled to venue (None = ignore)
    match_date    : YYYY-MM-DD string (used for heat/venue lookup)
    att_mod1/2    : manual attack adjustment (added on top of team_news.json)
    def_mod1/2    : manual defence adjustment
    rho           : Dixon-Coles correlation parameter (default: DIXON_COLES_RHO)
    """
    model       = _load_model()
    news        = _load_team_news()
    elo_dict    = _load_global_elo()
    team_params = model["team_params"]

    # ── Team parameter lookup ─────────────────────────────────────────────────
    p1 = team_params.get(team1, team_params.get(_resolve(team1), {"att": 0.0, "def": 0.0}))
    p2 = team_params.get(team2, team_params.get(_resolve(team2), {"att": 0.0, "def": 0.0}))

    att1 = p1["att"];  def1 = p1["def"]
    att2 = p2["att"];  def2 = p2["def"]

    # ── Apply team_news.json adjustments ─────────────────────────────────────
    for team in (team1, team2):
        entry = news.get(team, news.get(_resolve(team), {}))
        if entry:
            if team == team1:
                att1 += entry.get("att_mod", 0.0)
                def1 += entry.get("def_mod", 0.0)
            else:
                att2 += entry.get("att_mod", 0.0)
                def2 += entry.get("def_mod", 0.0)
            print(f"  Applied team_news adjustment for {team}")

    # ── Caller-supplied manual tweaks ─────────────────────────────────────────
    att1 += att_mod1;  def1 += def_mod1
    att2 += att_mod2;  def2 += def_mod2

    # ── Base lambdas ──────────────────────────────────────────────────────────
    home_adv       = 0.0 if neutral else model.get("home_adv", 0.0)
    base_intercept = model.get("intercept", 0.0)

    log_lambda1 = base_intercept + home_adv + att1 + def2
    log_lambda2 = base_intercept            + att2 - def1

    lambda1 = np.exp(log_lambda1)
    lambda2 = np.exp(log_lambda2)

    # ── Rest / travel adjustments ─────────────────────────────────────────────
    if rest1 is not None:
        rest_f  = 1.0 - (8 - rest1) * 0.01 if rest1 < 8 else 1.0
        trav_f  = 1.0 - (travel1 - 500) / 500 * 0.01 if (travel1 or 0) > 500 else 1.0
        lambda1 *= max(0.8, rest_f * trav_f)
    if rest2 is not None:
        rest_f  = 1.0 - (8 - rest2) * 0.01 if rest2 < 8 else 1.0
        trav_f  = 1.0 - (travel2 - 500) / 500 * 0.01 if (travel2 or 0) > 500 else 1.0
        lambda2 *= max(0.8, rest_f * trav_f)

    # ── Elo adjustment ────────────────────────────────────────────────────────
    elo1   = elo_dict.get(team1, elo_dict.get(_resolve(team1), 1500))
    elo2   = elo_dict.get(team2, elo_dict.get(_resolve(team2), 1500))
    diff   = (elo1 - elo2) * ELO_GOAL_RATIO
    factor = np.exp(diff * ELO_BLEND)
    lambda1 = float(np.clip(lambda1 * factor, 0.5, 8.0))
    lambda2 = float(np.clip(lambda2 / factor, 0.5, 8.0))

    # ── Heat / venue adjustment ───────────────────────────────────────────────
    if match_date:
        schedule = _load_schedule()
        venues, tolerance = _load_heat_factors()
        match_info = next(
            (m for m in schedule
             if m["team1"] == team1 and m["team2"] == team2
             and m["date"] == match_date),
            None
        )
        if match_info and venues:
            city = match_info.get("venue_city")
            if city and city in venues:
                vf     = venues[city]
                t1_tol = tolerance.get(team1, tolerance.get(_resolve(team1), 1.0))
                t2_tol = tolerance.get(team2, tolerance.get(_resolve(team2), 1.0))
                lambda1 = float(np.clip(lambda1 * vf * t1_tol, 0.5, 8.0))
                lambda2 = float(np.clip(lambda2 * vf * t2_tol, 0.5, 8.0))

    # ── Score probability grid (with Dixon-Coles) ─────────────────────────────
    joint    = _build_score_grid(lambda1, lambda2, rho=rho)
    home_pmf = stats.poisson.pmf(np.arange(MAX_GOALS + 1), lambda1)
    away_pmf = stats.poisson.pmf(np.arange(MAX_GOALS + 1), lambda2)

    home_win = float(np.tril(joint, -1).sum())
    away_win = float(np.triu(joint, 1).sum())
    draw     = float(np.diag(joint).sum())

    # ── Over/Under ────────────────────────────────────────────────────────────
    g      = np.arange(MAX_GOALS + 1)
    totals = np.add.outer(g, g)
    over15 = float(joint[totals > 1.5].sum())
    over25 = float(joint[totals > 2.5].sum())
    over35 = float(joint[totals > 3.5].sum())

    # ── BTTS ──────────────────────────────────────────────────────────────────
    # inclusion-exclusion: P(home≥1 AND away≥1) = 1 - P(home=0) - P(away=0) + P(both=0)
    btts = float(1.0 - home_pmf[0] - away_pmf[0] + home_pmf[0] * away_pmf[0])

    # ── Top score probabilities ───────────────────────────────────────────────
    top_scores = sorted(
        [(f"{i}-{j}", float(joint[i, j]))
         for i in range(min(6, MAX_GOALS + 1))
         for j in range(min(6, MAX_GOALS + 1))
         if joint[i, j] > 0.005],
        key=lambda x: -x[1],
    )[:5]

    return {
        "lambda_home": lambda1,
        "lambda_away": lambda2,
        "home_win":    home_win,
        "draw":        draw,
        "away_win":    away_win,
        "over_1.5":    over15,    "under_1.5": 1.0 - over15,
        "over_2.5":    over25,    "under_2.5": 1.0 - over25,
        "over_3.5":    over35,    "under_3.5": 1.0 - over35,
        "btts":        btts,      "btts_no":   1.0 - btts,
        "top_scores":  top_scores,
        "rho":         rho,
    }


def kelly_criterion(prob: float, odds: float,
                    fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.
    Uses post-tax effective odds. Returns 0 if no edge.
    """
    if odds <= 1.0 or prob <= 0.0:
        return 0.0
    net_odds = 1.0 + (odds - 1.0) * (1.0 - TAX_RATE)
    if net_odds <= 1.0:
        return 0.0
    b     = net_odds - 1.0
    kelly = (b * prob - (1.0 - prob)) / b
    return max(0.0, fraction * kelly)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC 2026 Match Predictor")
    parser.add_argument("--team1",      required=True)
    parser.add_argument("--team2",      required=True)
    parser.add_argument("--venue",      choices=["home", "neutral"], default="neutral")
    parser.add_argument("--match-date")
    parser.add_argument("--rest1",      type=int,   default=None)
    parser.add_argument("--rest2",      type=int,   default=None)
    parser.add_argument("--travel1",    type=float, default=None)
    parser.add_argument("--travel2",    type=float, default=None)
    parser.add_argument("--odds1",      type=float, default=None)
    parser.add_argument("--oddsX",      type=float, default=None)
    parser.add_argument("--odds2",      type=float, default=None)
    parser.add_argument("--odds-o25",   type=float, default=None)
    parser.add_argument("--odds-u25",   type=float, default=None)
    parser.add_argument("--rho",        type=float, default=DIXON_COLES_RHO,
                        help=f"Dixon-Coles ρ (default: {DIXON_COLES_RHO})")
    parser.add_argument("--list-teams", action="store_true")
    parser.add_argument("--search")
    args = parser.parse_args()

    model = _load_model()

    if args.list_teams:
        teams = sorted(model["team_params"].keys())
        news  = _load_team_news()
        print(f"\nAvailable teams ({len(teams)}):")
        for t in teams:
            marker = " *" if t in news else ""
            print(f"  {t}{marker}")
        if news:
            print("\n* = has manual adjustment in team_news.json")
        sys.exit(0)

    if args.search:
        matches = [t for t in model["team_params"] if args.search.lower() in t.lower()]
        print(f"\nTeams matching '{args.search}':")
        for t in sorted(matches):
            print(f"  {t}")
        sys.exit(0)

    probs = outcome_probs(
        args.team1, args.team2,
        neutral    = (args.venue == "neutral"),
        rest1      = args.rest1,    rest2      = args.rest2,
        travel1    = args.travel1,  travel2    = args.travel2,
        match_date = args.match_date,
        rho        = args.rho,
    )

    print(f"\n{'='*55}")
    print(f"  {args.team1} vs {args.team2}")
    print(f"  Venue: {args.venue}  |  Dixon-Coles ρ={probs['rho']:.2f}")
    print(f"{'='*55}")
    print(f"  Expected Goals: {probs['lambda_home']:.2f} – {probs['lambda_away']:.2f}")
    print(f"\n  Match Result (1X2):")
    print(f"    {args.team1} Win:  {probs['home_win']*100:5.1f}%")
    print(f"    Draw:             {probs['draw']*100:5.1f}%")
    print(f"    {args.team2} Win:  {probs['away_win']*100:5.1f}%")
    print(f"\n  Over/Under Goals:")
    for label, key in [("Over 1.5","over_1.5"),("Under 1.5","under_1.5"),
                        ("Over 2.5","over_2.5"),("Under 2.5","under_2.5"),
                        ("Over 3.5","over_3.5"),("Under 3.5","under_3.5")]:
        print(f"    {label:<12} {probs[key]*100:5.1f}%")
    print(f"\n  Both Teams to Score:")
    print(f"    Yes:  {probs['btts']*100:5.1f}%")
    print(f"    No:   {probs['btts_no']*100:5.1f}%")
    print(f"\n  Most Likely Score: {probs['top_scores'][0][0] if probs['top_scores'] else 'N/A'}")
    print(f"  Top Score Probabilities:")
    for score, prob in probs["top_scores"]:
        print(f"    {score}: {prob*100:5.1f}%")

    # Value betting
    bet_markets = [
        (f"{args.team1} Win", args.odds1,    probs["home_win"]),
        ("Draw",               args.oddsX,    probs["draw"]),
        (f"{args.team2} Win",  args.odds2,    probs["away_win"]),
        ("Over 2.5",           args.odds_o25, probs["over_2.5"]),
        ("Under 2.5",          args.odds_u25, probs["under_2.5"]),
    ]
    if any(o for _, o, _ in bet_markets if o):
        print(f"\n{'='*55}")
        print(f"  VALUE BETTING (post-{TAX_RATE*100:.0f}% tax, {KELLY_FRACTION*100:.0f}% Kelly)")
        print(f"{'='*55}")
        for name, odds, prob in bet_markets:
            if not odds:
                continue
            eff   = 1.0 + (odds - 1.0) * (1.0 - TAX_RATE)
            edge  = (prob - 1.0 / eff) * 100
            stake = kelly_criterion(prob, odds)
            print(f"  {name}: odds {odds:.2f} (eff {eff:.2f}) | "
                  f"model {prob*100:.1f}% | edge {edge:+.1f}%")
            if stake > 0:
                print(f"    >>> VALUE  Kelly stake: {stake*100:.1f}% of bankroll")
            else:
                print(f"    No value")

    print(f"\n{'='*55}\n")