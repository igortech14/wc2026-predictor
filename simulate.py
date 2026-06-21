import json
import pickle
import numpy as np
from itertools import combinations
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

DATA_DIR               = Path("data")
MODEL_PATH             = DATA_DIR / "model.pkl"
GROUPS_PATH            = DATA_DIR / "groups.json"
ENRICHED_SCHEDULE_PATH = DATA_DIR / "enriched_schedule.json"
BRACKET_PATH           = DATA_DIR / "knockout_bracket.json"
GLOBAL_ELO_PATH        = DATA_DIR / "global_elo.json"

N_SIMULATIONS       = 10_000
REST_IMPORTANCE     = 0.01
TRAVEL_KM_PER_PCT   = 500

# Elo parameters — must match predict.py exactly
ELO_BLEND      = 0.75
ELO_GOAL_RATIO = 0.5 / 100


# ── Name resolution ──────────────────────────────────────────────────────────
def load_name_map() -> dict:
    """Single source of truth — same file used by train.py and predict.py."""
    path = DATA_DIR / "team_name_map.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_NAME_MAP = load_name_map()

def resolve_name(team: str) -> str:
    return _NAME_MAP.get(team, team)


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_global_elo() -> dict:
    if not GLOBAL_ELO_PATH.exists():
        return {}
    with open(GLOBAL_ELO_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Build lookup keyed by the canonical model name
    elo = {}
    for team, rating in raw.items():
        elo[resolve_name(team)] = rating
        elo[team] = rating          # also keep original as fallback
    return elo

def load_heat_factors():
    path = DATA_DIR / "heat_factors.json"
    if not path.exists():
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.loads(f.read().strip())
        return data.get("venues", {}), data.get("team_tolerance", {})
    except Exception as e:
        print(f"  Warning: could not load heat_factors.json: {e}", file=sys.stderr)
        return {}, {}


# ── Core match simulator ──────────────────────────────────────────────────────
def simulate_match(team1, team2, model, elo_dict,
                   rest1=None, rest2=None, travel1=0, travel2=0,
                   neutral=True, venue_city=None, heat_factors=None):
    """
    Simulate a single match and return (goals1, goals2).
    Elo adjustment is multiplicative (log-space) — consistent with predict.py.
    """
    tp       = model["team_params"]
    home_adv = 0.0 if neutral else model["home_adv"]

    # Fall back to average parameters if team not in model
    p1 = tp.get(team1, tp.get(resolve_name(team1), {"att": 0.0, "def": 0.0}))
    p2 = tp.get(team2, tp.get(resolve_name(team2), {"att": 0.0, "def": 0.0}))

    lambda1 = np.exp(p1["att"] + p2["def"] + home_adv)
    lambda2 = np.exp(p2["att"] + p1["def"])

    # Rest / travel
    if rest1 is not None:
        rest_f  = 1.0 - (8 - rest1) * REST_IMPORTANCE if rest1 < 8 else 1.0
        trav_f  = (1.0 - (travel1 - 500) / TRAVEL_KM_PER_PCT * 0.01
                   if travel1 > 500 else 1.0)
        lambda1 *= max(0.8, rest_f * trav_f)
    if rest2 is not None:
        rest_f  = 1.0 - (8 - rest2) * REST_IMPORTANCE if rest2 < 8 else 1.0
        trav_f  = (1.0 - (travel2 - 500) / TRAVEL_KM_PER_PCT * 0.01
                   if travel2 > 500 else 1.0)
        lambda2 *= max(0.8, rest_f * trav_f)

    # FIX: Elo in log space (multiplicative) — was additive before
    elo1   = elo_dict.get(team1, elo_dict.get(resolve_name(team1), 1500))
    elo2   = elo_dict.get(team2, elo_dict.get(resolve_name(team2), 1500))
    diff   = (elo1 - elo2) * ELO_GOAL_RATIO
    factor = np.exp(diff * ELO_BLEND)
    lambda1 = np.clip(lambda1 * factor,  0.5, 8.0)
    lambda2 = np.clip(lambda2 / factor,  0.5, 8.0)

    # Heat / venue
    if heat_factors and venue_city:
        venues, tolerance = heat_factors
        if venue_city in venues:
            vf      = venues[venue_city]
            t1_tol  = tolerance.get(team1, tolerance.get(resolve_name(team1), 1.0))
            t2_tol  = tolerance.get(team2, tolerance.get(resolve_name(team2), 1.0))
            lambda1 = np.clip(lambda1 * vf * t1_tol, 0.5, 8.0)
            lambda2 = np.clip(lambda2 * vf * t2_tol, 0.5, 8.0)

    return int(np.random.poisson(lambda1)), int(np.random.poisson(lambda2))


# ── Group stage ───────────────────────────────────────────────────────────────
def simulate_group(teams, model, elo_dict, enriched_schedule, heat_factors):
    standings = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

    for t1, t2 in combinations(teams, 2):
        rest1 = rest2 = None
        travel1 = travel2 = 0
        venue   = None

        if enriched_schedule:
            match = next(
                (m for m in enriched_schedule
                 if (m["team1"] == t1 and m["team2"] == t2) or
                    (m["team1"] == t2 and m["team2"] == t1)),
                None
            )
            if match:
                if match["team1"] == t1:
                    rest1, rest2     = match.get("team1_rest"), match.get("team2_rest")
                    travel1, travel2 = match.get("team1_travel", 0), match.get("team2_travel", 0)
                else:
                    rest1, rest2     = match.get("team2_rest"), match.get("team1_rest")
                    travel1, travel2 = match.get("team2_travel", 0), match.get("team1_travel", 0)
                venue = match.get("venue_city")

        g1, g2 = simulate_match(t1, t2, model, elo_dict,
                                 rest1=rest1, rest2=rest2,
                                 travel1=travel1, travel2=travel2,
                                 venue_city=venue, heat_factors=heat_factors)

        standings[t1]["gf"] += g1;  standings[t1]["ga"] += g2
        standings[t2]["gf"] += g2;  standings[t2]["ga"] += g1

        if   g1 > g2: standings[t1]["pts"] += 3
        elif g2 > g1: standings[t2]["pts"] += 3
        else:         standings[t1]["pts"] += 1;  standings[t2]["pts"] += 1

    ranked = sorted(
        standings.items(),
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
        reverse=True
    )
    return [t for t, _ in ranked], standings


# ── Third-place qualification ─────────────────────────────────────────────────
def get_best_thirds(group_rankings, group_standings):
    thirds = []
    for gname, ranking in group_rankings.items():
        third = ranking[2]
        st    = group_standings[gname][third]
        thirds.append((third, gname, st["pts"], st["gf"] - st["ga"], st["gf"]))
    thirds.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    return thirds[:8]

def pop_third(allowed_groups, available_thirds):
    for i, (team, group, *_) in enumerate(available_thirds):
        if group in allowed_groups:
            return available_thirds.pop(i)[0]
    # fallback: best remaining regardless of group
    if available_thirds:
        return available_thirds.pop(0)[0]
    raise RuntimeError(f"No third-place team for groups {allowed_groups}")


# ── Bracket resolution ────────────────────────────────────────────────────────
def resolve_team(placeholder, direct_slots, available_thirds):
    ph = placeholder.strip()
    if "winners" in ph:
        group = ph.split()[1]
        return direct_slots[f"Group {group} winners"]
    if "runners-up" in ph:
        group = ph.split()[1]
        return direct_slots[f"Group {group} runners-up"]
    if "third place" in ph or "3rd place" in ph:
        if "Group " in ph:
            part = ph.split("Group ")[1].split(" third")[0].split(" 3rd")[0]
            allowed = [g.strip() for g in part.split("/")]
        else:
            allowed = ph.split()[-1].split("/")
        return pop_third(allowed, available_thirds)
    raise ValueError(f"Unknown bracket placeholder: {ph!r}")

def resolve_prev(placeholder, match_results):
    if "Winner Match" in placeholder:
        mid = int(placeholder.split("Winner Match ")[1])
        return match_results[mid]["winner"]
    if "Loser Match" in placeholder:
        mid = int(placeholder.split("Loser Match ")[1])
        return match_results[mid]["loser"]
    raise ValueError(f"Unknown result placeholder: {placeholder!r}")


# ── Main simulation loop ──────────────────────────────────────────────────────
def simulate_tournament(model):
    with open(GROUPS_PATH)  as f: groups  = json.load(f)
    with open(BRACKET_PATH) as f: bracket = json.load(f)

    enriched = (json.load(open(ENRICHED_SCHEDULE_PATH))
                if ENRICHED_SCHEDULE_PATH.exists() else None)

    heat_factors = load_heat_factors()
    elo_dict     = load_global_elo()

    # Warn if any team's Elo is missing (silent 1500 fallback is dangerous)
    all_teams = [t for g in groups.values() for t in g]
    missing_elo = [t for t in all_teams
                   if t not in elo_dict and resolve_name(t) not in elo_dict]
    if missing_elo:
        print(f"  Warning: no Elo for {missing_elo} — defaulting to 1500", file=sys.stderr)

    stats = {team: {k: 0 for k in
                    ("win_group","top2","advance","R16","QF","SF","Final","Winner")}
             for team in all_teams}

    round_order = ["R32","R16","QF","SF","Third place","Final"]

    for sim in range(N_SIMULATIONS):
        if sim % 1000 == 0:
            print(f"  Simulation {sim:,}/{N_SIMULATIONS:,}")

        # ── Group stage
        group_rankings  = {}
        group_standings = {}
        for gname, gteams in groups.items():
            ranking, st = simulate_group(gteams, model, elo_dict,
                                         enriched, heat_factors)
            group_rankings[gname]  = ranking
            group_standings[gname] = st

        # ── Best 8 third-place teams
        available_thirds = get_best_thirds(group_rankings, group_standings)

        # ── Direct slots (group winners / runners-up)
        direct_slots = {}
        for gname, ranking in group_rankings.items():
            direct_slots[f"Group {gname} winners"]    = ranking[0]
            direct_slots[f"Group {gname} runners-up"] = ranking[1]

        # ── Knockout bracket
        match_results = {}
        for match in sorted(bracket, key=lambda m: round_order.index(m["round"])):
            mid = match["match_id"]
            if match["round"] == "R32":
                t1 = resolve_team(match["team1_placeholder"], direct_slots, available_thirds)
                t2 = resolve_team(match["team2_placeholder"], direct_slots, available_thirds)
            else:
                t1 = resolve_prev(match["team1_placeholder"], match_results)
                t2 = resolve_prev(match["team2_placeholder"], match_results)

            g1, g2 = simulate_match(t1, t2, model, elo_dict, neutral=True)

            if   g1 > g2: winner, loser = t1, t2
            elif g2 > g1: winner, loser = t2, t1
            else:
                winner = t1 if np.random.rand() < 0.5 else t2
                loser  = t2 if winner == t1 else t1

            match_results[mid] = {"winner": winner, "loser": loser}

            round_stat = {
                "R32":         "R16",
                "R16":         "QF",
                "QF":          "SF",
                "SF":          "Final",
                "Final":       "Winner",
            }.get(match["round"])
            if round_stat:
                stats[winner][round_stat] += 1

        # ── Group stage tallies
        advancers = set()
        for gname, ranking in group_rankings.items():
            stats[ranking[0]]["win_group"] += 1
            stats[ranking[0]]["top2"]      += 1
            stats[ranking[1]]["top2"]      += 1
            advancers.add(ranking[0])
            advancers.add(ranking[1])
        for team, *_ in get_best_thirds(group_rankings, group_standings):
            advancers.add(team)
        for team in advancers:
            stats[team]["advance"] += 1

    # ── Print results
    print("\n" + "=" * 82)
    print(f"  TOURNAMENT SIMULATION  ({N_SIMULATIONS:,} trials)")
    print("=" * 82)
    header = (f"  {'Team':<24} {'Win Grp':>7} {'Top2':>7} {'R32':>7} "
              f"{'R16':>7} {'QF':>7} {'SF':>7} {'Final':>7} {'Champ':>7}")
    print(header)
    print("  " + "-" * 78)

    for team, s in sorted(stats.items(), key=lambda x: x[1]["Winner"], reverse=True):
        vals = [
            s["win_group"] / N_SIMULATIONS * 100,
            s["top2"]      / N_SIMULATIONS * 100,
            s["advance"]   / N_SIMULATIONS * 100,
            s["R16"]       / N_SIMULATIONS * 100,
            s["QF"]        / N_SIMULATIONS * 100,
            s["SF"]        / N_SIMULATIONS * 100,
            s["Final"]     / N_SIMULATIONS * 100,
            s["Winner"]    / N_SIMULATIONS * 100,
        ]
        print(f"  {team:<24} " + "  ".join(f"{v:6.1f}%" for v in vals))

    print("  " + "-" * 78)
    print("  R32 = advanced from groups | R16 = reached last 16 | etc.")
    print("=" * 82)


if __name__ == "__main__":
    simulate_tournament(load_model())