import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import predict

DATA_DIR      = Path("data")
SCHEDULE_PATH = DATA_DIR / "enriched_schedule.json"
ODDS_PATH     = DATA_DIR / "odds.json"

HOST_NATIONS = {'Mexico', 'United States', 'Canada'}


def load_schedule():
    try:
        with open(SCHEDULE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schedule file not found at {SCHEDULE_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse schedule file: {e}")
        sys.exit(1)


def load_odds(date_str):
    if not ODDS_PATH.exists():
        return {}
    with open(ODDS_PATH, 'r', encoding='utf-8') as f:
        all_odds = json.load(f)
    return all_odds.get(date_str, {})


def parse_odds_entry(entry):
    return {
        'odds1':    entry.get('odds1'),
        'oddsX':    entry.get('oddsX'),
        'odds2':    entry.get('odds2'),
        'odds_o25': entry.get('odds_o25'),
        'odds_u25': entry.get('odds_u25'),
    }


def interactive_odds(match):
    print(f"\n  Enter odds for {match['team1']} vs {match['team2']} "
          f"(press Enter to skip):")
    try:
        odds1 = input("    odds1 (home win): ").strip()
        if not odds1:
            return None
        oddsX    = input("    oddsX (draw): ").strip()
        odds2    = input("    odds2 (away win): ").strip()
        odds_o25 = input("    odds over 2.5: ").strip()
        odds_u25 = input("    odds under 2.5: ").strip()
        res = {}
        if odds1:    res['odds1']    = float(odds1)
        if oddsX:    res['oddsX']    = float(oddsX)
        if odds2:    res['odds2']    = float(odds2)
        if odds_o25: res['odds_o25'] = float(odds_o25)
        if odds_u25: res['odds_u25'] = float(odds_u25)
        return res if res else None
    except (ValueError, EOFError):
        return None


# Add a minimum edge threshold — ignore edges under 0.5%
MIN_EDGE_PCT = 0.5

def edge_and_stake(prob, odds_val):
    if not odds_val or odds_val <= 1:
        return 0, 0, 0
    effective_odds = 1 + (odds_val - 1) * (1 - predict.TAX_RATE)
    if effective_odds <= 1:
        return 0, 0, 0
    true_edge = prob - 1 / effective_odds
    if true_edge * 100 < MIN_EDGE_PCT:   # filter out noise
        return 0, 0, 0
    stake = predict.kelly_criterion(prob, odds_val)
    return true_edge * 100, stake, effective_odds


def print_detailed_probs(team1, team2, probs, venue):
    most_likely = probs['top_scores'][0][0] if probs['top_scores'] else "0-0"
    print(f"\n{'='*55}")
    print(f"  {team1} vs {team2}")
    print(f"  Venue: {venue}")
    print(f"{'='*55}")
    print(f"  Expected Goals: {probs['lambda_home']:.2f} - {probs['lambda_away']:.2f}")
    print(f"\n  Match Result (1X2):")
    print(f"    {team1} Win:  {probs['home_win']*100:5.1f}%")
    print(f"    Draw:             {probs['draw']*100:5.1f}%")
    print(f"    {team2} Win:  {probs['away_win']*100:5.1f}%")
    print(f"\n  Over/Under Goals:")
    print(f"    Over 1.5:   {probs['over_1.5']*100:5.1f}%")
    print(f"    Under 1.5:  {probs['under_1.5']*100:5.1f}%")
    print(f"    Over 2.5:   {probs['over_2.5']*100:5.1f}%")
    print(f"    Under 2.5:  {probs['under_2.5']*100:5.1f}%")
    print(f"    Over 3.5:   {probs['over_3.5']*100:5.1f}%")
    print(f"    Under 3.5:  {probs['under_3.5']*100:5.1f}%")
    print(f"\n  Both Teams to Score:")
    print(f"    Yes:  {probs['btts']*100:5.1f}%")
    print(f"    No:   {probs['btts_no']*100:5.1f}%")
    print(f"\n  Most Likely Score: {most_likely}")
    print("  Top Score Probabilities:")
    for score, prob in probs['top_scores']:
        print(f"    {score}: {prob*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Daily Bet-Slip with full detailed report')
    parser.add_argument('--date', help='Date YYYY-MM-DD (default: next matchday)')
    parser.add_argument('--odds-file', help='Path to odds JSON file (default: data/odds.json)')
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, default=True,
                        help='Ask for odds interactively (default: on, use --no-interactive to disable)')
    args = parser.parse_args()

    schedule = load_schedule()
    matches_by_date = {}
    for m in schedule:
        matches_by_date.setdefault(m['date'], []).append(m)

    today       = datetime.now().strftime('%Y-%m-%d')
    target_date = args.date
    if not target_date:
        for d in sorted(matches_by_date.keys()):
            if d >= today:
                target_date = d
                break
        if not target_date:
            print("No upcoming matches found in schedule.")
            return

    if target_date not in matches_by_date:
        print(f"No matches scheduled for {target_date}.")
        return

    matches_today = matches_by_date[target_date]

    # Build odds lookup keyed by original schedule order (team1, team2)
    odds_lookup = {}
    if not args.interactive:
        for entry in load_odds(target_date):
            key = (entry.get('team1'), entry.get('team2'))
            odds_lookup[key] = entry

    value_bets = []

    print(f"\n{'='*60}")
    print(f"  DAILY BET-SLIP - {target_date}")
    print(f"{'='*60}")

    for match in matches_today:
        # FIX: preserve original schedule names for odds lookup before any swapping
        orig_team1 = match['team1']
        orig_team2 = match['team2']

        team1   = orig_team1
        team2   = orig_team2
        rest1   = match.get('team1_rest')
        rest2   = match.get('team2_rest')
        travel1 = match.get('team1_travel', 0)
        travel2 = match.get('team2_travel', 0)

        home_team = team1 if team1 in HOST_NATIONS else (team2 if team2 in HOST_NATIONS else None)

        # Swap so host nation is always team1 for predict.py
        swapped = False
        if home_team == team2:
            team1,   team2   = team2,   team1
            rest1,   rest2   = rest2,   rest1
            travel1, travel2 = travel2, travel1
            swapped = True

        neutral = (home_team is None)

        probs = predict.outcome_probs(
            team1, team2,
            neutral  = neutral,
            rest1    = rest1,   rest2    = rest2,
            travel1  = travel1, travel2  = travel2,
            match_date = target_date,
        )

        # Swap probabilities back to original display order
        if swapped:
            probs['home_win'],    probs['away_win']    = probs['away_win'],    probs['home_win']
            probs['lambda_home'], probs['lambda_away'] = probs['lambda_away'], probs['lambda_home']
            team1, team2 = team2, team1

        venue = "home" if home_team else "neutral"
        print_detailed_probs(team1, team2, probs, venue)

        # FIX: odds lookup always uses original schedule order, never the swapped names
        match_odds = None
        if args.interactive:
            match_odds = interactive_odds(match)
        else:
            match_odds = odds_lookup.get((orig_team1, orig_team2))

        if not match_odds:
            print("  (No odds provided - value assessment skipped)")
            continue

        odds      = parse_odds_entry(match_odds)
        bets_found = []

        markets = [
            ('Home Win',  odds.get('odds1'),    probs['home_win']),
            ('Draw',      odds.get('oddsX'),    probs['draw']),
            ('Away Win',  odds.get('odds2'),    probs['away_win']),
            ('Over 2.5',  odds.get('odds_o25'), probs['over_2.5']),
            ('Under 2.5', odds.get('odds_u25'), probs['under_2.5']),
        ]
        for bet_name, odds_val, prob in markets:
            if odds_val:
                edge, stake, eff = edge_and_stake(prob, odds_val)
                if edge > 0:
                    bets_found.append((bet_name, odds_val, prob, edge, stake, eff))

        if bets_found:
            print("\n  VALUE ASSESSMENT:")
            for bet_name, odds_val, prob, edge, stake, eff in bets_found:
                print(f"    ✅ {bet_name}: odds={odds_val:.2f} (eff={eff:.2f}), "
                      f"model={prob*100:.1f}%, edge={edge:+.1f}%, "
                      f"Kelly stake={stake*100:.1f}%")
                value_bets.append((team1, team2, bet_name, odds_val, prob, edge, stake))
        else:
            print("  No value bets (post-tax).")

    if value_bets:
        print(f"\n{'='*60}")
        print("  VALUE BET SUMMARY (sorted by edge)")
        print(f"{'='*60}")
        value_bets.sort(key=lambda x: x[5], reverse=True)
        for t1, t2, bet_name, odds_val, prob, edge, stake in value_bets:
            print(f"  {t1} vs {t2}: {bet_name} @ {odds_val:.2f} | "
                  f"Edge {edge:+.1f}% | Stake {stake*100:.1f}%")
    else:
        print(f"\n  No value bets found for any match.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()