import sqlite3
import re
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matches.db"
SCHEDULE_PATH = DATA_DIR / "schedule.json"

def find_teams_in_schedule(team1, team2):
    """Search schedule.json for a match containing these two teams (any order)."""
    if not SCHEDULE_PATH.exists():
        return None, None
    with open(SCHEDULE_PATH, 'r', encoding='utf-8') as f:
        schedule = json.load(f)
    for match in schedule:
        if (match['team1'] == team1 and match['team2'] == team2) or \
           (match['team1'] == team2 and match['team2'] == team1):
            return match['date'], match.get('competition', 'World Cup 2026')
    return None, None

def insert_match(conn, home, away, home_score, away_score, match_date, competition):
    conn.execute("""
        INSERT OR REPLACE INTO matches
        (date, home_team, away_team, home_score, away_score, competition, season, venue, neutral)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (match_date, home, away, home_score, away_score, competition, int(match_date[:4]), "auto", 1))
    conn.commit()

def process_line(line, conn):
    # Match pattern: "Team A 2-1 Team B" or "Team A 2 - 1 Team B"
    pattern = r"^(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+)$"
    m = re.match(pattern, line.strip())
    if not m:
        print(f"  ❌ Could not parse: {line}")
        return
    team1_raw = m.group(1).strip()
    goals1 = int(m.group(2))
    goals2 = int(m.group(3))
    team2_raw = m.group(4).strip()
    home_team = team1_raw
    away_team = team2_raw
    home_goals = goals1
    away_goals = goals2

    match_date, competition = find_teams_in_schedule(home_team, away_team)
    if not match_date:
        match_date = datetime.now().strftime('%Y-%m-%d')
        competition = 'Friendly'
        print(f"  ⚠ Match not in schedule – assuming {competition} on {match_date}")

    insert_match(conn, home_team, away_team, home_goals, away_goals, match_date, competition)
    print(f"  ✅ {home_team} {home_goals}-{away_goals} {away_team}  ({match_date}, {competition})")

def main():
    import sys
    print("World Cup Quick Updater")
    print("Paste your results (one per line in format 'Team A 2-1 Team B'), then press Enter twice.")
    print("Example: Mexico 1-0 South Africa")
    print("Type 'done' to finish.\n")

    conn = sqlite3.connect(str(DB_PATH))
    lines = []
    for line in sys.stdin:
        if line.strip().lower() == 'done':
            break
        lines.append(line.strip())

    if not lines:
        print("No input.")
        return

    for line in lines:
        process_line(line, conn)

    conn.close()
    print("\nAll done. Now run: python train.py   (and optionally python simulate.py)")

if __name__ == "__main__":
    main()