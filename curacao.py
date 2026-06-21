# find_mismatches.py
import sqlite3, json
from pathlib import Path

with open("data/qualified_teams.json") as f:
    qualified = set(json.load(f))

conn = sqlite3.connect("data/matches.db")
db_teams = set(
    r[0] for r in conn.execute(
        "SELECT DISTINCT home_team FROM matches UNION SELECT DISTINCT away_team FROM matches"
    ).fetchall()
)
conn.close()

mismatches = [t for t in db_teams if t not in qualified]
print("DB names not in qualified_teams.json:")
for t in sorted(mismatches):
    # only show if something similar exists in qualified
    close = [q for q in qualified if q.lower().replace('ç','c').replace('ü','u').replace('ö','o') 
             == t.lower().replace('ç','c').replace('ü','u').replace('ö','o')]
    if close:
        print(f"  DB: '{t}'  →  qualified: '{close[0]}'")