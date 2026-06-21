import requests
import csv
import json
from pathlib import Path
from io import StringIO

DATA_DIR = Path("data")
ELO_PATH = DATA_DIR / "global_elo.json"

URL = "https://www.eloratings.net/2026_FIFA_World_Cup_qualification_elo_ratings.csv"

def main():
    print("Downloading World Football Elo Ratings (CSV)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(URL, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to download CSV (status {resp.status_code}).")
        return

    # Parse CSV
    reader = csv.DictReader(StringIO(resp.text))
    elo_dict = {}
    for row in reader:
        # Columns: Rank, Country, Points, etc.
        country = row.get('Country') or row.get('Team')
        points = row.get('Points') or row.get('Rating')
        if country and points:
            try:
                elo_dict[country.strip()] = float(points)
            except:
                pass

    with open(ELO_PATH, 'w', encoding='utf-8') as f:
        json.dump(elo_dict, f, indent=2)

    print(f"Saved {len(elo_dict)} Elo ratings to {ELO_PATH}")
    # Top 10
    sorted_elo = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Elo Ratings:")
    for team, rating in sorted_elo:
        print(f"  {team:25s} {rating:.0f}")

if __name__ == "__main__":
    main()