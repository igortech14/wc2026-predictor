import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import re
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

DATA_DIR = Path("data")
QUALIFIED_PATH = DATA_DIR / "qualified_teams.json"
XG_PATH = DATA_DIR / "xg_data.json"
TEAM_INDEX_URL = "https://fbref.com/en/teams/"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Mapping our DB team names to FBref naming (often matches, but some differ)
# We'll update after loading index; this is just a fallback.
FALLBACK_NAME_MAP = {
    "Korea Republic": "Korea Republic",
    "Czech Republic": "Czech Republic",
    "Turkey": "Türkiye",          # FBref uses "Türkiye"
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cape Verde",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Uzbekistan": "Uzbekistan",
    "Jordan": "Jordan",
    "Haiti": "Haiti",
    "United States": "United States",
    # Add any others as needed
}

def fetch_team_index():
    """Return dict mapping normalized team name -> fbref squad ID."""
    print("Fetching FBref national team index...")
    try:
        resp = requests.get(TEAM_INDEX_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        mapping = {}
        # Look for links containing "/en/squads/" and a team name
        for link in soup.select("a[href*='/en/squads/']"):
            href = link['href']
            # Extract ID: /en/squads/xxxxxxxx/...
            match = re.search(r'/en/squads/([a-f0-9]{8})/', href)
            if match:
                team_id = match.group(1)
                team_name = link.get_text(strip=True)
                # Normalize for our use (lowercase, but we'll keep original)
                mapping[team_name] = team_id
        print(f"  Found {len(mapping)} national teams.")
        return mapping
    except Exception as e:
        print(f"  Failed to fetch index: {e}")
        return {}

def map_our_team_to_fbref(our_name, index_mapping):
    """Attempt to find FBref name/ID for our team name."""
    # Direct match
    if our_name in index_mapping:
        return index_mapping[our_name]
    # Try with fallback map
    fb_name = FALLBACK_NAME_MAP.get(our_name, our_name)
    if fb_name in index_mapping:
        return index_mapping[fb_name]
    # Try case-insensitive
    our_lower = our_name.lower()
    for name, tid in index_mapping.items():
        if name.lower() == our_lower:
            return tid
    # Try "United States" -> "USA"? FBref uses "United States" usually.
    # If still not found, return None
    return None

def scrape_xg_for_team(team_name, team_id):
    """Scrape shooting match log for a given team, return list of dicts."""
    url = f"https://fbref.com/en/squads/{team_id}/matchlogs/all_comps/shooting"
    print(f"  Scraping {team_name} ({team_id})...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}")
            return []
        soup = BeautifulSoup(resp.content, 'html.parser')
        table = soup.find('table', id='matchlogs_for')
        if not table:
            print("    No shooting table found.")
            return []
        df = pd.read_html(str(table))[0]
        # Flatten multi-level columns
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Find columns with xG and xGA (may be "xG_Expected" etc.)
        xg_col = [c for c in df.columns if 'xG' in c and 'xGA' not in c]
        xga_col = [c for c in df.columns if 'xGA' in c]
        if not xg_col:
            print("    No xG column found.")
            return []
        # Extract Date, Opponent, xG, xGA
        df = df.rename(columns={
            'Date': 'Date',
            'Opponent': 'Opponent',
            xg_col[0]: 'xG',
            xga_col[0] if xga_col else xg_col[0]: 'xGA'
        })
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'xG'])
        df = df[df['Date'] >= '2023-01-01']  # Only recent matches
        results = []
        for _, row in df.iterrows():
            results.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'team': team_name,
                'opponent': str(row['Opponent']).strip(),
                'xg_for': float(row['xG']),
                'xg_against': float(row['xGA'] if 'xGA' in row and pd.notna(row['xGA']) else row['xG'])
            })
        print(f"    Got {len(results)} matches with xG since 2023.")
        return results
    except Exception as e:
        print(f"    Error: {e}")
        return []

def main():
    print("=" * 60)
    print("xG Scraper – Shooting Logs from FBref")
    print("=" * 60)

    # Load our qualified teams
    with open(QUALIFIED_PATH, 'r', encoding='utf-8') as f:
        qualified = json.load(f)
    print(f"Loaded {len(qualified)} World Cup teams.")

    # Fetch team index
    index_map = fetch_team_index()
    if not index_map:
        print("Cannot proceed without team IDs. Aborting.")
        with open(XG_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    all_xg = []
    for team in sorted(qualified):
        tid = map_our_team_to_fbref(team, index_map)
        if not tid:
            print(f"  No FBref ID for {team} – skipping.")
            continue
        xg_matches = scrape_xg_for_team(team, tid)
        all_xg.extend(xg_matches)
        time.sleep(3)   # be nice to the server

    # Save
    with open(XG_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_xg, f, indent=2)
    print(f"\nSaved {len(all_xg)} total xG entries to {XG_PATH}")
    if len(all_xg) < 100:
        print("Low coverage – some teams may be missing. The model will still work with actual goals.")

if __name__ == "__main__":
    main()