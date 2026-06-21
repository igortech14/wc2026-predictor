# save as reload_data.py
import sqlite3
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matches.db"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Delete old database
if DB_PATH.exists():
    DB_PATH.unlink()
    print("Deleted old database.")

# Create fresh database
conn = sqlite3.connect(str(DB_PATH))
conn.execute("""
    CREATE TABLE matches (
        date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_score INTEGER,
        away_score INTEGER,
        competition TEXT,
        season INTEGER,
        venue TEXT,
        neutral INTEGER,
        PRIMARY KEY (date, home_team, away_team, competition)
    )
""")
conn.commit()

# Download data
print("Downloading dataset...")
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
response = requests.get(url, headers=HEADERS, timeout=60)
df = pd.read_csv(StringIO(response.text))
print(f"Downloaded {len(df)} matches")

# Clean
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df[df['date'] >= '2010-01-01']
df = df.dropna(subset=['home_score', 'away_score'])
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)
df['neutral'] = df['neutral'].fillna(1).astype(int)

print(f"Filtered to {len(df)} matches from 2010+")

# Insert
new = 0
for _, row in df.iterrows():
    try:
        conn.execute("""
            INSERT INTO matches 
            (date, home_team, away_team, home_score, away_score, competition, season, venue, neutral)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['date'].strftime('%Y-%m-%d'),
            str(row['home_team']).strip(),
            str(row['away_team']).strip(),
            int(row['home_score']),
            int(row['away_score']),
            str(row.get('tournament', 'Unknown')),
            int(row['date'].year),
            str(row.get('city', 'Unknown')),
            int(row['neutral'])
        ))
        new += 1
    except:
        pass

conn.commit()
print(f"Inserted {new} matches")

# Verify
count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
print(f"Total in DB: {count}")
conn.close()
print("Done! Now run: python train.py")