import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import requests
from io import StringIO

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "matches.db"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

def init_db():
    """Create the matches table."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            tournament TEXT,
            match_type TEXT,
            venue TEXT,
            neutral INTEGER,
            PRIMARY KEY (date, home_team, away_team, tournament)
        )
    """)
    conn.commit()
    return conn

def load_international_dataset(conn):
    """
    Load from the comprehensive international results dataset.
    This contains 49,000+ international matches from 1872-2024.
    """
    logger.info("=" * 60)
    logger.info("LOADING INTERNATIONAL MATCH DATASET")
    logger.info("=" * 60)
    
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    logger.info("Downloading dataset (49,000+ matches)...")
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    
    df = pd.read_csv(StringIO(response.text))
    logger.info(f"Downloaded {len(df)} total matches")
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Filter for recent matches (2010 onwards for World Cup relevance)
    df = df[df['date'] >= '2010-01-01']
    logger.info(f"Filtered to {len(df)} matches from 2010 onwards")
    
    # Clean team names
    df['home_team'] = df['home_team'].astype(str).str.strip()
    df['away_team'] = df['away_team'].astype(str).str.strip()
    
    # Convert scores
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    
    # Prepare for database
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df['tournament'] = df['tournament'].fillna('Friendly')
    df['match_type'] = df['tournament'].apply(
        lambda x: 'Friendly' if 'friendly' in str(x).lower() else 'Competitive'
    )
    df['neutral'] = df['neutral'].fillna(1).astype(int)
    df['venue'] = df['city'].fillna('Unknown') if 'city' in df.columns else 'Unknown'
    
    # Insert into database in batches
    batch_size = 500
    new_count = 0
    skip_count = 0
    
    logger.info("Inserting matches into database...")
    
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size]
        
        for _, row in batch.iterrows():
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO matches 
                    (date, home_team, away_team, home_score, away_score, tournament, match_type, venue, neutral)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['date_str'],
                    row['home_team'],
                    row['away_team'],
                    int(row['home_score']),
                    int(row['away_score']),
                    row['tournament'],
                    row['match_type'],
                    row['venue'],
                    int(row['neutral'])
                ))
                if conn.total_changes > 0:
                    new_count += 1
                else:
                    skip_count += 1
            except Exception:
                skip_count += 1
                continue
        
        conn.commit()
        
        if (start // batch_size) % 10 == 0:
            logger.info(f"  Processed {start}/{len(df)}... ({new_count} new, {skip_count} duplicates)")
    
    logger.info(f"Done! Added {new_count} new matches ({skip_count} already existed)")
    return new_count

def main():
    logger.info("=" * 60)
    logger.info("  WORLD CUP PREDICTOR - DATA FETCHER v3")
    logger.info("=" * 60)
    
    conn = init_db()
    
    total_new = load_international_dataset(conn)
    
    # Show stats using direct cursor (avoids pandas column name issues)
    cursor = conn.execute("SELECT COUNT(*) FROM matches")
    total_matches = cursor.fetchone()[0]
    
    cursor = conn.execute("""
        SELECT COUNT(DISTINCT team) FROM (
            SELECT home_team AS team FROM matches
            UNION
            SELECT away_team AS team FROM matches
        )
    """)
    total_teams = cursor.fetchone()[0]
    
    # Get column names to check schema
    cursor = conn.execute("PRAGMA table_info(matches)")
    columns = [row[1] for row in cursor.fetchall()]
    logger.info(f"Database columns: {columns}")
    
    # Get recent matches
    cursor = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM matches
        ORDER BY date DESC
        LIMIT 10
    """)
    recent = cursor.fetchall()
    
    # Count by tournament/competition if column exists
    if 'tournament' in columns:
        cursor = conn.execute("""
            SELECT tournament, COUNT(*) as count 
            FROM matches 
            GROUP BY tournament 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_comp = cursor.fetchall()
    elif 'competition' in columns:
        cursor = conn.execute("""
            SELECT competition, COUNT(*) as count 
            FROM matches 
            GROUP BY competition 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_comp = cursor.fetchall()
    else:
        top_comp = None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  DATA FETCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  New matches added: {total_new}")
    logger.info(f"  Total matches in DB: {total_matches}")
    logger.info(f"  Unique teams: {total_teams}")
    
    if top_comp:
        logger.info(f"\n  Top competitions:")
        for comp, count in top_comp:
            logger.info(f"    {comp}: {count} matches")
    
    logger.info(f"\n  Most recent matches:")
    for row in recent:
        logger.info(f"    {row[0]}: {row[1]} {row[2]}-{row[3]} {row[4]}")
    
    logger.info(f"\n  Ready to train! Run: python train.py")
    
    conn.close()

if __name__ == "__main__":
    main()