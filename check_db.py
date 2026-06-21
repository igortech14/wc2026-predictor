# save as check_db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/matches.db")

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Check what tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {tables}")

if tables:
    # Check columns
    cursor.execute(f"PRAGMA table_info({tables[0][0]})")
    columns = cursor.fetchall()
    print(f"\nColumns in '{tables[0][0]}':")
    for col in columns:
        print(f"  {col}")
    
    # Count rows
    cursor.execute(f"SELECT COUNT(*) FROM {tables[0][0]}")
    count = cursor.fetchone()[0]
    print(f"\nRow count: {count}")
    
    if count > 0:
        # Show first row
        cursor.execute(f"SELECT * FROM {tables[0][0]} LIMIT 3")
        rows = cursor.fetchall()
        print(f"\nFirst 3 rows:")
        for row in rows:
            print(f"  {row}")
else:
    print("No tables found!")

conn.close()