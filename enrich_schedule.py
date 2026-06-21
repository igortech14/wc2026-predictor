import json
import math
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")
SCHEDULE_PATH = DATA_DIR / "schedule.json"
VENUE_COORDS_PATH = DATA_DIR / "venue_coords.json"
NAME_MAP_PATH = DATA_DIR / "team_name_map.json"
ENRICHED_PATH = DATA_DIR / "enriched_schedule.json"

def haversine(coord1, coord2):
    """Calculate distance in km between two [lat, lon] pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    schedule = load_json(SCHEDULE_PATH)
    venue_coords = load_json(VENUE_COORDS_PATH)
    name_map = load_json(NAME_MAP_PATH)

    # Normalise team names in schedule
    for match in schedule:
        match['team1'] = name_map.get(match['team1'], match['team1'])
        match['team2'] = name_map.get(match['team2'], match['team2'])

    # Sort by date (just in case)
    schedule.sort(key=lambda m: m['date'])

    # Track last match info for each team: (date, venue_city)
    team_last = {}

    enriched = []
    for match in schedule:
        date = datetime.strptime(match['date'], '%Y-%m-%d')
        enriched_match = match.copy()

        for team_key in ['team1', 'team2']:
            team = match[team_key]
            if team in team_last:
                last_date, last_venue = team_last[team]
                rest_days = (date - last_date).days
                # Travel distance: if same city, 0, else compute
                if match['venue_city'] == last_venue:
                    travel_km = 0.0
                else:
                    coord_prev = venue_coords.get(last_venue, [0,0])
                    coord_curr = venue_coords.get(match['venue_city'], [0,0])
                    travel_km = haversine(coord_prev, coord_curr)
            else:
                # First match: assume 14 days rest, 0 travel
                rest_days = 14
                travel_km = 0.0

            enriched_match[f'{team_key}_rest'] = rest_days
            enriched_match[f'{team_key}_travel'] = travel_km

        # Update team_last for both teams
        team_last[match['team1']] = (date, match['venue_city'])
        team_last[match['team2']] = (date, match['venue_city'])

        enriched.append(enriched_match)

    with open(ENRICHED_PATH, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)
    print(f"Enriched schedule saved to {ENRICHED_PATH} with {len(enriched)} matches.")

if __name__ == "__main__":
    main()