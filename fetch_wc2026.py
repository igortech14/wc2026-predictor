import requests
import json
from bs4 import BeautifulSoup
import re

# Wikipedia page for 2026 FIFA World Cup squads/teams (or use the main tournament page)
# Using the "2026 FIFA World Cup" page which has the qualified teams table.
url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Find the "Qualified teams" section – usually a table with class "wikitable"
# Multiple tables might match; we search for one with headers containing "Team" and "Qualification method"
tables = soup.find_all("table", class_="wikitable")
qualified_teams = []
group_assignments = {}

for table in tables:
    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    if "team" in headers and ("qualification" in headers or "method" in headers):
        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 1:
                team_cell = cols[0]
                # Remove footnotes, links, etc. – just get the text
                team_name = team_cell.get_text(strip=True)
                # Clean up: remove bracketed numbers, notes, etc.
                team_name = re.sub(r'\[.*?\]', '', team_name).strip()
                # Skip rows like "Total teams"
                if team_name and team_name.lower() not in ["total", "teams", ""]:
                    qualified_teams.append(team_name)
        break

# Alternatively, find the "Group stage" section tables to get group assignments
group_letters = [chr(65+i) for i in range(12)]  # A-L
# Wikipedia has a section with group tables; we'll find headers like "Group A"
group_re = re.compile(r'^Group ([A-L])$')
for header in soup.find_all(["h2", "h3", "h4"]):
    m = group_re.match(header.get_text(strip=True))
    if m:
        group = m.group(1)
        # Next element should be the table
        table = header.find_next("table", class_="wikitable")
        if table:
            group_teams = []
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if cols:
                    team_cell = cols[0] if len(cols) > 0 else None
                    if team_cell:
                        team = team_cell.get_text(strip=True)
                        team = re.sub(r'\[.*?\]', '', team).strip()
                        group_teams.append(team)
            group_assignments[group] = group_teams

# Save to files
with open("data/qualified_teams.json", "w", encoding="utf-8") as f:
    json.dump(qualified_teams, f, indent=2)
print(f"Saved {len(qualified_teams)} qualified teams to data/qualified_teams.json")

if group_assignments:
    with open("data/groups.json", "w", encoding="utf-8") as f:
        json.dump(group_assignments, f, indent=2)
    print("Saved group assignments to data/groups.json")
else:
    print("Could not parse group assignments automatically. You may need to provide groups.json manually.")