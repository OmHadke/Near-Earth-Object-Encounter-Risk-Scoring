# import requests

# API_KEY = "R4oF1wMa57H2MZm3r3u87kL28wEa6soL6BMABbXR"
# url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date=2025-10-01&end_date=2025-10-03&api_key={API_KEY}"

# response = requests.get(url)
# print(response.status_code)  should be 200 if successful
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "R4oF1wMa57H2MZm3r3u87kL28wEa6soL6BMABbXR" #My NASA API Key

def get_neo_feed(start_date, end_date):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={API_KEY}"
    r = requests.get(url)
    data = r.json()['near_earth_objects']
    
    all_neos = []
    for date in data:
        for neo in data[date]:
            entry = {
                'id': neo['id'],
                'name': neo['name'],
                'absolute_magnitude_h': neo['absolute_magnitude_h'],
                'estimated_diameter_min_km': neo['estimated_diameter']['kilometers']['estimated_diameter_min'],
                'estimated_diameter_max_km': neo['estimated_diameter']['kilometers']['estimated_diameter_max'],
                'is_potentially_hazardous': neo['is_potentially_hazardous_asteroid'],
                'close_approach_date': neo['close_approach_data'][0]['close_approach_date'],
                'relative_velocity_km_per_s': neo['close_approach_data'][0]['relative_velocity']['kilometers_per_second'],
                'miss_distance_km': neo['close_approach_data'][0]['miss_distance']['kilometers'],
                'orbiting_body': neo['close_approach_data'][0]['orbiting_body']
            }
            all_neos.append(entry)
    return pd.DataFrame(all_neos)

# 🔁 Loop over multiple weeks (example: 2 months = 8 weeks)
all_data = pd.DataFrame()
start = datetime(2025, 1, 1)
end = datetime(2025, 6, 1)

while start < end:
    week_end = start + timedelta(days=6)
    print(f"Fetching data from {start.date()} to {week_end.date()}...")
    df = get_neo_feed(start.date(), week_end.date())
    all_data = pd.concat([all_data, df], ignore_index=True)
    start += timedelta(days=7)

# Save full dataset
all_data.to_csv("data/nea_feed_large.csv", index=False)
print(f"✅ Saved {len(all_data)} entries to nea_feed_large.csv")
