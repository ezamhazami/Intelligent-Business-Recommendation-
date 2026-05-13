"""
collect_data.py
---------------
Extracts geospatial attributes from Google Maps Places API
for 56 sections in Shah Alam, Selangor, Malaysia.

Features:
  - Real 2021 DOSM population per section
  - No pagination (first page only, max 20 per type)
  - 3 types per category (max 60 per category)
  - accomodation removed (hotel hits cap everywhere, unreliable)
  - RADIUS = 1000m for realistic business catchment area
"""

import os
import time
import csv
import math
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
NEARBY_URL     = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
RADIUS         = 1000
OUTPUT_PATH    = "data/shah_alam_geospatial2.csv"


SECTIONS = {
    "Section 1":   (3.0671, 101.5017),
    "Section 2":   (3.0718, 101.5084),
    "Section 3":   (3.0742, 101.5097),
    "Section 4":   (3.0059, 101.5511),
    "Section 5":   (3.0831, 101.5163),
    "Section 6":   (3.0821, 101.5089),
    "Section 7":   (3.0732, 101.4924),
    "Section 8":   (3.0900, 101.5088),
    "Section 9":   (3.0832, 101.5283),
    "Section 10":  (3.0803, 101.5240),
    "Section 11":  (3.0744, 101.5289),
    "Section 12":  (3.0700, 101.5357),
    "Section 13":  (3.0818, 101.5394),
    "Section 14":  (3.0714, 101.5243),
    "Section 15":  (3.0870, 101.5350),
    "Section 16":  (3.0568, 101.5020),
    "Section 17":  (3.0463, 101.5058),
    "Section 18":  (3.0481, 101.5198),
    "Section 19":  (3.0484, 101.5341),
    "Section 20":  (3.0552, 101.5391),
    "Section 21":  (3.0563, 101.5495),
    "Section 22":  (3.0504, 101.5471),
    "Section 23":  (3.0465, 101.5288),
    "Section 24":  (3.0385, 101.5185),
    "Section 25":  (3.0225, 101.5385),
    "Section 26":  (3.0266, 101.5622),
    "Section 27":  (3.0235, 101.5713),
    "Section 28":  (3.0051, 101.5602),
    "Section 29":  (3.1731, 101.5138),
    "Section 30":  (2.9790, 101.5184),
    "Section 31":  (2.9864, 101.5395),
    "Section 32":  (3.0062, 101.5181),
    "Section 33":  (3.0214, 101.5424),
    "Section 34":  (3.0114, 101.4997),
    "Section 35":  (3.0318, 101.5124),
    "Section 36":  (3.0323, 101.4899),
    "Section U1":  (3.0819, 101.5604),
    "Section U2":  (3.0963, 101.5568),
    "Section U3":  (3.1198, 101.5635),
    "Section U4":  (3.1564, 101.5598),
    "Section U5":  (3.1605, 101.5491),
    "Section U6":  (3.1385, 101.5307),
    "Section U7":  (3.1276, 101.5449),
    "Section U8":  (3.1108, 101.5469),
    "Section U9":  (3.1300, 101.5177),
    "Section U10": (3.1367, 101.4959),
    "Section U11": (3.0952, 101.5072),
    "Section U12": (3.0845, 101.4854),
    "Section U13": (3.0943, 101.4443),
    "Section U14": (3.0878, 101.4884),
    "Section U15": (3.1776, 101.4621),
    "Section U16": (3.1608, 101.5122),
    "Section U17": (3.2203, 101.4991),
    "Section U18": (3.2110, 101.5344),
    "Section U19": (3.2022, 101.5642),
    "Section U20": (3.2092, 101.5425),
}

# 2021 DOSM Population per section 
# Source: Department of Statistics Malaysia, Population Census 2021
# Section 4&5 combined in census — split equally
# Section 11&12 combined in census — split equally
SECTION_POPULATION = {
    "Section 1":   1664,
    "Section 2":   1991,
    "Section 3":   1332,
    "Section 4":   851,
    "Section 5":   851,
    "Section 6":   3033,
    "Section 7":   44646,
    "Section 8":   7440,
    "Section 9":   6403,
    "Section 10":  1072,
    "Section 11":  1793,
    "Section 12":  1793,
    "Section 13":  18187,
    "Section 14":  4198,
    "Section 15":  7453,
    "Section 16":  19795,
    "Section 17":  25008,
    "Section 18":  13193,
    "Section 19":  15583,
    "Section 20":  11403,
    "Section 21":  12832,
    "Section 22":  3273,
    "Section 23":  3407,
    "Section 24":  18029,
    "Section 25":  38812,
    "Section 26":  13337,
    "Section 27":  21377,
    "Section 28":  12180,
    "Section 29":  3352,
    "Section 30":  20885,
    "Section 31":  29228,
    "Section 32":  43792,
    "Section 33":  10040,
    "Section 34":  3279,
    "Section 35":  9799,
    "Section 36":  12653,
    "Section U1":  20375,
    "Section U2":  14535,
    "Section U3":  14674,
    "Section U4":  4312,
    "Section U5":  60830,
    "Section U6":  15058,
    "Section U7":  7920,
    "Section U8":  12618,
    "Section U9":  12555,
    "Section U10": 10346,
    "Section U11": 4047,
    "Section U12": 6151,
    "Section U13": 86548,
    "Section U14": 1003,
    "Section U15": 5396,
    "Section U16": 28067,
    "Section U17": 17410,
    "Section U18": 7593,
    "Section U19": 20807,
    "Section U20": 24250,
}


FEATURE_TYPES = {
    "food_beverage":    ["restaurant", "cafe", "bakery"],
    "retail_outlet":    ["clothing_store", "supermarket", "electronics_store"],
    "service_business": ["beauty_salon", "laundry", "hair_care"],
    "entertainment":    ["gym", "movie_theater", "night_club"],
    "educational_inst": ["school", "university", "library"],
    "corporate_office": ["real_estate_agency", "lawyer", "insurance_agency"],
    "financial_inst":   ["bank", "atm"],
    "shopping_mall":    ["shopping_mall", "department_store"],
    "automotive":       ["car_repair", "gas_station", "car_wash"],
    "healthcare":       ["hospital", "doctor", "pharmacy"],
    "transportation":   ["bus_station", "train_station", "transit_station"],
}


def count_places(lat: float, lng: float, place_types: list) -> int:
    """
    Return total count of places within radius for given types.
    First page only (max 20 per type) — no pagination.
    Prevents inflated counts and preserves variation between sections.
    """
    total = 0
    for ptype in place_types:
        params = {
            "location": f"{lat},{lng}",
            "radius":   RADIUS,
            "type":     ptype,
            "key":      GOOGLE_API_KEY,
        }
        try:
            resp = requests.get(NEARBY_URL, params=params, timeout=10)
            data = resp.json()
            total += len(data.get("results", []))

        except Exception as e:
            print(f"  ⚠  Error for type '{ptype}': {e}")

        time.sleep(0.2)

    return total


def compute_diversity_index(counts: dict) -> float:
    """Shannon diversity index across amenity categories."""
    values = [v for v in counts.values() if v > 0]
    total  = sum(values)
    if total == 0:
        return 0.0
    proportions = [v / total for v in values]
    return round(-sum(p * math.log(p) for p in proportions), 4)



def collect():
    os.makedirs("data", exist_ok=True)

    fieldnames = [
        "section", "lat", "lng", "population",
        "food_beverage", "retail_outlet", "service_business",
        "entertainment", "educational_inst",
        "corporate_office", "financial_inst", "shopping_mall",
        "automotive", "healthcare", "transportation",
        "amenity_diversity_index", "business_type",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for section, (lat, lng) in SECTIONS.items():
            print(f"📍 Collecting {section} ({lat}, {lng})..")
            row = {
                "section":    section,
                "lat":        lat,
                "lng":        lng,
                "population": SECTION_POPULATION.get(section, 0),
            }
            counts = {}

            for feature, types in FEATURE_TYPES.items():
                count           = count_places(lat, lng, types)
                row[feature]    = count
                counts[feature] = count
                print(f"   {feature:<20}: {count}")

            row["amenity_diversity_index"] = compute_diversity_index(counts)
            row["business_type"]           = ""
            writer.writerow(row)
            f.flush()
            print(f"   👥 population          : {row['population']}")
            print(f"   ✅ {section} done  (diversity={row['amenity_diversity_index']})\n")

    print(f"\n✅ Data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("❌  GOOGLE_MAPS_API_KEY not set in .env")
    else:
        collect()