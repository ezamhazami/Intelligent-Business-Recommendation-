"""
label_data.py
-------------
Assigns business_type labels to the collected geospatial CSV.

5 classes for Shah Alam sections:
  1. Food & Beverage     — food culture + population base
  2. Retail & Commerce   — retail + mall + high footfall
  3. Community Services  — education + healthcare gap in populated areas
  4. Business & Trade    — corporate + automotive + financial
  5. Leisure & Lifestyle — entertainment + transport accessibility

Two complementary demand signals:
  - population      : real DOSM 2021 census — how many people live there

Scoring uses Z-score normalization computed dynamically from actual
CSV data — no hardcoded values, always reflects real collected data.

Usage:
    python label_data.py
"""

import csv
import math
import os

INPUT_PATH  = "data/shah_alam_geospatial2.csv"
OUTPUT_PATH = "data/shah_alam_labeled2.csv"

FEATURES = [
    "population",
    "food_beverage", "retail_outlet", "service_business",
    "entertainment", "educational_inst",
    "corporate_office", "financial_inst", "shopping_mall",
    "automotive", "healthcare", "transportation",
]



def compute_stats(rows: list) -> dict:
    """
    Compute mean and std dynamically from actual CSV data.
    No hardcoded values — auto-updates if data changes.
    """
    stats = {}
    for feat in FEATURES:
        vals = [int(r.get(feat, 0)) for r in rows]
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        stats[feat] = (mean, max(std, 1.0))
    return stats


def zscore(feature: str, value: int, stats: dict) -> float:
    """
    Normalize raw value to z-score.
    Positive = above average (strong presence).
    Negative = below average (gap = opportunity for gap-based classes).
    """
    mean, std = stats[feature]
    return (value - mean) / std



def assign_label(row: dict, stats: dict) -> str:
    pop  = int(row.get("population",      0))
    fb   = int(row.get("food_beverage",   0))
    ret  = int(row.get("retail_outlet",   0))
    svc  = int(row.get("service_business",0))
    ent  = int(row.get("entertainment",   0))
    edu  = int(row.get("educational_inst",0))
    corp = int(row.get("corporate_office",0))
    fin  = int(row.get("financial_inst",  0))
    mall = int(row.get("shopping_mall",   0))
    auto = int(row.get("automotive",      0))
    hc   = int(row.get("healthcare",      0))
    tran = int(row.get("transportation",  0))

    z_pop  = zscore("population",      pop,  stats)
    z_fb   = zscore("food_beverage",   fb,   stats)
    z_ret  = zscore("retail_outlet",   ret,  stats)
    z_svc  = zscore("service_business",svc,  stats)
    z_ent  = zscore("entertainment",   ent,  stats)
    z_edu  = zscore("educational_inst",edu,  stats)
    z_corp = zscore("corporate_office",corp, stats)
    z_fin  = zscore("financial_inst",  fin,  stats)
    z_mall = zscore("shopping_mall",   mall, stats)
    z_auto = zscore("automotive",      auto, stats)
    z_hc   = zscore("healthcare",      hc,   stats)
    z_tran = zscore("transportation",  tran, stats)

    scores = {

        #1. Food & Beverage 
        # population = customer base size
        # food_beverage = proven food culture 
        # corporate = weekday lunch crowd
        # penalise automotive dominance — industrial ≠ food hub
        "Food & Beverage": (
            z_pop  * 1.5 +
            z_fb   * 2.0 +
            z_corp * 0.8
            - max(0, z_auto) * 0.5
        ),

        # 2. Retail & Commerce 
        # mall is rarest feature = strongest signal when present
        # existing retail = shopping destination (agglomeration)
        "Retail & Commerce": (
            z_mall * 3.0 +
            z_ret  * 2.0 +
            z_pop  * 1.0 +
            z_corp * 0.8
        ),

        # 3. Community Services 
        # GAP-BASED:
        # low existing edu/healthcare = underserved = opportunity
        # heavy penalty if edu/hc already saturated (> 1 std above avg)
        "Community Services": (
            z_pop  * 2.0 +
            z_edu  * 0.5 +
            z_hc   * 0.5
            - max(0, z_edu - 1.0) * 2.0
            - max(0, z_hc  - 1.0) * 2.0
        ),

        #  4. Business & Trade 
        # corporate density = clearest business district signal
        # automotive clusters naturally in Shah Alam (Section 7 effect)
        # financial (rare) = strong commercial activity confirmation
        "Business & Trade": (
            z_corp * 2.5 +
            z_auto * 1.5 +
            z_fin  * 2.0 +
            z_svc  * 0.5
        ),

        # 5. Leisure & Lifestyle 
        # entertainment is rarest feature = highest discriminating power
        # transport = people can reach the leisure area
        # food nearby = lifestyle area confirmation
        "Leisure & Lifestyle": (
            z_ent  * 3.5 +
            z_tran * 1.5 +
            z_fb   * 0.5
        ),
    }

    return max(scores, key=scores.get)



def label_csv():
    if not os.path.exists(INPUT_PATH):
        print(f"❌  {INPUT_PATH} not found. Run collect_data.py first.")
        return

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("❌  CSV is empty.")
        return

    stats = compute_stats(rows)
    print("📊 Feature stats computed from data:")
    print(f"   {'Feature':<20} {'Mean':>10}  {'Std':>10}")
    print(f"   {'-'*46}")
    for feat in FEATURES:
        m, s = stats[feat]
        print(f"   {feat:<20} {m:>10.1f}  {s:>10.1f}")
    print()

    for row in rows:
        row["business_type"] = assign_label(row, stats)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅  Labelled CSV saved to {OUTPUT_PATH}")
    _print_distribution(rows)


def _print_distribution(rows: list):
    dist = {}
    for r in rows:
        dist[r["business_type"]] = dist.get(r["business_type"], 0) + 1

    total = len(rows)
    print(f"\n   Label distribution ({total} sections):")
    print(f"   {'Business Type':<25} {'Count':>5}  {'%':>5}  Chart")
    print(f"   {'-'*65}")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        bar = "█" * v
        print(f"   {k:<25} {v:>5}  {pct:>4.1f}%  {bar}")

    print()
    max_count = max(dist.values())
    min_count = min(dist.values())

    if max_count / total > 0.40:
        dominant = max(dist, key=dist.get)
        print(f"   ⚠️  '{dominant}' dominates at {max_count/total*100:.1f}% — weights may need rebalancing")
    for k, v in dist.items():
        if v < 5:
            print(f"   ⚠️  '{k}' only has {v} samples — too few for reliable ML training")
    if max_count / max(min_count, 1) <= 2.5:
        print(f"   ✅  Distribution looks balanced — good for ML training")


if __name__ == "__main__":
    label_csv()