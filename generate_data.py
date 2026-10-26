"""Step 1 - DATA COLLECTION: creates a realistic sample dataset (canteen_sales.csv).
Sunday closed, Saturday half-day, summer vacation dip, juice sells more in hot months.
A little dirty data (missing values, duplicates) is injected on purpose to practise cleaning."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
MENU = {  # item: (category, price, base popularity)
    "Idli": ("Breakfast", 20, 12), "Dosa": ("Breakfast", 40, 10),
    "Meals": ("Meals", 70, 16), "Fried Rice": ("Meals", 60, 9),
    "Sandwich": ("Snacks", 35, 7), "Samosa": ("Snacks", 12, 14),
    "Tea": ("Beverages", 10, 20), "Coffee": ("Beverages", 15, 9), "Juice": ("Beverages", 30, 4),
}
DOW = {0: 1.0, 1: 1.05, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.4, 6: 0.0}      # Mon..Sun
MONTH = {1: 1.0, 2: 1.05, 3: 1.0, 4: 0.9, 5: 0.45, 6: 0.5, 7: 1.0, 8: 1.05, 9: 1.1}

rows, tid = [], 1
for day in pd.date_range("2026-01-01", "2026-09-30"):
    n = int(rng.poisson(95 * DOW[day.dayofweek] * MONTH[day.month]))
    if n == 0:
        continue
    w = np.array([v[2] for v in MENU.values()], dtype=float)
    if day.month in (3, 4, 5, 6):                      # summer: juice demand triples
        w[list(MENU).index("Juice")] *= 3
    items = rng.choice(list(MENU), size=n, p=w / w.sum())
    for it in items:
        cat, price, _ = MENU[it]
        qty = int(min(rng.geometric(0.55 if it in ("Tea", "Samosa") else 0.75), 5))
        rows.append([f"T{tid:06d}", day.strftime("%Y-%m-%d"), it, cat, qty, price,
                     qty * price, rng.choice(["UPI", "Cash", "Card"], p=[0.55, 0.35, 0.10])])
        tid += 1

df = pd.DataFrame(rows, columns=["Transaction ID", "Date", "Food Item", "Category",
                                 "Quantity Sold", "Unit Price", "Total Amount", "Payment Method"])
# add the hour of purchase (separate RNG so earlier numbers stay the same)
rng2 = np.random.default_rng(7)
HRS = {"Breakfast": ([8, 9, 10, 11], [.35, .4, .2, .05]), "Meals": ([12, 13, 14, 15], [.3, .4, .25, .05]),
       "Snacks": (list(range(10, 18)), [.05, .08, .12, .1, .1, .2, .2, .15]),
       "Beverages": (list(range(8, 18)), [.1, .1, .08, .08, .1, .1, .1, .12, .12, .1])}
df["Hour"] = [int(rng2.choice(HRS[c][0], p=HRS[c][1])) for c in df["Category"]]
# inject dirty data
for col, frac in [("Payment Method", .01), ("Unit Price", .005), ("Total Amount", .005), ("Category", .003)]:
    df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan
df = pd.concat([df, df.sample(frac=0.01, random_state=2)]).sample(frac=1, random_state=3).sort_values("Date")
df.to_csv("canteen_sales.csv", index=False)
print(f"Saved canteen_sales.csv with {len(df)} rows")
