"""Core data-science functions used by the dashboard and the EDA script."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------- Step 2: preprocessing & cleaning ----------
def clean_data(raw):
    """Returns (clean_df, log) where log explains every cleaning step."""
    df, log = raw.copy(), {"Raw rows": len(raw)}
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Quantity Sold", "Unit Price", "Total Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Hour" in df: df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    n = len(df)
    df = df.drop_duplicates()
    log["Exact duplicate rows removed"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates("Transaction ID")
    log["Repeated Transaction IDs removed"] = n - len(df)
    n = len(df)
    df = df.dropna(subset=["Date", "Food Item", "Quantity Sold"])
    log["Rows dropped (missing date/item/quantity)"] = n - len(df)
    df["Food Item"] = df["Food Item"].str.strip().str.title()
    log["Missing Unit Price filled (item median)"] = int(df["Unit Price"].isna().sum())
    df["Unit Price"] = df["Unit Price"].fillna(df["Food Item"].map(df.groupby("Food Item")["Unit Price"].median()))
    log["Missing Category filled (from item)"] = int(df["Category"].isna().sum())
    cat_map = df.dropna(subset=["Category"]).groupby("Food Item")["Category"].agg(lambda s: s.mode()[0])
    df["Category"] = df["Category"].fillna(df["Food Item"].map(cat_map))
    log["Missing Payment Method set to 'Unknown'"] = int(df["Payment Method"].isna().sum())
    df["Payment Method"] = df["Payment Method"].fillna("Unknown")
    if "Hour" in df: df["Hour"] = df["Hour"].fillna(df["Hour"].median()).astype(int)
    m = df["Total Amount"].isna()
    log["Missing Total Amount recomputed (qty x price)"] = int(m.sum())
    df.loc[m, "Total Amount"] = df["Quantity Sold"] * df["Unit Price"]
    n = len(df)
    df = df[(df["Quantity Sold"] > 0) & (df["Total Amount"] > 0)]
    log["Invalid rows (qty/amount <= 0) removed"] = n - len(df)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Day Name"] = df["Date"].dt.day_name()
    df["Week"] = df["Date"].dt.to_period("W").dt.start_time
    log["Clean rows"] = len(df)
    return df.reset_index(drop=True), log


# ---------- Step 3/4: EDA ----------
def kpis(df):
    rev = df["Total Amount"].sum()
    return {"revenue": rev, "transactions": len(df), "items": int(df["Quantity Sold"].sum()),
            "aov": rev / max(len(df), 1), "best": df.groupby("Food Item")["Quantity Sold"].sum().idxmax()}

def previous_kpis(df_all, dates, items, cats):
    """KPIs of the period just before the selected one (same length) for % change."""
    s, e = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    n = (e - s).days + 1
    prev = df_all[df_all["Date"].between(s - pd.Timedelta(days=n), s - pd.Timedelta(days=1))
                  & df_all["Food Item"].isin(items) & df_all["Category"].isin(cats)]
    return kpis(prev) if len(prev) else None

def anomalies(df, z=2.5):
    """Unusual days: revenue far from the typical value for the same weekday (+-2 weeks)."""
    d = daily_sales(df)
    d["Day Name"] = d["Date"].dt.day_name()
    d["Expected"] = d.groupby("Day Name")["Revenue"].transform(lambda s: s.rolling(5, center=True, min_periods=2).median())
    r = d["Revenue"] - d["Expected"]
    d["Z-score"] = r / r.std()
    out = d[d["Z-score"].abs() > z][["Date", "Day Name", "Revenue", "Expected", "Z-score"]].copy()
    out[["Revenue", "Expected", "Z-score"]] = out[["Revenue", "Expected", "Z-score"]].round(1)
    return out

def hour_pivot(df):
    return df.pivot_table(index="Day Name", columns="Hour", values="Total Amount", aggfunc="sum").reindex(DAY_ORDER).dropna(how="all")

def item_ranking(df):
    return (df.groupby("Food Item").agg(Quantity=("Quantity Sold", "sum"), Revenue=("Total Amount", "sum"),
            Transactions=("Transaction ID", "count")).sort_values("Revenue", ascending=False).reset_index())

def daily_sales(df):
    return df.groupby("Date").agg(Revenue=("Total Amount", "sum"), Transactions=("Transaction ID", "count"),
                                  Items=("Quantity Sold", "sum")).reset_index()

def monthly_sales(df):
    return df.groupby("Month")["Total Amount"].sum().reset_index(name="Revenue")

def category_sales(df):
    return df.groupby("Category")["Total Amount"].sum().sort_values(ascending=False).reset_index(name="Revenue")

def weekday_avg(df):
    d = daily_sales(df)
    d["Day Name"] = d["Date"].dt.day_name()
    return d.groupby("Day Name")["Revenue"].mean().reindex(DAY_ORDER).dropna().reset_index(name="Avg Revenue")

def weekday_month_pivot(df):
    return df.pivot_table(index="Day Name", columns="Month", values="Total Amount", aggfunc="sum").reindex(DAY_ORDER).dropna(how="all")

# ---------- Step 4: statistical analysis ----------
def stats_summary(df):
    d = daily_sales(df)
    r = d["Revenue"]
    d["dow"] = d["Date"].dt.dayofweek
    groups = [g["Revenue"].values for _, g in d.groupby("dow") if len(g) > 1]
    f, p = stats.f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)
    return {"Mean daily revenue": r.mean(), "Median daily revenue": r.median(), "Std deviation": r.std(),
            "Coefficient of variation (%)": r.std() / r.mean() * 100,
            "ANOVA F (weekday effect)": f, "ANOVA p-value": p,
            "Weekday effect significant (p<0.05)": bool(p < 0.05)}

# ---------- Optional ML: forecasting daily revenue ----------
def _features(dates, start):
    return pd.DataFrame({"t": (dates - start).days, "month": dates.month, "dow": dates.dayofweek})

def forecast(df, horizon=14):
    d = daily_sales(df).sort_values("Date")
    start = d["Date"].min()
    X = _features(pd.DatetimeIndex(d["Date"]), start)
    y = d["Revenue"].values
    cut = int(len(d) * 0.8)                                  # time-based split: NO shuffling
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    dum = lambda A: pd.get_dummies(A, columns=["dow"]).reindex(columns=cols, fill_value=0)
    cols = pd.get_dummies(X, columns=["dow"]).columns
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)}
    base = pd.Series(ytr).groupby(Xtr["dow"].values).mean()
    preds = {"Baseline (weekday average)": Xte["dow"].map(base).values}
    for n, m in models.items():
        preds[n] = m.fit(dum(Xtr), ytr).predict(dum(Xte))
    rows = [{"Model": n, "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5,
             "R2": r2_score(yte, p), "MAPE %": np.mean(np.abs((yte - p) / yte)) * 100} for n, p in preds.items()]
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = metrics.loc[0, "Model"]
    test = pd.DataFrame({"Date": d["Date"].values[cut:], "Actual": yte, "Predicted": preds[best]})
    fut = pd.date_range(d["Date"].max() + pd.Timedelta(days=1), periods=horizon * 2)
    fut = fut[fut.dayofweek != 6][:horizon]                  # canteen closed on Sundays
    Xf = _features(fut, start)
    if best in models:
        final = models[best].fit(dum(X), y)
        fy = final.predict(dum(Xf))
    else:
        fy = Xf["dow"].map(pd.Series(y).groupby(X["dow"].values).mean()).values
    band = 1.96 * metrics.loc[0, "RMSE"]                      # approx. 95% range from test error
    return {"metrics": metrics, "best": best, "test": test,
            "future": pd.DataFrame({"Date": fut, "Forecast Revenue": fy.round(0),
                                    "Lower": (fy - band).clip(0).round(0), "Upper": (fy + band).round(0)})}


# ---------- Step 6/7: insights & recommendations ----------
def recommendations(df):
    items, wk, cat, mon = item_ranking(df), weekday_avg(df), category_sales(df), monthly_sales(df)
    q = df.groupby("Food Item")["Quantity Sold"].sum().sort_values()
    pay = df["Payment Method"].value_counts(normalize=True) * 100
    top_share = items["Revenue"].head(3).sum() / items["Revenue"].sum() * 100
    out = [f"Best seller by quantity is {q.index[-1]} ({int(q.iloc[-1]):,} units); top revenue item is {items.loc[0,'Food Item']}. Keep these always in stock and prepare more at peak times.",
           f"The top 3 items bring {top_share:.0f}% of revenue. Protect their quality and supply.",
           f"{q.index[0]} is the least-sold item ({int(q.iloc[0]):,} units). Try combo offers, discounts, or a menu review.",
           f"Busiest weekday: {wk.loc[wk['Avg Revenue'].idxmax(),'Day Name']} (avg Rs {wk['Avg Revenue'].max():,.0f}); quietest: {wk.loc[wk['Avg Revenue'].idxmin(),'Day Name']} (avg Rs {wk['Avg Revenue'].min():,.0f}). Schedule staff and raw material accordingly.",
           f"Highest revenue month: {mon.loc[mon['Revenue'].idxmax(),'Month']}; lowest: {mon.loc[mon['Revenue'].idxmin(),'Month']}. Plan purchases and reduce wastage in low months (vacations/exams).",
           f"'{cat.loc[0,'Category']}' is the top category by revenue; '{cat.loc[len(cat)-1,'Category']}' is the lowest - promote it with bundle deals.",
           f"{pay.get('UPI',0):.0f}% of payments are UPI. Keep QR payment reliable to shorten queues."]
    if "Hour" in df:
        h = df.groupby("Hour")["Total Amount"].sum()
        out.append(f"Rush hour is {int(h.idxmax())}:00-{int(h.idxmax())+1}:00. Keep extra counters/staff and ready-made food for this hour.")
    return out

def build_report(df, log):
    k, s = kpis(df), stats_summary(df)
    lines = ["COLLEGE CANTEEN SALES - SUMMARY REPORT", "=" * 40,
             f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}",
             f"Total revenue: Rs {k['revenue']:,.0f}", f"Total transactions: {k['transactions']:,}",
             f"Total items sold: {k['items']:,}", f"Average order value: Rs {k['aov']:.2f}",
             f"Best-selling item: {k['best']}", "", "STATISTICS"]
    lines += [f"- {a}: {b:,.3f}" if isinstance(b, float) else f"- {a}: {b}" for a, b in s.items()]
    lines += ["", "TOP ITEMS BY REVENUE", item_ranking(df).head(5).to_string(index=False), "", "RECOMMENDATIONS"]
    lines += [f"{i}. {r}" for i, r in enumerate(recommendations(df), 1)]
    lines += ["", "DATA CLEANING LOG"] + [f"- {a}: {b}" for a, b in log.items()]
    return "\n".join(lines)
