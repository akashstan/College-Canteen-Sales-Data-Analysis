"""College Canteen Sales Dashboard.

Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import canteen_analysis as ca

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CSV_PATH = Path(__file__).parent / "canteen_sales.csv"
COLORS = {
    "primary": "#9D2805",
    "dark": "#010318",
    "teal": "#0A8886",
    "gold": "#856705",
    "grey": "#0542AA",
}
PALETTE = [COLORS[c] for c in ("primary", "teal", "gold", "dark", "grey")]

st.set_page_config(page_title="Canteen Sales Dashboard", page_icon="🍽️", layout="wide")

CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #E4572E 0%, #F18F01 100%);
    padding: 28px 34px; border-radius: 16px; color: white; margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(228, 87, 46, .35);
}
.hero h1 { margin: 0; font-size: 2.1rem; color: white; }
.hero p  { margin: 6px 0 0 0; opacity: .92; font-size: 1.02rem; }
[data-testid="stMetric"] {
    background: white; border-left: 6px solid #E4572E; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 2px 10px rgba(0, 0, 0, .08);
}
[data-testid="stMetricLabel"] { color: #8D99AE; font-weight: 600; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 10px 16px; }
.stTabs [aria-selected="true"] { color: #E4572E; }
h2, h3 { color: #2B2D42; }
footer { visibility: hidden; }
.foot { text-align: center; color: #8D99AE; font-size: .85rem; margin-top: 30px; }
</style>
"""

sns.set_theme(
    style="whitegrid",
    palette=PALETTE,
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "grid.alpha": 0.4,
        "figure.facecolor": "white",
    },
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data
def load_default(path):
    """Read and clean the bundled CSV (cached so it only runs once)."""
    return ca.clean_data(pd.read_csv(path))


def new_fig(width=10, height=4):
    """Create a matplotlib figure with a consistent size."""
    return plt.subplots(figsize=(width, height))


def show(fig, title=None, ax=None):
    """Render a matplotlib figure in Streamlit and release its memory."""
    if title and ax is not None:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
def sidebar_filters():
    """Draw the sidebar and return the filtered data plus the filter values."""
    st.sidebar.title("🍽️ Filters")
    upload = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")

    if upload is None and not CSV_PATH.exists():
        st.error("canteen_sales.csv not found. Run `python generate_data.py` first.")
        st.stop()
    df_all, log = ca.clean_data(pd.read_csv(upload)) if upload else load_default(CSV_PATH)

    first, last = df_all["Date"].min().date(), df_all["Date"].max().date()
    dates = st.sidebar.date_input("Date range", (first, last), min_value=first, max_value=last)
    all_items = sorted(df_all["Food Item"].unique())
    all_cats = sorted(df_all["Category"].unique())
    items = st.sidebar.multiselect("Food items", all_items, default=all_items)
    cats = st.sidebar.multiselect("Categories", all_cats, default=all_cats)

    if len(dates) != 2:
        st.info("Select both a start and an end date.")
        st.stop()

    mask = (
        df_all["Date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
        & df_all["Food Item"].isin(items)
        & df_all["Category"].isin(cats)
    )
    if not mask.any():
        st.warning("No data for the selected filters.")
        st.stop()
    return df_all[mask], df_all, log, dates, items, cats


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #
def render_header():
    st.markdown(
        """<div class="hero"><h1>🍽️ College Canteen Sales Analysis</h1>
        <p>Data Science &amp; Engineering project &mdash; explore sales, trends,
        peak times and recommendations</p></div>""",
        unsafe_allow_html=True,
    )


def render_kpis(df, df_all, dates, items, cats):
    """Top row of KPI cards with % change versus the previous period."""
    now = ca.kpis(df)
    before = ca.previous_kpis(df_all, dates, items, cats)

    def change(key):
        if before is None or not before[key]:
            return None
        return f"{(now[key] - before[key]) / before[key] * 100:+.1f}% vs previous period"

    cols = st.columns(5)
    cols[0].metric("Total Revenue", f"₹{now['revenue']:,.0f}", change("revenue"))
    cols[1].metric("Transactions", f"{now['transactions']:,}", change("transactions"))
    cols[2].metric("Items Sold", f"{now['items']:,}", change("items"))
    cols[3].metric("Avg Order Value", f"₹{now['aov']:.2f}", change("aov"))
    cols[4].metric("Best-Selling Item", now["best"])


def tab_trends(df):
    daily = ca.daily_sales(df)
    daily["avg7"] = daily["Revenue"].rolling(7, min_periods=1).mean()
    unusual = ca.anomalies(df)

    fig, ax = new_fig(11, 4)
    ax.plot(daily["Date"], daily["Revenue"], color=COLORS["grey"], alpha=0.4, label="Daily revenue")
    ax.plot(daily["Date"], daily["avg7"], color=COLORS["primary"], lw=2.5, label="7-day average")
    ax.scatter(unusual["Date"], unusual["Revenue"], color="red", zorder=5, label="Unusual day")
    ax.set_ylabel("Revenue (₹)")
    ax.legend(frameon=False)
    show(fig, "Daily Sales Trend", ax)
    st.caption(f"{len(unusual)} unusual days flagged (red dots). Details are in the Peak Days tab.")

    fig, ax = new_fig(11, 4)
    sns.barplot(data=ca.monthly_sales(df), x="Month", y="Revenue", color=COLORS["primary"], ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.0f", fontsize=8)
    show(fig, "Monthly Revenue", ax)

    st.line_chart(df.groupby("Week")["Total Amount"].sum(), height=220, color=COLORS["teal"])
    st.caption("Weekly revenue")


def tab_items(df):
    ranking = ca.item_ranking(df)
    top_n = st.slider("Show top N items", 3, len(ranking), len(ranking)) if len(ranking) > 3 else len(ranking)

    left, right = st.columns(2)
    with left:
        fig, ax = new_fig(6, 4)
        sns.barplot(data=ranking.head(top_n), y="Food Item", x="Revenue", color=COLORS["primary"], ax=ax)
        show(fig, "Revenue by Item", ax)
    with right:
        by_qty = ranking.sort_values("Quantity", ascending=False).head(top_n)
        fig, ax = new_fig(6, 4)
        sns.barplot(data=by_qty, y="Food Item", x="Quantity", color=COLORS["teal"], ax=ax)
        show(fig, "Quantity Sold by Item", ax)

    best = ranking.loc[ranking["Quantity"].idxmax(), "Food Item"]
    worst = ranking.loc[ranking["Quantity"].idxmin(), "Food Item"]
    st.success(f"Best seller (quantity): **{best}**   |   Least sold: **{worst}**")
    st.dataframe(ranking, use_container_width=True, hide_index=True)

    st.subheader("Item drill-down")
    pick = st.selectbox("Choose an item", ranking["Food Item"])
    st.bar_chart(df[df["Food Item"] == pick].groupby("Month")["Quantity Sold"].sum(), color=COLORS["gold"])
    st.caption(f"Monthly quantity sold: {pick}")


def tab_categories(df):
    left, right = st.columns(2)
    with left:
        fig, ax = new_fig(6, 4)
        sns.barplot(data=ca.category_sales(df), x="Category", y="Revenue", ax=ax)
        show(fig, "Category-wise Revenue", ax)
    with right:
        payments = df["Payment Method"].value_counts()
        fig, ax = new_fig(6, 4)
        ax.pie(payments, labels=payments.index, autopct="%1.0f%%", startangle=90,
               colors=PALETTE, wedgeprops={"width": 0.45, "edgecolor": "white"})
        show(fig, "Payment Methods", ax)

    monthly = df.pivot_table(index="Month", columns="Category", values="Total Amount", aggfunc="sum")
    fig, ax = new_fig(11, 4)
    monthly.plot(kind="bar", stacked=True, ax=ax, color=PALETTE)
    ax.set_ylabel("Revenue (₹)")
    ax.legend(frameon=False, ncol=4)
    plt.xticks(rotation=0)
    show(fig, "Monthly Revenue by Category", ax)


def tab_peak(df):
    weekday = ca.weekday_avg(df)
    bar_colors = [COLORS["primary"] if v == weekday["Avg Revenue"].max() else COLORS["grey"]
                  for v in weekday["Avg Revenue"]]
    fig, ax = new_fig(9, 4)
    ax.bar(weekday["Day Name"], weekday["Avg Revenue"], color=bar_colors)
    plt.xticks(rotation=30)
    show(fig, "Average Revenue by Weekday (peak in orange)", ax)
    st.info(f"Peak day: **{weekday.loc[weekday['Avg Revenue'].idxmax(), 'Day Name']}**")

    fig, ax = new_fig(11, 4)
    sns.heatmap(ca.weekday_month_pivot(df), annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    show(fig, "Revenue Heatmap: Weekday x Month", ax)

    if "Hour" in df:
        st.subheader("Peak hours")
        by_hour = df.groupby("Hour")["Total Amount"].sum()
        rush = int(by_hour.idxmax())
        st.info(f"Rush hour: **{rush}:00 - {rush + 1}:00**")
        fig, ax = new_fig(11, 4)
        sns.heatmap(ca.hour_pivot(df), cmap="YlGnBu", ax=ax)
        show(fig, "Revenue Heatmap: Weekday x Hour", ax)

    st.subheader("Unusual days (anomaly detection)")
    st.caption("Days whose revenue is far from the normal for that weekday in the surrounding weeks.")
    st.dataframe(ca.anomalies(df), hide_index=True)

    st.subheader("Statistical analysis")
    summary = ca.stats_summary(df)
    table = pd.DataFrame({
        "Measure": list(summary),
        "Value": [f"{v:,.4g}" if isinstance(v, float) else str(v) for v in summary.values()],
    })
    st.dataframe(table, hide_index=True)


def tab_forecast(df):
    st.write("Models predict **daily revenue** from weekday, month and time trend. "
             "The last 20% of dates is the test set (time-based split, no shuffling).")
    if df["Date"].nunique() < 40:
        st.warning("Select a longer date range (40+ days) to train the forecast model.")
        return

    result = ca.forecast(df)
    st.dataframe(result["metrics"].round(2), hide_index=True)
    st.success(f"Best model: {result['best']}")

    test = result["test"]
    fig, ax = new_fig(11, 4)
    ax.plot(test["Date"], test["Actual"], color=COLORS["dark"], label="Actual")
    ax.plot(test["Date"], test["Predicted"], color=COLORS["primary"], label="Predicted")
    ax.legend(frameon=False)
    show(fig, "Test Set: Actual vs Predicted", ax)

    future = result["future"]
    st.subheader("Next 14 open days")
    fig, ax = new_fig(11, 4)
    ax.plot(future["Date"], future["Forecast Revenue"], marker="o", color=COLORS["primary"], label="Forecast")
    ax.fill_between(future["Date"], future["Lower"], future["Upper"], color=COLORS["teal"],
                    alpha=0.25, label="~95% range")
    ax.legend(frameon=False)
    show(fig, "Forecast with confidence range", ax)
    st.dataframe(future, hide_index=True)


def tab_insights(df, log):
    st.subheader("Business recommendations")
    for tip in ca.recommendations(df):
        st.markdown(f"- {tip}")

    with st.expander("Data cleaning log"):
        st.json({key: int(value) for key, value in log.items()})
    with st.expander("View filtered data (first 500 rows)"):
        st.dataframe(df.drop(columns=["Week"]).head(500), hide_index=True)

    left, right = st.columns(2)
    left.download_button("⬇️ Download filtered data (CSV)", df.drop(columns=["Week"]).to_csv(index=False),
                         "filtered_sales.csv", "text/csv")
    right.download_button("⬇️ Download summary report (TXT)", ca.build_report(df, log), "canteen_report.txt")


def tab_about(log):
    st.header("About this project")
    st.markdown(
        """
**Aim:** analyse college canteen sales to find trends, popular items and peak times, and give business recommendations.

**Workflow:** Data collection → Cleaning → EDA → Statistics (ANOVA) → Visualisation → Forecasting (Random Forest) → Recommendations

**Extra features:** % change vs previous period, unusual-day detection, peak-hour heatmap, item drill-down, forecast range.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, Streamlit.
"""
    )
    st.write("Cleaning log:", {key: int(value) for key, value in log.items()})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    df, df_all, log, dates, items, cats = sidebar_filters()
    render_header()
    render_kpis(df, df_all, dates, items, cats)
    st.write("")

    tabs = st.tabs(["📈 Trends", "🍛 Food Items", "🗂️ Categories & Payments", "🔥 Peak Days",
                    "🤖 Forecast", "💡 Insights & Reports", "ℹ️ Project Info"])
    with tabs[0]:
        tab_trends(df)
    with tabs[1]:
        tab_items(df)
    with tabs[2]:
        tab_categories(df)
    with tabs[3]:
        tab_peak(df)
    with tabs[4]:
        tab_forecast(df)
    with tabs[5]:
        tab_insights(df, log)
    with tabs[6]:
        tab_about(log)

    st.markdown('<div class="foot">College Canteen Sales Analysis · DSE Project · Built with Streamlit</div>',
                unsafe_allow_html=True)


main()
