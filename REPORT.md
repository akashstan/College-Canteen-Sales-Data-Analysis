# College Canteen Sales Data Analysis - Project Report

## 1. Introduction
College canteens record many small sales every day, but the data is rarely used. This project applies the data science workflow to canteen sales to find trends, popular items, peak periods and practical recommendations, and presents them in an interactive Streamlit dashboard.

## 2. Objectives
1. Analyse daily, weekly and monthly sales.
2. Identify best- and least-selling items.
3. Compute total revenue, average order value and quantity sold.
4. Find peak days and busy periods.
5. Visualise trends with charts.
6. Give data-driven recommendations (and optionally forecast sales).

## 3. Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy (ANOVA), scikit-learn (forecasting), Streamlit, CSV.

## 4. Dataset
`canteen_sales.csv` - synthetic data, 1 Jan to 30 Sep 2026, 18,112 raw rows, 9 items in 4 categories (Breakfast, Meals, Snacks, Beverages). Columns: Transaction ID, Date, Food Item, Category, Quantity Sold, Unit Price, Total Amount, Payment Method. The canteen is closed on Sundays, runs half-day on Saturdays, has a vacation dip in May-June, and sells more juice in summer. Missing values and duplicates were added deliberately to practise cleaning.

## 5. Methodology
1. **Collection** - `generate_data.py` creates the CSV (or upload your own).
2. **Cleaning** - removed 179 duplicate rows; filled 90 missing prices (item median), 54 categories (from the item), 179 payment methods ("Unknown"); recomputed 90 missing totals (quantity x price). Result: 17,933 clean rows.
3. **EDA** - grouped sales by date, week, month, item, category and weekday.
4. **Statistics** - mean, median, standard deviation, coefficient of variation, one-way ANOVA on weekday revenue.
5. **Visualisation** - line, bar, donut, stacked-bar and heatmap charts (`/charts` folder and dashboard).
6. **Forecasting (optional)** - daily revenue predicted from weekday, month and time trend. Chronological 80/20 split (no shuffling). Models: weekday-average baseline, Linear Regression, Random Forest. Metrics: MAE, RMSE, R², MAPE.
7. **Recommendations** - generated from the results.

## 6. Results
| Measure | Value |
|---|---|
| Total revenue | ₹7,75,990 |
| Transactions | 17,933 |
| Items sold | 26,592 |
| Average order value | ₹43.27 |
| Best seller (quantity) | Tea (6,169 units) |
| Top revenue item | Meals (₹2,57,600) |
| Least sold item | Juice (1,558 units) |
| Top category | Meals (₹3,80,540) |
| Lowest category | Snacks (₹1,10,295) |
| Peak weekday | Wednesday (avg ₹4,046/day) |
| Quietest day | Saturday (avg ₹1,400/day) |
| Highest / lowest month | Sep 2026 (₹1,10,163) / May 2026 (₹43,202) |
| Payments | 54% UPI |

**Statistics:** mean daily revenue ₹3,316, median ₹3,717, std dev ₹1,373 (CV 41%). ANOVA: F = 32.4, p ≈ 6.7e-25, so weekday differences are statistically significant.

**Forecast (test set):**
| Model | MAE | RMSE | R² | MAPE % |
|---|---|---|---|---|
| Random Forest (best) | 713 | 838 | 0.61 | 21.4 |
| Weekday-average baseline | 986 | 1,186 | 0.22 | 22.0 |
| Linear Regression | 1,711 | 1,874 | -0.94 | 42.8 |

Random Forest beats the baseline. Linear Regression fails because it fits one straight trend through the May-June vacation dip. All forecasts are limited by having under one year of data.

**Added analyses:** 10 unusual days were flagged (revenue more than 2.5 standard deviations from the normal value for that weekday in nearby weeks). The rush hour is 1 PM - 2 PM (lunch). The forecast shows a ~95% range (forecast +/- 1.96 x test RMSE). KPIs show % change against the previous period of equal length.

## 7. Key Findings
- Meals give about a third of revenue per item, while Tea sells the most units at a low price.
- Top 3 items bring 61% of revenue.
- Midweek is busiest, Saturday is weakest; vacation months lose over half the revenue.
- Juice sells least overall but gains in summer months.

## 8. Recommendations
1. Keep Meals, Tea and Fried Rice always stocked; prepare more on Tue-Wed.
2. Reduce staff and raw material on Saturdays and in vacation months to cut wastage.
3. Run combo offers (Juice + Sandwich/Samosa) and a summer juice promotion.
4. Promote Snacks with bundles to lift the lowest category.
5. Keep UPI/QR payment reliable to shorten queues.
6. Use the forecast to plan weekly purchases; collect hour-of-day data in future.

## 9. Limitations
Synthetic data; under one year of history; no time-of-day, weather or event information; forecast has a 21% average error.

## 10. Conclusion
The project shows the full workflow from raw data to business advice. The dashboard lets canteen staff filter by date, item and category, view trends, forecast sales and download reports. With real data and more history, the same code can support real purchasing and staffing decisions.
