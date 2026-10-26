# College Canteen Sales Data Analysis (DSE Project)

## Project structure
```
canteen_project/
├── generate_data.py     # creates canteen_sales.csv (sample dataset)
├── canteen_analysis.py  # cleaning, EDA, statistics, forecasting, recommendations
├── app.py               # Streamlit dashboard
├── run_eda.py           # saves charts to /charts + summary_report.txt
├── canteen_sales.csv    # dataset (18,112 raw rows)
├── .streamlit/config.toml  # dashboard colours
├── requirements.txt
└── REPORT.md            # project report
```

## New features
% change vs previous period on KPIs, top-N slider, item drill-down, peak-hour heatmap (new `Hour` column), unusual-day detection, forecast confidence range, Project Info tab, custom theme (`.streamlit/config.toml`).

## Run in VS Code (step by step)
1. Install **Python 3.10+** (tick "Add Python to PATH") and **VS Code** with the *Python* extension.
2. Unzip/copy the `canteen_project` folder. In VS Code: **File > Open Folder**.
3. Open the terminal: **Terminal > New Terminal**.
4. Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
5. Install libraries: `pip install -r requirements.txt`
6. (Optional) Regenerate the dataset: `python generate_data.py`
7. (Optional) Save charts and text report: `python run_eda.py`
8. Launch the dashboard: `streamlit run app.py` - the browser opens at http://localhost:8501.
9. Stop with `Ctrl + C` in the terminal.

**Using your own data:** upload a CSV in the dashboard sidebar. It needs the columns:
`Transaction ID, Date, Food Item, Category, Quantity Sold, Unit Price, Total Amount, Payment Method`.

## Troubleshooting
- `streamlit is not recognized`: activate the venv, or run `python -m streamlit run app.py`.
- Wrong interpreter: press `Ctrl+Shift+P` > *Python: Select Interpreter* > choose the `venv` one.

## Likely viva questions
- **Why clean data first?** Duplicates inflate revenue; missing values break calculations.
- **Why a time-based split?** Shuffling would let the model "see the future" (data leakage).
- **What do MAE / RMSE / R² / MAPE mean?** Average error / error that punishes big misses / share of variance explained / percentage error.
- **What is ANOVA used for here?** Tests whether average revenue really differs between weekdays (p < 0.05 = yes).
- **Limitations?** Data is synthetic; one year of history cannot learn yearly seasonality; no hour-of-day column.
