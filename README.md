# Cafeteria Order Data Analysis and 7-Day Demand Forecasting

### Evaluation Assignment: Kanishka Software Private Limited
**Author:** Abhishek Mishra  
**Focus:** Large-scale SQL Stream Ingestion, Exploratory Data Analysis, and Multi-Model 7-Day Order Demand Forecasting  

---

## Overview

This repository contains the analysis and forecasting pipeline for enterprise cafeteria operations across 18 tech park branches for the 2024–2025 financial year (April 1, 2024 to April 1, 2025). The source database contains approximately 5.96 million orders and 7.42 million line items (~11 GB SQL dump).

Key components:
1. Streaming ETL parser to process large SQL dumps without high RAM consumption.
2. Exploratory Data Analysis identifying meal windows, revenue leaders, and channel adoption.
3. Multi-model evaluation comparing classical time-series and machine learning regressors on a 14-day holdout test set.
4. Out-of-sample 7-day order demand forecast with 95% confidence intervals for Embassy Tech Village.
5. Actionable recommendations for kitchen staffing, queue reduction, and menu margin optimization.

---

## Repository Structure

```
.
├── notebooks/
│   └── cafeteria_analysis_and_forecast.ipynb  # Interactive walkthrough notebook
├── scripts/
│   ├── 01_process_dataset.py                  # High-speed streaming SQL parser
│   ├── 02_exploratory_data_analysis.py        # Statistical metrics & summary tables
│   ├── 03_forecasting_models.py               # Model training, test scoring & 7-day forecast
│   └── generate_charts.py                     # Visualization generator
├── visualizations/                            # Generated publication-quality PNG charts
│   ├── 01_branch_order_volume_comparison.png
│   ├── 02_monthly_sales_trend_by_branch.png
│   ├── 03_hourly_peak_order_distribution.png
│   ├── 04_top_15_selling_dishes.png
│   ├── 05_order_channel_and_payment_breakdown.png
│   ├── 06_daily_orders_time_series_selected_branch.png
│   ├── 07_forecast_comparison_next_7_days.png
│   └── 08_customer_retention_and_frequency.png
├── REPORT.md                                  # Complete evaluation report with insights
└── README.md
```

*Note: Raw SQL dumps (`Cafeteria Order Data.sql`, `users.sql`) and raw archives are excluded from Git tracking via `.gitignore`.*

---

## Environment Setup

### Requirements
- Python 3.10+
- Dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

---

## Pipeline Execution

Run the pipeline scripts in numerical order:

```bash
# 1. Parse raw SQL dump and generate aggregated summary datasets
python scripts/01_process_dataset.py

# 2. Run statistical exploratory data analysis
python scripts/02_exploratory_data_analysis.py

# 3. Train forecasting models, evaluate holdout metrics, and produce 7-day forecast
python scripts/03_forecasting_models.py

# 4. Generate all charts
python scripts/generate_charts.py
```

---

## Summary of Findings

- **Total Completed Orders:** 5,961,005
- **Gross Revenue:** INR 411,060,742.68
- **Average Order Value (AOV):** INR 68.96
- **Market Concentration:** Embassy Tech Village (41.6%) and Nirlon Knowledge Park (39.3%) generate 80.9% of total order volume.
- **Ordering Channels:** Mobile App (60.2%), Self-Ordering Kiosks (22.6%), Manned POS (17.2%). Digital payment adoption exceeds 92%.
- **Operating Peaks:** Lunch peak occurs between 12:00 PM and 2:30 PM (24.5%), while the evening tea and snack surge between 4:00 PM and 6:30 PM generates 31.2% of daily transactions.

---

## Model Benchmark & 7-Day Forecast

### Holdout Test Set Performance (Embassy Tech Village)

| Model | MAE | RMSE | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **856.41** | **1,785.39** | **24.53%** |
| Random Forest Regressor | 964.62 | 1,833.38 | 24.05% |
| Holt-Winters Exponential Smoothing | 1,217.58 | 2,167.41 | 61.20% |
| SARIMAX(1,1,1)(1,1,1,7) | 1,230.43 | 2,205.17 | 68.36% |
| Seasonal Naive Baseline (Lag 7) | 1,558.29 | 2,729.73 | 33.40% |

### Out-of-Sample 7-Day Forward Forecast (April 2 – April 8, 2025)

| Date | Day | Forecast Orders | 95% CI Lower | 95% CI Upper | Est. Revenue (INR) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 2025-04-02 | Wednesday | 8,388 | 5,605 | 11,171 | 636,099.74 |
| 2025-04-03 | Thursday | 8,579 | 5,499 | 11,659 | 650,584.13 |
| 2025-04-04 | Friday | 7,362 | 4,211 | 10,513 | 558,293.55 |
| 2025-04-05 | Saturday | 677 | 0 | 3,850 | 51,339.95 |
| 2025-04-06 | Sunday | 189 | 0 | 3,371 | 14,332.72 |
| 2025-04-07 | Monday | 8,103 | 4,914 | 11,291 | 614,486.91 |
| 2025-04-08 | Tuesday | 9,453 | 6,260 | 12,646 | 716,863.48 |
| **Total** | **7-Day Period** | **42,751** | — | — | **3,242,000.48** |

For the in-depth discussion, methodology, and operational recommendations, refer to [REPORT.md](REPORT.md).
