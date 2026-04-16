# 📈 Time Series Forecasting Studio

**Utah State University · DATA 5630 Final Project**  
**Author:** Sierra Reynolds · Huntsman School of Business

---

## Overview

A Streamlit web app for end-to-end time series forecasting. Load a built-in demo dataset or upload your own CSV, configure your models and split, and get interactive forecasts with performance metrics.

---

## Features

- **Built-in demo dataset** — 30-Year U.S. Mortgage Rate (1971–2024), no download needed
- **File upload** with date column + target variable selectors
- **Missing value handling** — forward fill, linear interpolation, or drop
- **Date range filter** — optionally restrict to last N periods
- **Train/test holdout split** — configurable via slider
- **3 forecasting models** selectable via checkboxes:
  - Holt-Winters Exponential Smoothing (statsmodels)
  - ARIMA(1,1,1) (statsmodels)
  - Random Forest with lag + rolling-mean features (scikit-learn)
- **Metrics table** — MAE, RMSE, MAPE with best-model highlighting
- **Interactive Plotly charts** — actual vs forecast overlay
- **Future forecast download** as CSV
- **About page** via sidebar navigation

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run forecasting_app_3col_SR.py
```

---

## Data Split

Simple holdout split. The slider controls what percentage of observations go to training; the remainder forms the test set. The Random Forest uses recursive one-step-ahead prediction for the future horizon.

---

## Metrics Guide

| Metric | Meaning | Lower is better |
|--------|---------|-----------------|
| MAE | Mean Absolute Error — average error in original units | ✅ |
| RMSE | Root Mean Squared Error — penalizes large errors more | ✅ |
| MAPE | Mean Absolute Percentage Error — scale-independent % | ✅ |

---

## Models

| Model | Type | Library |
|---|---|---|
| Holt-Winters (ETS) | Exponential smoothing with additive trend + seasonality | statsmodels |
| ARIMA(1,1,1) | Classic differencing/autoregressive model | statsmodels |
| Random Forest | Ensemble ML with hand-crafted lag + rolling-mean features | scikit-learn |

**Random Forest feature engineering:** 6 lag features (t-1 through t-6) + 3-period rolling mean + 6-period rolling mean, all constructed manually using pandas — no AutoML or PyCaret.

---

## Project Structure

```
├── forecasting_app_3col_SR.py   # Main Streamlit app
├── theme_manager.py             # UI theme definitions and CSS
├── requirements.txt             # Python dependencies
└── README.md
```
