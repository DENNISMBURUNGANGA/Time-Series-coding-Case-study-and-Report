# Household Appliance Energy Forecasting

Forecasting household appliance energy consumption 24 hours ahead, and testing whether
sophisticated models actually beat simple calendar heuristics.

**They don't — at least not significantly.** SARIMA achieved the lowest error of any
model tested (MASE 0.698 against the weekly seasonal naive's 0.813), but a
Diebold–Mariano test could not distinguish the two at conventional levels (p = 0.309).
Appliance energy use turns out to be structurally dominated by weekly occupancy
routine, which makes a one-line heuristic very hard to beat.

MSc Data Science coursework, University of Hertfordshire.

---

## The problem

A smart-home controller has to decide tonight how to schedule battery storage,
pre-heating and tariff-sensitive loads for tomorrow. That decision needs a 24-hour-ahead
forecast of appliance demand.

- **Data:** [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) — a single household, 10-minute readings, 137 days (Jan–May 2016), resampled to hourly
- **Horizon:** 24 hours
- **Test period:** final 14 days, evaluated as 14 non-overlapping rolling origins
- **Metrics:** MAE, RMSE, MASE (headline), Bias

## Results

| Model | MAE (Wh) | RMSE (Wh) | MASE | Bias (Wh) |
|---|---|---|---|---|
| SARIMA(1,0,1)(1,1,1)₂₄ | 37.26 | 64.53 | **0.698** | −4.68 |
| Gradient boosting — true forecast | 39.36 | 70.11 | 0.737 | −6.42 |
| Gradient boosting — conditional | 42.97 | 68.24 | 0.804 | +4.29 |
| Seasonal naive, weekly | 43.45 | 81.40 | 0.813 | −13.15 |
| Seasonal naive, daily | 48.30 | 85.56 | 0.904 | +1.75 |

Diebold–Mariano, SARIMA vs weekly seasonal naive: **p = 0.309**.

## What the analysis found

**Weekly routine beats daily recency.** The weekly seasonal naive outperforms the daily
one, so last Tuesday predicts this Tuesday better than yesterday predicts today.
Calendar features dominate every other feature family by roughly threefold in
permutation importance.

**Perfect foresight made the model worse.** Two feature sets were fitted: a *true
forecast* using only origin-available information, and a *conditional forecast* adding
realised indoor sensor and weather readings at the forecast timestamp. The conditional
model degraded from MASE 0.737 to 0.804 despite information no deployed system could
have. Indoor temperature and humidity are endogenous — physical consequences of the
appliance activity being predicted — so the model learned a contemporaneous association
that does not transfer to forecasting.

**Error tracks time of day, not forecast distance.** With origins fixed 24 hours apart,
forecast step *k* always lands on the same clock hour. The error profile is a diurnal
curve, not horizon decay — the two are confounded by the evaluation design.

**The prediction intervals are not calibrated.** SARIMA residuals are strongly
leptokurtic; the 95% interval extends below zero, which is impossible for energy
consumption. Any risk-aware use requires empirical recalibration.

## The leakage guard

The methodological core of the project.

Under a rolling origin, block *b* covers hours `O_b + 1` to `O_b + 24`. A feature at
time `t = O_b + k` built from lag `L` refers to `y(O_b + k − L)`, observable at the
origin only when `L ≥ k`. Since `k` reaches 24, **every target-derived feature must use
a lag of at least 24 hours.** Lags of 1, 2, 3, 6 or 12 — standard in one-step-ahead
work — are unavailable, and a pipeline that includes them reports one-step accuracy
while claiming a 24-hour horizon. Rolling statistics are shifted by 24, not by 1.

This is enforced by assertion and verified by regression test: all post-origin target
values are multiplied by ten and offset by 5,000 Wh, and no pre-origin feature moves.

```bash
pytest tests/
```

## Repository layout

```
├── data/
│   ├── raw/              # UCI CSV (not committed — see below)
│   └── processed/        # hourly series (regenerated)
├── notebooks/            # full analysis, Parts 1–11
├── src/appliance_energy/ # config, data preparation, EDA modules
├── scripts/              # runnable pipeline entry points
├── outputs/
│   ├── figures/          # fig1.png … fig10.png
│   └── tables/           # metrics, stationarity, feature importance
├── reports/              # final written report
├── tests/                # leakage regression tests
└── requirements.txt
```

## Running it

The dataset is not committed — it is ~12 MB and freely available. The notebook
downloads it automatically, falling back to a manual upload prompt if UCI is
unreachable.

**Colab (recommended):** open the notebook in `notebooks/` and run top to bottom. Cell 0
installs anything the runtime is missing.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_prepare_and_eda.py
```

## Limitations

- Single household, 137 days, January to May only — no summer cooling behaviour observed
- 14 independent forecast origins is a small effective sample for the Diebold–Mariano
  test, so the non-significant result is not evidence of equivalence
- Horizon and time of day are confounded by the non-overlapping origin scheme
- **The zero-shot foundation model was not evaluated.** The Chronos install failed and
  the pipeline fell back to a labelled placeholder returning the daily seasonal naive.
  No conclusion about foundation model performance can be drawn from this work.

## Stack

Python · pandas · NumPy · statsmodels · scikit-learn · Matplotlib · pytest
