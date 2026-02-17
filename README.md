# Retail Demand Forecasting and Inventory Simulation

A demand-planning project using a global LightGBM model, per-series Prophet
baselines, pandas feature preparation, and hierarchical M5 evaluation. It forecasts
all **30,490 item-store series** over a chronological **28-day holdout**, then
compares both methods using the same hierarchy, scale window, and dollar weights.

## Data and split

Place the M5 Accuracy files in `data/raw/`:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Obtain them from the [M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
under its access terms. The repository contains source code and a synthetic
fixture generator; the M5 data itself stays outside version control. The official
[competition methods and guide](https://github.com/Mcompetitions/M5-methods)
describe the hierarchy and evaluation protocol.

For the evaluation CSV with 1,941 labeled days, the pipeline trains through
`d_1913` and holds out `d_1914`–`d_1941`. Other compatible daily lengths use their
last 28 labeled days. LightGBM's supervised training matrix uses the most recent
365 pre-holdout days by default; lag warmup and RMSSE scales can use earlier
training history. A fixed configured tree count keeps holdout targets out of
early stopping or model selection.

## Global LightGBM

One Poisson-objective model learns across item, department, category, store, and
state identifiers. Features include lagged sales at 1, 7, 14, and 28 days; shifted
7/28-day means; 28-day variability and zero share; current price, training-relative
price, and 7-day price change; weekday, month, year, day-of-month, ISO week;
event identities; and state-specific SNAP eligibility.

Every demand feature uses times strictly before its forecast day. During the
28-day forecast, earlier model predictions are fed into later lags and rolling
windows. Held-out sales are never copied into that history. Calendar events,
SNAP, and scheduled future prices are treated as known planning inputs; revenue
or future unit sales are not features. Missing pre-launch prices remain missing
for LightGBM, rather than being backward-filled from a future launch.

Categorical vocabularies come from series metadata and known calendar inputs.
Price normalization uses pre-holdout weeks only. Inputs are validated for unique
series, unique weekly prices, contiguous dates, finite nonnegative units, and
positive observed prices. SHA-256 hashes identify all three source files.

## Prophet comparison

The optional baseline fits one Prophet model per selected item-store series, with
weekly/yearly seasonality and known price, SNAP, and event-presence regressors.
Price imputation uses each series' training median only. All-zero training histories
produce a zero forecast and are counted separately. Other fitting failures stop
the comparison instead of silently substituting a different forecaster.

Prophet uses all pre-holdout history, while LightGBM uses the configured recent
supervised window with longer lag history available. Both receive known calendar
and price inputs, use nonnegative point forecasts, and are scored on identical
series and days. The models have different inductive biases and information
representations; the report records the baseline settings so the comparison is
interpretable. The full baseline involves 30,490 individual fits and can require
substantial CPU time. Spawned workers share read-only memory-mapped context files.

## WRMSSE

The scorer aggregates bottom-level forecasts and observations through all twelve
M5 levels: total; state; store; category; department; state-category;
state-department; store-category; store-department; item; item-state; item-store.
The full universe yields **42,840 aggregate series**.

For each aggregate series, RMSSE divides holdout mean squared error by the mean
squared one-day difference over training history after its first positive sale.
Weights use sales revenue in the final 28 training days, aggregated at each level
and normalized within that level. WRMSSE averages the twelve weighted level scores.
The scorer computes scales in bounded groups of 256 aggregate series.

A positive unit sale without a known positive historical price is rejected.
Zero-weight nodes contribute zero. A positive-weight node with a zero scale makes
the metric undefined and is reported as an input error rather than stabilized
with an arbitrary epsilon. Subset runs are explicitly labeled and normalize over
the selected universe; their scores are distinct from full-M5 scores.

## Setup and usage

Requires Python 3.11–3.13. LightGBM may require the platform's OpenMP runtime.
Dependencies are pinned in `pyproject.toml`.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
m5-forecast --data data/raw --output artifacts/global-01
m5-forecast --data data/raw --output artifacts/comparison-01 --with-prophet
```

Output directories must be new. `configs/default.json` controls the supervised
window, boosting rounds, tree settings, thread count, seed, and Prophet workers.
The full training matrix is disk-backed float32, avoiding a single melted panel
DataFrame. The default 30,490 × 365 × 24 feature matrix occupies roughly 1.1 GB,
plus targets, model bins, source history, and working memory. Reserve sufficient
RAM and disk, and adjust the window or series count for smaller environments.

```sh
m5-forecast --data data/raw --output artifacts/subset-01 --limit 128 --with-prophet
python scripts/make_fixture.py --output data/raw-fixture --series 12 --days 140
m5-forecast --data data/raw-fixture --output artifacts/fixture-01 --limit 12
python -m pytest
```

The fixture is explicitly synthetic and supports pipeline development. Full runs
require exactly 30,490 rows; `--limit` enables a scoped subset. To limit nested
numerical-library parallelism when using many Prophet workers, set the appropriate
BLAS/OpenMP thread environment before starting the process.

## Quantile forecasts and inventory simulation

`--with-inventory` fits global LightGBM quantile models on the same pre-holdout
feature matrix, then generates **10,000 demand paths per selected item** for a
28-day inventory comparison. Independent calibration paths select policy settings.

```sh
m5-forecast --data data/raw --output artifacts/inventory-01 --with-inventory
m5-forecast --data data/raw --output artifacts/inventory-subset --limit 128 --with-prophet --with-inventory
```

`configs/inventory.json` controls quantiles, path counts, lead time, cycle quantity,
service level, unit-cost proxy, random seed, and cost-match tolerance.
`--inventory-series` can scope simulation separately from global-model training.
The pipeline reports pinball loss, interval coverage, and quantile crossings;
monotone rearrangement corrects crossings before demand sampling.

Quantile forecasts share a recursively updated median demand history. A Gaussian
AR(1) copula supplies within-item temporal dependence, with a shrunk rank-based
correlation estimated from training demand innovations. Inverse-CDF interpolation
and stochastic rounding generate nonnegative integer demand. The distribution
has explicit bounded upper-tail extrapolation; cross-item shocks are independent.
These are modeled demand scenarios rather than observed future sales.

The simulator uses daily review, lost sales, fixed order quantities, and a
configurable deterministic replenishment lead time. Each policy starts with its
reorder point plus one order quantity on hand and an empty pipeline. Inventory
capital includes both on-hand and in-transit units, valued at a configurable
fraction of the last known selling price as a procurement-cost proxy.

The baseline uses a mean-plus-normal-safety-stock reorder point. Candidate reorder
points come from simulated lead-time demand quantiles. A portfolio Lagrangian
allocation chooses low/high per-item reorder points and a replication-level mixing
probability to match the baseline's **expected calibration inventory capital**.
Both policies face the same fresh evaluation demand paths. The report includes
realized cost gaps and only emits `matched_stockout_reduction_percent` when the
cost gap falls within the configured tolerance. Negative reductions indicate
worse stockout performance; a zero-stockout baseline has no relative reduction.
The uncertainty interval reflects Monte Carlo sampling error, not all model or
business uncertainty. Policy selection never uses held-out observed sales.

Artifacts include `quantile_forecast.npy`, quantile model files,
`inventory_calibration_curves.csv`, `reorder_policies.csv`,
`inventory_by_item.csv`, and `inventory_evaluation.json`. Full-universe simulation
is computationally intensive, so item-by-item processing bounds path memory.

## Artifacts

- `lightgbm.txt` — fitted global model
- `training_matrix.json` and float32 matrix files — feature order, shape, and settings
- `categories.json`, `series.csv`, `training_price_mean.npy` — reproducible encodings
- `lightgbm_forecast.npy/.csv` — bottom-series forecasts with exact ID order
- `lightgbm_hierarchy.csv` — per-node scales, weights, errors, and score contributions
- `evaluation.json` — split dates, data hashes, scope, versions, WRMSSE, and comparison
- `feature_importance.csv` and `demand.png` — model and holdout summaries
- Optional Prophet forecasts, hierarchy details, and read-only worker context

`relative_wrmsse_reduction_percent` is computed as `100 × (1 - global / Prophet)`
only after a complete matched baseline is available. Negative values indicate the
global model scored worse. Predictions and reports are generated from pipeline
outputs, and experiment folders remain outside version control.

## Layout

`data.py` validates and aligns M5 files; `features.py` builds historical features;
`model.py` trains and recursively forecasts; `baseline.py` handles per-series
Prophet workers; `metrics.py` constructs the hierarchy and WRMSSE; `cli.py` writes
experiment artifacts. Test contracts cover first-sale trimming, hierarchical
weighting, zero-scale handling, historical price requirements, and recursive
prediction independence from held-out labels.

Sources: [M5 methods](https://github.com/Mcompetitions/M5-methods),
[LightGBM](https://lightgbm.readthedocs.io/), and
[Prophet](https://github.com/facebook/prophet).

Project code is MIT licensed. Dataset access follows the source's terms.
