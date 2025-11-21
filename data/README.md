# M5 data layout

Download the M5 Accuracy dataset from the
[competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
and place `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` in
`data/raw/`. Accept the source's terms using your own account. Raw data and account
credentials are excluded from Git.

The evaluation sales file contains 30,490 bottom-level series and 1,941 labeled
days. The pipeline holds out its final 28 days. Prices join by item, store, and
`wm_yr_wk`; calendar rows join by `d`. Official protocol material is available in
[Mcompetitions/M5-methods](https://github.com/Mcompetitions/M5-methods).

`scripts/make_fixture.py` creates a small synthetic file set using this schema
for development. Fixture and subset metrics carry a subset scope label.
