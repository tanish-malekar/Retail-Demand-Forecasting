from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd

META = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FILES = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Dataset:
    metadata: pd.DataFrame
    sales: np.ndarray
    calendar: pd.DataFrame
    prices: np.ndarray
    week_index: np.ndarray
    categories: dict
    metadata_codes: np.ndarray
    state_codes: np.ndarray
    event_codes: np.ndarray
    hashes: dict

    @property
    def full_m5(self):
        return len(self.metadata) == 30490

    def day_prices(self, days):
        return self.prices[:, self.week_index[np.asarray(days)]]


def load_dataset(directory, limit=None):
    directory = Path(directory)
    for name in FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Place {name} in {directory}; see data/README.md")
    columns = pd.read_csv(directory / FILES[0], nrows=0).columns
    days = sorted((c for c in columns if re.fullmatch(r"d_[0-9]+", c)), key=lambda c: int(c[2:]))
    if len(days) < 84 or days != [f"d_{i}" for i in range(1, len(days) + 1)]:
        raise ValueError("Expected contiguous daily sales columns with enough history")
    if limit is not None and limit < 1:
        raise ValueError("Series limit must be positive")
    raw = pd.read_csv(directory / FILES[0], usecols=META + days, dtype={c: "float32" for c in days}, nrows=limit)
    if raw[META].isna().any().any():
        raise ValueError("Series metadata must not contain missing identities")
    metadata = raw[META].astype(str).reset_index(drop=True)
    if metadata.id.duplicated().any() or metadata.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Series identities must be unique")
    if limit is None and len(metadata) != 30490:
        raise ValueError("Full M5 evaluation requires 30,490 series; use --limit for a subset")
    sales = raw[days].to_numpy(dtype=np.float32, copy=True)
    del raw
    if not np.isfinite(sales).all() or np.any(sales < 0):
        raise ValueError("Sales must be finite and nonnegative")
    calendar = pd.read_csv(directory / FILES[1]).set_index("d")
    if not calendar.index.is_unique:
        raise ValueError("Calendar day identities must be unique")
    calendar = calendar.loc[days].copy().reset_index()
    calendar["date"] = pd.to_datetime(calendar.date)
    if calendar.date.isna().any() or not calendar.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Calendar must contain consecutive daily dates")
    snap = calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(float)
    if not np.isfinite(snap).all() or not np.isin(snap, [0, 1]).all() or calendar.wm_yr_wk.isna().any():
        raise ValueError("Calendar SNAP flags and week identities must be valid")
    weeks = sorted(calendar.wm_yr_wk.unique())
    week_map = {value: index for index, value in enumerate(weeks)}
    week_index = calendar.wm_yr_wk.map(week_map).to_numpy(np.int32)
    prices = pd.read_csv(directory / FILES[2], usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"], dtype={"sell_price": "float32"})
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Weekly prices must be unique per item-store")
    if not np.isfinite(prices.sell_price).all() or (prices.sell_price <= 0).any():
        raise ValueError("Observed prices must be finite and positive")
    keys = metadata[["store_id", "item_id"]].assign(series_row=np.arange(len(metadata), dtype=np.int32))
    prices = prices.merge(keys, on=["store_id", "item_id"], how="inner", validate="many_to_one")
    prices = prices.loc[prices.wm_yr_wk.isin(weeks)]
    price_matrix = np.full((len(metadata), len(weeks)), np.nan, dtype=np.float32)
    price_matrix[prices.series_row.to_numpy(), prices.wm_yr_wk.map(week_map).to_numpy()] = prices.sell_price.to_numpy()
    del prices
    categories = {name: sorted(metadata[name].unique().tolist()) for name in META[1:]}
    codes = np.column_stack([metadata[name].map({value: i for i, value in enumerate(categories[name])}).to_numpy(np.int32) for name in META[1:]])
    state_map = {"CA": 0, "TX": 1, "WI": 2}
    if not metadata.state_id.isin(state_map).all():
        raise ValueError("M5 states must be CA, TX, or WI")
    events = []
    for name in ("event_name_1", "event_name_2"):
        labels = calendar[name].fillna("__none__").astype(str)
        categories[name] = sorted(labels.unique())
        events.append(labels.map({label: i for i, label in enumerate(categories[name])}).to_numpy(np.int32))
    return Dataset(metadata, sales, calendar, price_matrix, week_index, categories, codes,
                   metadata.state_id.map(state_map).to_numpy(np.int32), np.column_stack(events),
                   {name: sha256_file(directory / name) for name in FILES})
