import numpy as np
from .data import META

LAGS = (1, 7, 14, 28)
NAMES = (META[1:] + ["price", "price_ratio", "price_change_7"] + [f"lag_{lag}" for lag in LAGS]
         + ["mean_7", "mean_28", "std_28", "zero_share_28"]
         + ["weekday", "month", "year", "day_of_month", "iso_week", "event_name_1", "event_name_2", "snap"])
CATEGORICAL = META[1:] + ["event_name_1", "event_name_2"]


class FeatureBuilder:
    names = NAMES
    categorical = CATEGORICAL

    def __init__(self, data, cutoff):
        self.data = data
        if cutoff < 28 or cutoff > data.sales.shape[1]:
            raise ValueError("Invalid training cutoff")
        counts = np.bincount(data.week_index[:cutoff], minlength=data.prices.shape[1]).astype(float)
        valid = np.isfinite(data.prices)
        numerator = np.where(valid, data.prices, 0) @ counts
        denominator = valid.astype(float) @ counts
        self.training_price_mean = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
        calendar = data.calendar
        self.time = np.column_stack([calendar.date.dt.weekday, calendar.date.dt.month, calendar.date.dt.year,
                                     calendar.date.dt.day, calendar.date.dt.isocalendar().week]).astype(np.float32)
        self.snap = calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.float32)

    def day(self, t, history):
        if t < 28 or t >= len(self.data.calendar) or history.shape[0] != len(self.data.metadata):
            raise ValueError("Feature date or history shape is invalid")
        recent = history[:, t - 28:t]
        if recent.shape[1] != 28 or not np.isfinite(recent).all():
            raise ValueError("Lag history must contain only observed or recursively predicted past values")
        prices = self.data.day_prices([t, t - 7])
        current, previous = prices[:, 0], prices[:, 1]
        change = np.divide(current, previous, out=np.full_like(current, np.nan), where=previous > 0) - 1
        n = len(self.data.metadata)
        columns = [self.data.metadata_codes,
                   np.column_stack([current, current / self.training_price_mean, change]),
                   np.column_stack([history[:, t - lag] for lag in LAGS]),
                   np.column_stack([recent[:, -7:].mean(axis=1), recent.mean(axis=1), recent.std(axis=1), (recent == 0).mean(axis=1)]),
                   np.tile(self.time[t], (n, 1)), np.tile(self.data.event_codes[t], (n, 1)),
                   self.snap[t, self.data.state_codes][:, None]]
        result = np.column_stack(columns).astype(np.float32)
        if result.shape[1] != len(self.names) or np.isinf(result).any():
            raise ValueError("Invalid feature matrix")
        return result
