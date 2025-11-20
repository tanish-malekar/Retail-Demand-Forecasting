"""Per-series Prophet with shared read-only, training-only target context."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import multiprocessing
import numpy as np
import pandas as pd

_CONTEXT = {}


def initialize_worker(directory):
    root = Path(directory)
    _CONTEXT.clear()
    _CONTEXT.update({"history": np.load(root / "history.npy", mmap_mode="r"),
                     "prices": np.load(root / "prices.npy", mmap_mode="r"),
                     "weeks": np.load(root / "weeks.npy", mmap_mode="r"),
                     "states": np.load(root / "states.npy", mmap_mode="r"),
                     "config": json.loads((root / "config.json").read_text())})
    with np.load(root / "calendar.npz", allow_pickle=False) as arrays:
        _CONTEXT.update({name: arrays[name] for name in arrays.files})


def predict_series(index):
    from prophet import Prophet
    history = _CONTEXT["history"][index].astype(float)
    config = _CONTEXT["config"]
    cutoff, horizon = len(history), config["horizon"]
    if not np.any(history > 0):
        return index, np.zeros(horizon, dtype=np.float32), False
    price = np.array(_CONTEXT["prices"][index, _CONTEXT["weeks"]], dtype=float)
    historic = price[:cutoff]
    fill = float(np.nanmedian(historic)) if np.isfinite(historic).any() else 1.0
    price = np.where(np.isfinite(price), price, fill)
    frame = pd.DataFrame({"ds": pd.to_datetime(_CONTEXT["dates"]), "price": price,
                          "snap": _CONTEXT["snap"][:, _CONTEXT["states"][index]], "event": _CONTEXT["event"]})
    training = frame.iloc[:cutoff].copy(); training["y"] = history
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=.05, seasonality_prior_scale=10, uncertainty_samples=0)
    model.add_regressor("price", standardize=True)
    model.add_regressor("snap", standardize=False)
    model.add_regressor("event", standardize=False)
    model.fit(training, seed=config["seed"] + index)
    prediction = model.predict(frame.iloc[cutoff:cutoff + horizon]).yhat.to_numpy(float)
    if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid Prophet forecast for series row {index}")
    return index, np.maximum(prediction, 0).astype(np.float32), True


def run_prophet(data, cutoff, config, directory):
    if config.prophet_workers < 1:
        raise ValueError("Prophet worker count must be positive")
    root = Path(directory); root.mkdir(exist_ok=False)
    np.save(root / "history.npy", data.sales[:, :cutoff], allow_pickle=False)
    np.save(root / "prices.npy", data.prices, allow_pickle=False)
    np.save(root / "weeks.npy", data.week_index[:cutoff + config.horizon], allow_pickle=False)
    np.save(root / "states.npy", data.state_codes, allow_pickle=False)
    calendar = data.calendar.iloc[:cutoff + config.horizon]
    np.savez(root / "calendar.npz", dates=calendar.date.dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
             snap=calendar[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(float),
             event=calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).to_numpy(float))
    (root / "config.json").write_text(json.dumps({"horizon": config.horizon, "seed": config.seed}))
    predictions = np.full((len(data.metadata), config.horizon), np.nan, dtype=np.float32)
    fitted = 0
    # Spawn keeps numerical-library thread state out of forked workers.
    with ProcessPoolExecutor(max_workers=config.prophet_workers, initializer=initialize_worker,
                             initargs=(str(root),), mp_context=multiprocessing.get_context("spawn")) as workers:
        for index, forecast, used_model in workers.map(predict_series, range(len(data.metadata)), chunksize=8):
            predictions[index] = forecast
            fitted += int(used_model)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Prophet did not cover the complete series universe")
    return predictions, {"fitted_series": fitted, "zero_history_series": len(data.metadata) - fitted,
                         "weekly_seasonality": True, "yearly_seasonality": True,
                         "known_regressors": ["price", "snap", "event"], "workers": config.prophet_workers}
