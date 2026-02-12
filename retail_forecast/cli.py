import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
from .baseline import run_prophet
from .data import load_dataset
from .features import FeatureBuilder
from .metrics import dollar_weights, wrmsse
from .model import Training, recursive_forecast, train_global


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_forecast(directory, name, data, prediction):
    np.save(directory / f"{name}.npy", prediction, allow_pickle=False)
    frame = pd.DataFrame(prediction, columns=[f"F{i + 1}" for i in range(prediction.shape[1])])
    frame.insert(0, "id", data.metadata.id.to_numpy())
    frame.to_csv(directory / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Chronological M5 demand forecasting")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--limit", type=int, help="Use a labeled subset of bottom series")
    parser.add_argument("--with-prophet", action="store_true", help="Fit every selected series separately for a matched comparison")
    parser.add_argument("--with-inventory", action="store_true", help="Fit quantile models and compare simulated reorder policies")
    parser.add_argument("--inventory-config", type=Path, default=Path("configs/inventory.json"))
    parser.add_argument("--inventory-series", type=int, help="Limit the inventory comparison universe, while retaining the global model")
    args = parser.parse_args()
    config = Training(**json.loads(args.config.read_text()))
    if config.horizon != 28:
        raise ValueError("This comparison fixes the evaluation horizon at 28 days")
    data = load_dataset(args.data, args.limit)
    cutoff = data.sales.shape[1] - config.horizon
    if cutoff < 56:
        raise ValueError("Insufficient pre-holdout history")
    builder = FeatureBuilder(data, cutoff)
    args.output.mkdir(parents=True, exist_ok=False)
    data.metadata.to_csv(args.output / "series.csv", index=False)
    write_json(args.output / "categories.json", data.categories)
    np.save(args.output / "training_price_mean.npy", builder.training_price_mean, allow_pickle=False)
    model = train_global(data, builder, cutoff, config, args.output)
    prediction = recursive_forecast(model, data, builder, cutoff, config.horizon)
    save_forecast(args.output, "lightgbm_forecast", data, prediction)
    actual = data.sales[:, cutoff:]
    revenue = dollar_weights(data.sales[:, cutoff - 28:cutoff], data.day_prices(np.arange(cutoff - 28, cutoff)))
    score, detail = wrmsse(data.metadata, data.sales[:, :cutoff], actual, prediction, revenue)
    detail.to_csv(args.output / "lightgbm_hierarchy.csv", index=False)
    if data.full_m5 and score["hierarchy_series"] != 42840:
        raise ValueError("Full M5 metadata should produce 42,840 aggregate series")
    report = {"scope": "full_m5" if data.full_m5 else "subset", "bottom_series": len(data.metadata),
              "source_sha256": data.hashes, "config": asdict(config), "cutoff_day": cutoff,
              "training_end": data.calendar.date.iloc[cutoff - 1].date().isoformat(),
              "holdout_start": data.calendar.date.iloc[cutoff].date().isoformat(),
              "holdout_end": data.calendar.date.iloc[-1].date().isoformat(),
              "weight_window_days": 28, "scale_window": "All history before holdout, trimmed before first positive sale",
              "lightgbm": score, "prophet": None, "relative_wrmsse_reduction_percent": None,
              "versions": {"python": platform.python_version(), **{name: version(name) for name in ("lightgbm", "prophet", "numpy", "pandas", "scipy")}},
              "future_covariate_assumption": "Calendar events, SNAP, and scheduled sell prices are known over the forecast horizon"}
    write_json(args.output / "evaluation.json", report)
    if args.with_prophet:
        baseline, baseline_config = run_prophet(data, cutoff, config, args.output / "prophet_context")
        save_forecast(args.output, "prophet_forecast", data, baseline)
        baseline_score, baseline_detail = wrmsse(data.metadata, data.sales[:, :cutoff], actual, baseline, revenue)
        baseline_detail.to_csv(args.output / "prophet_hierarchy.csv", index=False)
        report["prophet"] = {**baseline_score, "configuration": baseline_config}
        if baseline_score["wrmsse"] > 0:
            report["relative_wrmsse_reduction_percent"] = float((1 - score["wrmsse"] / baseline_score["wrmsse"]) * 100)
        write_json(args.output / "evaluation.json", report)
    if args.with_inventory:
        from .inventory import Inventory
        from .inventory_run import run_inventory
        inventory_config = Inventory(**json.loads(args.inventory_config.read_text()))
        report["inventory"] = run_inventory(data, builder, cutoff, prediction, args.output, inventory_config, args.inventory_series)
        write_json(args.output / "evaluation.json", report)
    from .plots import plot_totals
    plot_totals(data.calendar.date.iloc[cutoff:], actual, prediction, args.output / "demand.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
