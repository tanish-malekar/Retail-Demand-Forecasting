from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from .demand_paths import historical_copula_rho, sample_paths
from .inventory import Inventory, allocate_cost_budget, simulate
from .quantiles import forecast_quantiles, quantile_metrics, train_quantiles


def run_inventory(data, builder, cutoff, point_forecast, directory, config, series_limit=None):
    directory = Path(directory)
    horizon = point_forecast.shape[1]
    config.validate(horizon)
    n = len(data.metadata) if series_limit is None else series_limit
    if not 1 <= n <= len(data.metadata):
        raise ValueError("Inventory series limit must be within the selected data universe")
    models = train_quantiles(directory, config.levels)
    quantiles, crossing = forecast_quantiles(models, config.levels, data, builder, cutoff, horizon)
    np.save(directory / "quantile_forecast.npy", quantiles, allow_pickle=False)
    curve_records, curves, items = [], [], []
    baseline_budget = 0.
    latest_prices = data.day_prices([cutoff - 1])[:, 0]
    for index in range(n):
        rho = historical_copula_rho(data.sales[index, :cutoff])
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index, 1]))
        paths = sample_paths(quantiles[index], config.levels, config.calibration_paths, rho, rng)
        mean = float(point_forecast[index].mean())
        quantity = max(1, int(np.ceil(mean * config.cycle_days)))
        std = float(data.sales[index, max(0, cutoff - 56):cutoff].std())
        baseline_r = max(0, int(np.ceil(mean * config.lead_days + norm.ppf(config.baseline_service) * std * np.sqrt(config.lead_days))))
        price = float(latest_prices[index])
        if not np.isfinite(price):
            price = float(builder.training_price_mean[index])
        unit_cost = price * config.unit_cost_fraction
        baseline = simulate(paths, baseline_r, quantity, config.lead_days, unit_cost)
        baseline_budget += float(baseline["capital"].mean())
        lead_demand = paths[:, :config.lead_days].sum(axis=1)
        candidates = np.unique(np.r_[0, baseline_r,
            np.ceil(np.quantile(lead_demand, [.5, .65, .75, .8, .85, .9, .925, .95, .975, .99, .995])),
            max(int(paths.sum(axis=1).max()) + quantity, 2 * baseline_r + quantity)]).astype(int)
        curve = []
        for threshold in candidates:
            outcomes = simulate(paths, threshold, quantity, config.lead_days, unit_cost)
            capital, stockout = float(outcomes["capital"].mean()), float(outcomes["stockout_days"].mean())
            curve.append([threshold, capital, stockout])
            curve_records.append({"series_row": index, "id": data.metadata.id.iloc[index],
                                  "reorder_point": int(threshold), "capital": capital, "stockout_days": stockout})
        curves.append(np.asarray(curve))
        items.append({"series_row": index, "id": data.metadata.id.iloc[index], "baseline_reorder_point": baseline_r,
                      "order_quantity": quantity, "unit_cost": unit_cost, "copula_rho": rho})
    cheap, expensive, probability, budget = allocate_cost_budget(curves, baseline_budget)
    pd.DataFrame(curve_records).to_csv(directory / "inventory_calibration_curves.csv", index=False)
    totals = {name: np.zeros(config.paths) for name in ("baseline_stockout_days", "proposed_stockout_days", "baseline_capital", "proposed_capital", "baseline_lost_units", "proposed_lost_units", "demand_units")}
    policy_rows, comparison_rows = [], []
    for index, item in enumerate(items):
        low_r, high_r = int(curves[index][cheap[index], 0]), int(curves[index][expensive[index], 0])
        policy_rows.append({**item, "low_reorder_point": low_r, "high_reorder_point": high_r,
                            "high_policy_probability": probability})
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index, 2]))
        paths = sample_paths(quantiles[index], config.levels, config.paths, item["copula_rho"], rng)
        chooser = np.random.default_rng(np.random.SeedSequence([config.seed, index, 3]))
        threshold = np.where(chooser.random(config.paths) < probability, high_r, low_r)
        baseline = simulate(paths, item["baseline_reorder_point"], item["order_quantity"], config.lead_days, item["unit_cost"])
        proposed = simulate(paths, threshold, item["order_quantity"], config.lead_days, item["unit_cost"])
        totals["demand_units"] += baseline["demand_units"]
        for name in ("stockout_days", "capital", "lost_units"):
            totals[f"baseline_{name}"] += baseline[name]
            totals[f"proposed_{name}"] += proposed[name]
        comparison_rows.append({"id": item["id"], **{f"{policy}_{name}": float(values[name].mean())
            for policy, values in (("baseline", baseline), ("proposed", proposed))
            for name in ("stockout_days", "capital", "lost_units")}})
    pd.DataFrame(policy_rows).to_csv(directory / "reorder_policies.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(directory / "inventory_by_item.csv", index=False)
    averages = {name: float(value.mean()) for name, value in totals.items()}
    gap = averages["proposed_capital"] / averages["baseline_capital"] - 1
    matched = abs(gap) <= config.cost_tolerance
    baseline_events = averages["baseline_stockout_days"]
    reduction, interval = None, None
    if matched and baseline_events > 0:
        ratio = averages["proposed_stockout_days"] / baseline_events
        influence = (totals["proposed_stockout_days"] - averages["proposed_stockout_days"]) / baseline_events - ratio * (totals["baseline_stockout_days"] - baseline_events) / baseline_events
        se = float(np.std(influence, ddof=1) / np.sqrt(config.paths) * 100)
        reduction = float((1 - ratio) * 100)
        interval = [reduction - 1.96 * se, reduction + 1.96 * se]
    report = {"scope": "full_m5" if data.full_m5 and n == len(data.metadata) else "subset", "inventory_series": n,
              "model_training_series": len(data.metadata), "horizon_days": horizon,
              "configuration": asdict(config), "quantile_metrics": quantile_metrics(data.sales[:n, cutoff:], quantiles[:n], config.levels),
              "quantile_rearrangement": crossing, "budget": budget,
              "evaluation_means": averages, "realized_capital_gap_percent": float(gap * 100),
              "capital_match_within_tolerance": matched, "matched_stockout_reduction_percent": reduction,
              "matched_reduction_monte_carlo_95_interval": interval,
              "baseline_stockout_day_rate": baseline_events / (n * horizon),
              "proposed_stockout_day_rate": averages["proposed_stockout_days"] / (n * horizon),
              "assumptions": ["Lost sales, daily review, fixed order quantity, deterministic lead time",
                              "Inventory capital is end-of-day on-hand plus in-transit units times a selling-price-derived unit-cost proxy",
                              "Initial on-hand equals reorder point plus order quantity; initial pipeline is empty",
                              "Conditional marginal quantiles use shared median recursion; Gaussian AR(1) copula supplies temporal dependence",
                              "Item paths are independent; upper-tail extrapolation is bounded; integer demand uses stochastic rounding",
                              "Policies fit independent calibration paths; evaluation uses shared fresh paths and no held-out demand labels for selection",
                              "A per-replication low/high reorder-point mixture matches the calibration budget in expectation"]}
    (directory / "inventory_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
