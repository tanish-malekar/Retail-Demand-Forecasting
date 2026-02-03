from pathlib import Path
import json
import lightgbm as lgb
import numpy as np


def train_quantiles(directory, levels):
    directory = Path(directory)
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 3 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1 or not np.any(levels == .5):
        raise ValueError("Use increasing quantile levels in (0,1), including 0.5")
    spec = json.loads((directory / "training_matrix.json").read_text())
    features = np.memmap(directory / "training_features.f32", mode="r", dtype="float32", shape=tuple(spec["shape"]))
    target = np.memmap(directory / "training_targets.f32", mode="r", dtype="float32", shape=(spec["shape"][0],))
    dataset = lgb.Dataset(features, label=target, feature_name=spec["features"], categorical_feature=spec["categorical"])
    models = []
    for level in levels:
        parameters = {**spec["parameters"], "objective": "quantile", "metric": "quantile", "alpha": float(level)}
        model = lgb.train(parameters, dataset, num_boost_round=spec["boosting_rounds"])
        model.save_model(str(directory / f"quantile_{level:.4f}.txt"))
        models.append(model)
    (directory / "quantile_models.json").write_text(json.dumps({"levels": levels.tolist(),
        "recursion": "Median history shared by conditional marginal quantile models"}, indent=2) + "\n")
    return models


def forecast_quantiles(models, levels, data, builder, cutoff, horizon):
    levels = np.asarray(levels, float)
    if len(models) != len(levels) or not np.any(levels == .5):
        raise ValueError("One model per quantile and a median are required")
    history = np.full((len(data.metadata), cutoff + horizon), np.nan, dtype=np.float32)
    history[:, :cutoff] = data.sales[:, :cutoff]
    result = np.empty((len(data.metadata), horizon, len(levels)), dtype=np.float32)
    median = int(np.flatnonzero(levels == .5)[0])
    crossing_count = 0
    for step in range(horizon):
        features = builder.day(cutoff + step, history)
        raw = np.column_stack([model.predict(features) for model in models])
        if raw.shape != (len(data.metadata), len(levels)) or not np.isfinite(raw).all():
            raise ValueError("Invalid quantile predictions")
        crossing_count += int(np.any(np.diff(raw, axis=1) < 0, axis=1).sum())
        result[:, step] = np.sort(np.maximum(raw, 0), axis=1)
        history[:, cutoff + step] = result[:, step, median]
    return result, {"crossed_series_days_before_rearrangement": crossing_count}


def quantile_metrics(actual, quantiles, levels):
    actual, quantiles, levels = np.asarray(actual, float), np.asarray(quantiles, float), np.asarray(levels, float)
    if quantiles.shape != (*actual.shape, len(levels)) or not np.isfinite(quantiles).all():
        raise ValueError("Quantile evaluation shape mismatch")
    error = actual[:, :, None] - quantiles
    pinball = np.maximum(levels * error, (levels - 1) * error)
    return {"mean_pinball_loss": float(pinball.mean()),
            "pinball_by_level": {str(level): float(pinball[:, :, i].mean()) for i, level in enumerate(levels)},
            "interval": [float(levels[0]), float(levels[-1])],
            "interval_empirical_coverage": float(np.mean((actual >= quantiles[:, :, 0]) & (actual <= quantiles[:, :, -1])))}
