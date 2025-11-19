from dataclasses import dataclass
from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class Training:
    horizon: int = 28
    training_days: int = 365
    boosting_rounds: int = 500
    learning_rate: float = .05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    threads: int = 8
    seed: int = 42
    prophet_workers: int = 4


def train_global(data, builder, cutoff, config, directory):
    directory = Path(directory)
    begin = max(28, cutoff - config.training_days)
    if begin >= cutoff or config.boosting_rounds < 1 or config.training_days < 1 or config.threads < 1 or config.num_leaves < 2 or not 0 < config.learning_rate <= 1:
        raise ValueError("Invalid training configuration")
    rows = (cutoff - begin) * len(data.metadata)
    shape = (rows, len(builder.names))
    features = np.memmap(directory / "training_features.f32", dtype="float32", mode="w+", shape=shape)
    labels = np.memmap(directory / "training_targets.f32", dtype="float32", mode="w+", shape=(rows,))
    n = len(data.metadata)
    for offset, day in enumerate(range(begin, cutoff)):
        selection = slice(offset * n, (offset + 1) * n)
        features[selection] = builder.day(day, data.sales)
        labels[selection] = data.sales[:, day]
    features.flush(); labels.flush()
    params = {"objective": "poisson", "learning_rate": config.learning_rate, "num_leaves": config.num_leaves,
              "min_data_in_leaf": config.min_data_in_leaf, "feature_fraction": .9, "bagging_fraction": .9,
              "bagging_freq": 1, "max_bin": 127, "num_threads": config.threads,
              "seed": config.seed, "feature_fraction_seed": config.seed, "bagging_seed": config.seed,
              "deterministic": True, "force_col_wise": True, "verbosity": -1}
    dataset = lgb.Dataset(features, label=labels, feature_name=list(builder.names), categorical_feature=list(builder.categorical))
    # Fixed tree count: holdout labels do not enter an eval_set or early stopping.
    model = lgb.train(params, dataset, num_boost_round=config.boosting_rounds)
    model.save_model(str(directory / "lightgbm.txt"))
    pd.DataFrame({"feature": builder.names, "gain": model.feature_importance(importance_type="gain")}).to_csv(directory / "feature_importance.csv", index=False)
    spec = {"shape": list(shape), "dtype": "float32", "features": list(builder.names),
            "categorical": list(builder.categorical), "first_training_day": begin + 1,
            "last_training_day": cutoff, "parameters": params, "boosting_rounds": config.boosting_rounds}
    (directory / "training_matrix.json").write_text(json.dumps(spec, indent=2) + "\n")
    return model


def recursive_forecast(model, data, builder, cutoff, horizon):
    if cutoff < 28 or horizon < 1 or cutoff + horizon > data.sales.shape[1]:
        raise ValueError("Invalid forecast split")
    history = np.full((len(data.metadata), cutoff + horizon), np.nan, dtype=np.float32)
    history[:, :cutoff] = data.sales[:, :cutoff]
    predictions = np.empty((len(data.metadata), horizon), dtype=np.float32)
    for step in range(horizon):
        day = cutoff + step
        values = np.asarray(model.predict(builder.day(day, history)), dtype=np.float32)
        if values.shape != (len(data.metadata),) or not np.isfinite(values).all():
            raise ValueError("Model returned invalid predictions")
        predictions[:, step] = np.maximum(values, 0)
        history[:, day] = predictions[:, step]
    return predictions
