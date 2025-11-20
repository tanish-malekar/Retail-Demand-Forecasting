import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

LEVELS = [(), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",),
          ("state_id", "cat_id"), ("state_id", "dept_id"), ("store_id", "cat_id"),
          ("store_id", "dept_id"), ("item_id",), ("item_id", "state_id"), ("item_id", "store_id")]


def aggregation(metadata, keys):
    if not keys:
        codes, labels = np.zeros(len(metadata), dtype=np.int32), ["Total"]
    else:
        index = pd.MultiIndex.from_frame(metadata[list(keys)])
        codes, unique = pd.factorize(index, sort=True)
        labels = ["|".join(map(str, value)) for value in unique.tolist()]
    matrix = csr_matrix((np.ones(len(metadata), dtype=np.float64), (codes, np.arange(len(metadata)))), shape=(len(labels), len(metadata)))
    return matrix, labels


def scales(training):
    training = np.asarray(training, float)
    if training.ndim != 2 or training.shape[1] < 2:
        raise ValueError("At least two historical days required")
    active = training > 0
    first = active.argmax(axis=1)
    differences = np.diff(training, axis=1) ** 2
    mask = np.arange(training.shape[1] - 1)[None, :] >= first[:, None]
    count = training.shape[1] - 1 - first
    numerator = np.sum(np.where(mask, differences, 0), axis=1)
    return np.divide(numerator, count, out=np.zeros(len(training)), where=active.any(axis=1) & (count > 0))


def dollar_weights(training, historic_prices):
    if training.shape != historic_prices.shape:
        raise ValueError("Price and sales shapes differ")
    if np.any((training > 0) & (~np.isfinite(historic_prices) | (historic_prices <= 0))):
        raise ValueError("Every positive unit sale needs a known positive historical price")
    return np.sum(np.where(training > 0, training * np.nan_to_num(historic_prices, nan=0), 0), axis=1)


def wrmsse(metadata, training, actual, predicted, revenue):
    training, actual, predicted, revenue = map(np.asarray, (training, actual, predicted, revenue))
    n = len(metadata)
    if training.ndim != 2 or actual.ndim != 2 or actual.shape != predicted.shape or len(actual) != n or len(training) != n or revenue.shape != (n,) or actual.shape[1] == 0:
        raise ValueError("Metric inputs must align to the same bottom series and horizon")
    if not all(np.isfinite(value).all() for value in (training, actual, predicted, revenue)) or any(np.any(value < 0) for value in (training, actual, predicted, revenue)) or revenue.sum() <= 0:
        raise ValueError("Finite nonnegative observations, predictions, and positive revenue weights required")
    details, level_scores = [], []
    for number, keys in enumerate(LEVELS, 1):
        matrix, labels = aggregation(metadata, keys)
        weights = np.asarray(matrix @ revenue).ravel() / revenue.sum()
        truth, forecast = matrix @ actual, matrix @ predicted
        mse = np.mean((truth - forecast) ** 2, axis=1)
        scale = np.empty(len(labels))
        for begin in range(0, len(labels), 256):
            scale[begin:begin + 256] = scales(matrix[begin:begin + 256] @ training)
        if np.any((weights > 0) & (scale <= 0)):
            raise ValueError(f"Level {number} has positive-weight series with zero RMSSE scale")
        rmsse = np.sqrt(np.divide(mse, scale, out=np.zeros_like(mse), where=scale > 0))
        contributions = weights * rmsse
        level_scores.append(float(contributions.sum()))
        details.extend({"level": number, "keys": ",".join(keys) or "total", "series": label,
                        "weight_within_level": float(weight), "scale": float(denom),
                        "rmsse": float(error), "wrmsse_contribution": float(contribution / 12)}
                       for label, weight, denom, error, contribution in zip(labels, weights, scale, rmsse, contributions))
    return {"wrmsse": float(np.mean(level_scores)), "level_scores": {str(i): score for i, score in enumerate(level_scores, 1)},
            "hierarchy_series": len(details)}, pd.DataFrame(details)
