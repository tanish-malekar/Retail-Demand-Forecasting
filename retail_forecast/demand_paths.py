import numpy as np
from scipy.stats import norm, rankdata


def historical_copula_rho(history, window=84):
    history = np.asarray(history, float)
    if history.ndim != 1 or not np.isfinite(history).all() or np.any(history < 0):
        raise ValueError("Historical demand must be a finite nonnegative vector")
    begin = max(7, len(history) - window)
    residual = np.array([history[t] - history[t - 7:t].mean() for t in range(begin, len(history))])
    if len(residual) < 4 or np.std(residual) < 1e-10:
        return 0.0
    z = norm.ppf((rankdata(residual) - .5) / len(residual))
    if np.std(z[:-1]) < 1e-10 or np.std(z[1:]) < 1e-10:
        return 0.0
    rho = .5 * np.corrcoef(z[:-1], z[1:])[0, 1]
    return float(np.clip(rho, -.7, .9))


def sample_paths(quantiles, levels, count, rho, rng):
    values, levels = np.asarray(quantiles, float), np.asarray(levels, float)
    if levels.ndim != 1 or len(levels) < 2 or not np.isfinite(levels).all() or values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(levels) or count < 1 or not -.99 < rho < .99 or not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values, axis=1) < 0) or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("Invalid ordered quantile marginals or path settings")
    horizon = len(values)
    z = np.empty((count, horizon))
    z[:, 0] = rng.normal(size=count)
    for t in range(1, horizon):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho * rho) * rng.normal(size=count)
    uniforms = norm.cdf(z)
    paths = np.empty_like(z, dtype=np.float32)
    grid = np.r_[0., levels, 1.]
    for t in range(horizon):
        # Explicit bounded-tail assumption rather than unbounded extrapolation.
        tail = values[t, -1] + 2 * max(0, values[t, -1] - values[t, -2])
        curve = np.r_[0., values[t], tail]
        sampled = np.interp(uniforms[:, t], grid, curve)
        floor = np.floor(sampled)
        paths[:, t] = floor + (rng.uniform(size=count) < sampled - floor)
    return paths
