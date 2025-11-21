from types import SimpleNamespace
import numpy as np
from retail_forecast.model import recursive_forecast


class LagBuilder:
    def day(self, t, history):
        assert np.isnan(history[:, t:]).all()
        return history[:, t - 1, None]


class IncreasingModel:
    def predict(self, features):
        return features[:, 0] + 1


def test_future_actuals_never_enter_recursive_lags():
    original = np.tile(np.arange(60, dtype=np.float32), (2, 1))
    data = SimpleNamespace(metadata=["a", "b"], sales=original.copy())
    first = recursive_forecast(IncreasingModel(), data, LagBuilder(), 32, 28)
    data.sales[:, 32:] = 999999
    second = recursive_forecast(IncreasingModel(), data, LagBuilder(), 32, 28)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.arange(32, 60))
