import numpy as np
import pandas as pd
import pytest
from retail_forecast.metrics import LEVELS, dollar_weights, scales, wrmsse


def metadata():
    return pd.DataFrame({"state_id": ["CA", "TX"], "store_id": ["CA_1", "TX_1"],
                         "cat_id": ["FOODS", "FOODS"], "dept_id": ["FOODS_1", "FOODS_1"],
                         "item_id": ["a", "b"]})


def test_first_sale_jump_is_excluded_from_rmsse_scale():
    np.testing.assert_allclose(scales([[0, 0, 2, 4, 6]]), [4])
    np.testing.assert_allclose(scales([[0, 0, 0, 0, 0]]), [0])


def test_perfect_forecast_and_level_weights():
    training = np.array([[1, 2, 3, 2], [2, 4, 3, 5]], float)
    actual = np.array([[2, 4], [4, 6]], float)
    report, details = wrmsse(metadata(), training, actual, actual, [10, 20])
    assert len(LEVELS) == 12 and report["wrmsse"] == 0
    np.testing.assert_allclose(details.groupby("level").weight_within_level.sum(), np.ones(12))
    report, details = wrmsse(metadata(), training, actual, actual + 1, [10, 20])
    assert report["wrmsse"] > 0
    assert details.wrmsse_contribution.sum() == pytest.approx(report["wrmsse"])


def test_historical_positive_sales_require_price_but_prelaunch_zeros_do_not():
    np.testing.assert_allclose(dollar_weights(np.array([[0., 2.]]), np.array([[np.nan, 3.]])), [6])
    with pytest.raises(ValueError):
        dollar_weights(np.array([[1.]]), np.array([[np.nan]]))


def test_positive_weight_zero_scale_is_explicitly_rejected():
    with pytest.raises(ValueError, match="zero RMSSE scale"):
        wrmsse(metadata(), np.ones((2, 4)), np.ones((2, 2)), np.ones((2, 2)), [1, 1])
