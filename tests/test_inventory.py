import numpy as np
import pytest
from retail_forecast.demand_paths import sample_paths
from retail_forecast.inventory import allocate_cost_budget, simulate


def test_lost_sales_and_lead_time_accounting():
    demand = np.array([[2., 0., 2., 0.]])
    result = simulate(demand, 0, 1, 1, 1)
    assert result["lost_units"][0] == 2
    assert result["stockout_days"][0] == 2
    assert result["capital"][0] == 1
    served = result["demand_units"] - result["lost_units"]
    np.testing.assert_allclose(result["initial_units"] + result["ordered_units"],
                               served + result["ending_on_hand"] + result["ending_on_order"])


def test_cost_mixture_matches_a_fixed_budget():
    curves = [np.array([[0., 1., 5.], [1., 2., 2.], [2., 3., 0.]])] * 2
    cheap, expensive, probability, report = allocate_cost_budget(curves, 4.5)
    actual = sum((1 - probability) * curve[a, 1] + probability * curve[b, 1]
                 for curve, a, b in zip(curves, cheap, expensive))
    assert actual == pytest.approx(4.5)
    assert report["calibration_expected_capital"] == pytest.approx(4.5)


def test_zero_quantile_paths_stay_zero_and_seeds_reproduce():
    values = np.zeros((7, 3))
    levels = [.1, .5, .9]
    first = sample_paths(values, levels, 100, .3, np.random.default_rng(12))
    second = sample_paths(values, levels, 100, .3, np.random.default_rng(12))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 7) and np.count_nonzero(first) == 0


def test_quantile_crossing_is_rejected_at_sampling_boundary():
    with pytest.raises(ValueError):
        sample_paths(np.array([[2., 1., 3.]]), [.1, .5, .9], 100, 0, np.random.default_rng(1))
