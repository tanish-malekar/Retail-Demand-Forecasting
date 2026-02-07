from dataclasses import dataclass
import numpy as np


@dataclass
class Inventory:
    levels: tuple = (.05, .1, .25, .5, .75, .9, .95)
    paths: int = 10000
    calibration_paths: int = 2000
    lead_days: int = 7
    cycle_days: int = 7
    baseline_service: float = .95
    unit_cost_fraction: float = .6
    cost_tolerance: float = .01
    seed: int = 1729

    def validate(self, horizon):
        if self.paths < 2 or self.calibration_paths < 2 or not 1 <= self.lead_days <= horizon or self.cycle_days < 1 or not .5 < self.baseline_service < 1 or not 0 < self.unit_cost_fraction <= 1 or not 0 <= self.cost_tolerance < 1:
            raise ValueError("Invalid inventory simulation configuration")


def simulate(demand, reorder_point, quantity, lead_days, unit_cost):
    demand = np.asarray(demand, float)
    if demand.ndim != 2 or not np.isfinite(demand).all() or np.any(demand < 0) or quantity < 1 or lead_days < 1 or unit_cost <= 0:
        raise ValueError("Invalid lost-sales simulation inputs")
    paths, horizon = demand.shape
    threshold = np.broadcast_to(np.asarray(reorder_point, float), (paths,))
    if not np.isfinite(threshold).all() or np.any(threshold < 0):
        raise ValueError("Reorder points must be finite and nonnegative")
    on_hand = threshold.copy() + quantity
    pipeline = np.zeros((paths, horizon + lead_days))
    on_order = np.zeros(paths)
    lost = np.zeros(paths); stockouts = np.zeros(paths); capital = np.zeros(paths); ordered = np.zeros(paths)
    for day in range(horizon):
        arrivals = pipeline[:, day]
        on_hand += arrivals; on_order -= arrivals
        shortage = np.maximum(demand[:, day] - on_hand, 0)
        lost += shortage; stockouts += shortage > 0
        on_hand = np.maximum(on_hand - demand[:, day], 0)
        order = (on_hand + on_order <= threshold) * quantity
        pipeline[:, day + lead_days] += order
        on_order += order; ordered += order
        # Inventory capital includes units in transit; no horizon-end free orders.
        capital += (on_hand + on_order) * unit_cost
    return {"lost_units": lost, "stockout_days": stockouts, "capital": capital / horizon,
            "demand_units": demand.sum(axis=1), "ending_on_hand": on_hand,
            "ending_on_order": on_order, "ordered_units": ordered, "initial_units": threshold + quantity}


def allocate_cost_budget(curves, budget):
    """Convexified multiple-choice allocation with a shared mixture probability."""
    if not curves or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Positive inventory budget and candidate curves required")
    for curve in curves:
        if curve.ndim != 2 or curve.shape[1] != 3 or not np.isfinite(curve).all() or np.any(curve < 0):
            raise ValueError("Each curve contains reorder point, capital, and stockout days")
    def choose(penalty):
        indices = np.array([np.argmin(curve[:, 2] + penalty * curve[:, 1]) for curve in curves])
        cost = float(sum(curve[index, 1] for curve, index in zip(curves, indices)))
        return indices, cost
    expensive, expensive_cost = choose(0)
    if expensive_cost < budget:
        # Spend slack on the maximum-capital candidate, preserving a defined common budget.
        cheap, cheap_cost = expensive, expensive_cost
        expensive = np.array([np.argmax(curve[:, 1]) for curve in curves])
        expensive_cost = float(sum(curve[index, 1] for curve, index in zip(curves, expensive)))
    else:
        low, high = 0., 1.
        cheap, cheap_cost = choose(high)
        for _ in range(100):
            if cheap_cost <= budget:
                break
            high *= 2; cheap, cheap_cost = choose(high)
        if cheap_cost > budget:
            raise ValueError("Candidate policies cannot meet the baseline capital budget")
        for _ in range(80):
            middle = (low + high) / 2
            candidate, cost = choose(middle)
            if cost > budget:
                low = middle; expensive, expensive_cost = candidate, cost
            else:
                high = middle; cheap, cheap_cost = candidate, cost
    if expensive_cost < budget - 1e-6 or cheap_cost > budget + 1e-6:
        raise ValueError("Inventory budget is outside the candidate envelope")
    delta = expensive_cost - cheap_cost
    probability = 0. if abs(delta) < 1e-10 else float(np.clip((budget - cheap_cost) / delta, 0, 1))
    return cheap, expensive, probability, {"budget": budget, "calibration_expected_capital":
        (1 - probability) * cheap_cost + probability * expensive_cost, "high_policy_probability": probability}
