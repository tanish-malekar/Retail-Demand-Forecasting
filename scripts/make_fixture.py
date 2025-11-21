"""Create clearly synthetic M5-shaped inputs for a small development universe."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--series", type=int, default=12)
    parser.add_argument("--days", type=int, default=140)
    args = parser.parse_args()
    if args.series < 1 or args.days < 84:
        raise ValueError("Use positive series count and at least 84 days")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2014-01-01", periods=args.days)
    calendar = pd.DataFrame({"d": [f"d_{i + 1}" for i in range(args.days)], "date": dates,
                             "wm_yr_wk": np.arange(args.days) // 7 + 1})
    calendar["event_name_1"] = ["SyntheticEvent" if i % 17 == 0 else None for i in range(args.days)]
    calendar["event_name_2"] = None
    for shift, state in enumerate(["CA", "TX", "WI"]):
        calendar[f"snap_{state}"] = ((np.arange(args.days) + shift) % 30 < 8).astype(int)
    rows, prices = [], []
    for index in range(args.series):
        state = ["CA", "TX", "WI"][index % 3]
        item, store = f"SYNTH_{index // 3:03d}", f"{state}_1"
        units = rng.poisson(2 + index / 4 + 1.5 * (dates.weekday >= 5)).astype(int)
        units[:min(index, args.days // 3)] = 0
        rows.append({"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": f"SYNTH_{index // 3 % 2}",
                     "cat_id": "SYNTH", "store_id": store, "state_id": state,
                     **{f"d_{day + 1}": int(value) for day, value in enumerate(units)}})
        for week in calendar.wm_yr_wk.unique():
            prices.append({"store_id": store, "item_id": item, "wm_yr_wk": week,
                           "sell_price": round(1 + index / 10 + .05 * np.sin(week), 3)})
    pd.DataFrame(rows).to_csv(args.output / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(args.output / "calendar.csv", index=False)
    pd.DataFrame(prices).to_csv(args.output / "sell_prices.csv", index=False)
    (args.output / "SYNTHETIC.txt").write_text("Synthetic development fixture; not Walmart observations.\n")


if __name__ == "__main__":
    main()
