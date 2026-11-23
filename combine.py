"""Forecast combination: does a simple average of the ML and AR forecasts beat both?

Motivated by McWilliams, Stewart and Isengildina Massa (Food Policy, 2026), who find that
their random-forest model and a SARIMA benchmark have complementary strengths for U.S.
food price inflation, and that a simple average of the two improves accuracy.

Reads the forecasts saved by forecast.py and reports RMSE relative to the random walk for
AR, GB and AVG = (AR + GB) / 2, before and after 2020.

    python forecast.py     # once, to create results/forecasts_h*.csv
    python combine.py      -> results/combination_table.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BREAK = pd.Timestamp("2020-01-01")


def rel_rmse(g: pd.DataFrame, k: str) -> float:
    rmse = lambda col: np.sqrt(np.mean((g[col] - g.actual) ** 2))
    return round(float(rmse(k) / rmse("RW")), 2)


rows = []
for h in (1, 3, 6, 12):
    f = pd.read_csv(ROOT / "results" / f"forecasts_h{h}.csv", index_col=0, parse_dates=True)
    f["AVG"] = (f.AR + f.GB) / 2
    for period, m in (("2010-2019", f.index < BREAK), ("2020-2025", f.index >= BREAK)):
        g = f[m]
        rows.append({"horizon": h, "period": period,
                     **{k: rel_rmse(g, k) for k in ("AR", "GB", "AVG")}})

table = pd.DataFrame(rows)
table.to_csv(ROOT / "results" / "combination_table.csv", index=False)
print("RMSE relative to random walk (below 1 = better)")
print(table.to_string(index=False))
