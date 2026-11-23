"""Does machine learning forecast Kazakhstan's inflation better than simple benchmarks,
and does the answer change after 2020?

Target: 12-month CPI inflation h months ahead,
    y(t+h) = 100 * [CPI(t+h) / CPI(t+h-12) - 1],  h in {1, 3, 6, 12}.
Information at the forecast date t: inflation history, the tenge/US dollar exchange rate,
and the Brent oil price, all observed at or before t.

Models (one "direct" model per horizon):
    RW     random walk: forecast = today's 12-month inflation y(t)        (Atkeson-Ohanian)
    AR     linear regression of y(t+h) on y(t) and the last 3 monthly inflation rates
    GB     gradient boosted trees on inflation lags only
    GB+X   gradient boosted trees on inflation lags + exchange rate + oil

Evaluation: expanding window. For each forecast date t from 2010-01, every model is
refit (every 3 months; in between, the latest fit is reused) on the observations whose target y(s+h) was already known at t (s + h <= t), then
forecasts y(t+h). Accuracy is RMSE, reported for 2010-2019 and 2020 onward, relative to RW.
Differences are tested with a Diebold-Mariano test (Newey-West variance, h-1 lags).

    python forecast.py            -> results/*.csv, results/*.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
HORIZONS = (1, 3, 6, 12)
EVAL_START = pd.Timestamp("2010-01-01")
SAMPLE_START = pd.Timestamp("2001-01-01")   # skip the 1990s transition and hyperinflation
BREAK = pd.Timestamp("2020-01-01")
SEED = 0


# ---------------------------------------------------------------- data
def dbnomics(name: str) -> pd.Series:
    j = json.loads((DATA / name).read_text(encoding="utf-8"))
    doc = j["series"]["docs"][0]
    s = pd.Series(doc["value"], index=pd.to_datetime(doc["period"]), dtype="float64")
    return s.replace("NA", np.nan).dropna()


def repair_cpi(cpi: pd.Series) -> pd.Series:
    """The IMF index has a base-year splice in January 2020: it jumps 194.27 -> 380.06,
    a fake +95.6% month. Kazakhstan's Bureau of National Statistics reports January 2020
    inflation of 0.7% ("Inflation for the month, in %", January table in its January 2023
    release, stat.gov.kz/en/industries/economy/prices/publications/38575/). Rebuild the
    index from monthly changes using 0.7% for that month."""
    m = cpi.pct_change()
    m.loc["2020-01-01"] = 0.007
    fixed = cpi.iloc[0] * (1 + m.fillna(0)).cumprod()
    # sanity checks against the same BNS table (January m/m, %): 2019 0.5, 2021 0.6
    for date, bns in (("2019-01-01", 0.5), ("2021-01-01", 0.6)):
        assert abs(100 * fixed.pct_change().loc[date] - bns) < 0.06, date
    big = fixed.pct_change()[fixed.index >= "2001-01-01"].abs() > 0.06
    assert not big.any(), f"suspicious monthly jumps after 2001: {list(fixed.index[big.values])}"
    return fixed


def load() -> pd.DataFrame:
    cpi = repair_cpi(dbnomics("IMF_IFS_M_KZ_PCPI_IX.json"))
    fx = dbnomics("IMF_IFS_M_KZ_ENDA_XDC_USD_RATE.json")
    oil = pd.read_csv(DATA / "brent.csv", parse_dates=["observation_date"]).set_index("observation_date")["POILBREUSDM"]
    df = pd.DataFrame({"cpi": cpi, "fx": fx, "oil": oil}).dropna(subset=["cpi"])
    df = df.ffill()                       # fx/oil: carry the last value if a month is missing
    lf, lo = np.log(df.fx), np.log(df.oil)
    df["pi_m"] = 100 * (df.cpi / df.cpi.shift(1) - 1)    # monthly inflation, %
    df["pi_y"] = 100 * (df.cpi / df.cpi.shift(12) - 1)   # 12-month inflation, % (as officially reported)
    for k in (1, 3, 12):
        df[f"fx_{k}"] = 100 * lf.diff(k)  # tenge depreciation over k months, %
        df[f"oil_{k}"] = 100 * lo.diff(k)
    return df[df.index >= SAMPLE_START]


def features(df: pd.DataFrame, extra: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["pi_y"] = df.pi_y
    for lag in range(12):
        X[f"pi_m_l{lag}"] = df.pi_m.shift(lag)
    if extra:
        for k in (1, 3, 12):
            X[f"fx_{k}"], X[f"oil_{k}"] = df[f"fx_{k}"], df[f"oil_{k}"]
    return X


# ---------------------------------------------------------------- models
def gb():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2,
                                     subsample=0.8, min_samples_leaf=5, random_state=SEED)


def run(df: pd.DataFrame, h: int) -> pd.DataFrame:
    y_fut = df.pi_y.shift(-h)             # target known h months after the forecast date
    X_ar = pd.DataFrame({"pi_y": df.pi_y, "pi_m_l0": df.pi_m,
                         "pi_m_l1": df.pi_m.shift(1), "pi_m_l2": df.pi_m.shift(2)})
    X_gb, X_gbx = features(df, False), features(df, True)
    rows, fits = [], None
    dates = df.index[(df.index >= EVAL_START) & y_fut.notna().values]
    for i, t in enumerate(dates):
        if fits is None or i % 3 == 0:
            pos = df.index.get_loc(t)
            train = df.index[: pos - h + 1]   # s with s + h <= t: target already observed at t
            ok = y_fut.loc[train].notna() & X_gbx.loc[train].notna().all(axis=1)
            tr = train[ok.values]
            fits = (LinearRegression().fit(X_ar.loc[tr], y_fut.loc[tr]),
                    gb().fit(X_gb.loc[tr], y_fut.loc[tr]),
                    gb().fit(X_gbx.loc[tr], y_fut.loc[tr]))
        ar, g, gx = fits
        rows.append({"date": t, "actual": y_fut.loc[t], "RW": df.pi_y.loc[t],
                     "AR": float(ar.predict(X_ar.loc[[t]])[0]),
                     "GB": float(g.predict(X_gb.loc[[t]])[0]),
                     "GB+X": float(gx.predict(X_gbx.loc[[t]])[0])})
    out = pd.DataFrame(rows).set_index("date")
    out.index = out.index + pd.DateOffset(months=h)    # label each forecast by its target month
    out.index.name = "target_month"
    return out


# ---------------------------------------------------------------- evaluation
def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> float:
    """Two-sided Diebold-Mariano p-value for equal squared error (Newey-West, h-1 lags)."""
    d = e_model ** 2 - e_bench ** 2
    n, dbar = len(d), d.mean()
    lr = np.sum((d - dbar) ** 2) / n
    for k in range(1, h):
        lr += 2 * (1 - k / h) * np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n
    if lr <= 0:
        return float("nan")
    stat = dbar / math.sqrt(lr / n)
    return float(math.erfc(abs(stat) / math.sqrt(2)))


def summarize(res: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in res.items():
        for period, m in (("2010-2019", f.index < BREAK), ("2020-2025", f.index >= BREAK)):
            g = f[m]
            e = {k: (g[k] - g.actual).to_numpy() for k in ("RW", "AR", "GB", "GB+X")}
            rmse_rw = float(np.sqrt(np.mean(e["RW"] ** 2)))
            for k in ("RW", "AR", "GB", "GB+X"):
                rmse = float(np.sqrt(np.mean(e[k] ** 2)))
                rows.append(dict(horizon=h, period=period, model=k, n=len(g), rmse=round(rmse, 2),
                                 rel_to_RW=round(rmse / rmse_rw, 2),
                                 dm_p_vs_RW=None if k == "RW" else round(dm_test(e[k], e["RW"], h), 3)))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, res: dict[int, pd.DataFrame]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    f = res[3]
    a.plot(df.pi_y[df.index >= "2009-01-01"], color="0.2", lw=1.6, label="Actual 12-month inflation")
    for k, c in (("RW", "0.6"), ("AR", "#3b78c2"), ("GB+X", "#d03b31")):
        a.plot(f[k], color=c, lw=1.1, label=f"{k} forecast, 3 months ahead")
    a.axvline(BREAK, color="0.5", ls=":")
    a.set_ylabel("%"); a.set_title("Kazakhstan: 12-month CPI inflation and 3-month-ahead forecasts", fontsize=10)
    a.legend(fontsize=7)
    for k, c in (("AR", "#3b78c2"), ("GB", "#999933"), ("GB+X", "#d03b31")):
        r = ((f[k] - f.actual) ** 2).rolling(24).mean() ** 0.5 / ((f.RW - f.actual) ** 2).rolling(24).mean() ** 0.5
        b.plot(r, color=c, label=k)
    b.axhline(1, color="0.4", lw=0.8); b.axvline(BREAK, color="0.5", ls=":")
    b.set_title("Rolling 24-month RMSE relative to random walk (h = 3); below 1 = better", fontsize=10)
    b.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "forecasts_h3.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    print(f"data: {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}, {len(df)} months; "
          f"inflation mean {df.pi_y.mean():.1f}%, max {df.pi_y.max():.1f}% ({df.pi_y.idxmax():%Y-%m})")
    res = {h: run(df, h) for h in HORIZONS}
    for h, f in res.items():
        f.to_csv(OUT / f"forecasts_h{h}.csv", float_format="%.3f")
    table = summarize(res)
    table.to_csv(OUT / "rmse_table.csv", index=False)
    print(table.to_string(index=False))
    plot(df, res)


if __name__ == "__main__":
    main()
