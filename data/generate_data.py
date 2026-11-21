"""Generates a synthetic daily sales dataset (3 years) with trend, weekly + yearly
seasonality, festive-season peaks, missing days, duplicate rows and a few outlier spikes.
Run: python data/generate_data.py
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
t = np.arange(len(dates))

trend = 2000 + 2.2 * t
weekly = np.array([0.92, 0.95, 0.97, 1.00, 1.10, 1.25, 1.18])[dates.dayofweek]
yearly = 1 + 0.12 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
festive = np.where(dates.month.isin([10, 11, 12]), 1.18, 1.0)
noise = rng.normal(1, 0.05, len(dates))
sales = trend * weekly * yearly * festive * noise

df = pd.DataFrame({"date": dates, "sales": sales.round(2)})

# make the data messy on purpose (so the cleaning step is real)
df.loc[rng.choice(len(df), 12, replace=False), "sales"] = np.nan
spike_idx = rng.choice(len(df), 5, replace=False)
df.loc[spike_idx, "sales"] = df.loc[spike_idx, "sales"] * 4
df = pd.concat([df, df.sample(6, random_state=1)]).sort_values("date")
df.to_csv("data/sales_data.csv", index=False)
print("saved", df.shape)
