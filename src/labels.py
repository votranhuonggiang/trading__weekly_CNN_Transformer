from __future__ import annotations

import numpy as np
import pandas as pd


def build_rebalance_calendar(feature_df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": pd.to_datetime(feature_df["date"]).drop_duplicates().sort_values()})
    iso = dates["date"].dt.isocalendar()
    dates["iso_year"] = iso["year"]
    dates["iso_week"] = iso["week"]
    calendar = (
        dates.groupby(["iso_year", "iso_week"], as_index=False)["date"]
        .max()
        .rename(columns={"date": "rebalance_date"})
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )
    calendar["next_rebalance_date"] = calendar["rebalance_date"].shift(-1)
    return calendar


def build_weekly_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    calendar = build_rebalance_calendar(df)

    rebal_dates = calendar["rebalance_date"]
    snap = df[df["date"].isin(rebal_dates)].copy()
    snap = snap.merge(calendar, left_on="date", right_on="rebalance_date", how="left")

    history_counts = (
        df.groupby("ticker")["close"]
        .rolling(window=60, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
    )
    df["valid_days_60d"] = history_counts
    snap = snap.merge(
        df[["date", "ticker", "valid_days_60d"]],
        on=["date", "ticker"],
        how="left",
    )

    snap["liquidity_threshold_30pct"] = snap.groupby("rebalance_date")["avg_trading_value_20d"].transform(
        lambda s: s.quantile(0.30)
    )
    snap["eligible"] = (
        (snap["valid_days_60d"] >= 45)
        & snap["avg_trading_value_20d"].notna()
        & (snap["avg_trading_value_20d"] >= snap["liquidity_threshold_30pct"])
        & snap["next_rebalance_date"].notna()
    )

    next_close_lookup = (
        df[["date", "ticker", "close"]]
        .rename(columns={"date": "next_rebalance_date", "close": "next_close"})
    )
    snap = snap.merge(next_close_lookup, on=["ticker", "next_rebalance_date"], how="left")
    snap["next_week_return"] = np.log(snap["next_close"] / snap["close"])
    snap["eligible"] = snap["eligible"] & snap["next_week_return"].notna()

    eligible = snap[snap["eligible"]].copy()
    eligible["return_rank_pct"] = eligible.groupby("rebalance_date")["next_week_return"].rank(
        method="first",
        pct=True,
    )
    eligible["label"] = 0
    eligible.loc[eligible["return_rank_pct"] > 0.70, "label"] = 1
    eligible["label_name"] = eligible["label"].map({0: "NotBuy", 1: "Buy"})

    keep_cols = [
        "rebalance_date",
        "next_rebalance_date",
        "ticker",
        "close",
        "next_close",
        "next_week_return",
        "valid_days_60d",
        "avg_trading_value_20d",
        "liquidity_threshold_30pct",
        "return_rank_pct",
        "label",
        "label_name",
    ]
    return eligible[keep_cols].sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)
