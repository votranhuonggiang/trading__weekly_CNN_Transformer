from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import settings


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


def _resolve_same_day_barrier_hit(
    day_open: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> tuple[int, str]:
    """
    Daily OHLC does not contain intraday timestamps for high/low.
    If both barriers are touched on the same day, fall back to a configurable policy.
    """
    policy = settings.triple_barrier_ambiguous_policy.lower()
    if policy == "take_profit":
        return 2, "take_profit"
    if policy == "stop_loss":
        return 0, "stop_loss"

    if policy == "open_distance":
        tp_distance = abs(take_profit_price - day_open)
        sl_distance = abs(day_open - stop_loss_price)
        if tp_distance < sl_distance:
            return 2, "take_profit"
        if sl_distance < tp_distance:
            return 0, "stop_loss"

    return 1, "vertical_barrier"


def _apply_triple_barrier(
    future_path: pd.DataFrame,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[int, str, pd.Timestamp | pd.NaT, float]:
    take_profit_price = entry_price * (1.0 + take_profit_pct)
    stop_loss_price = entry_price * (1.0 - stop_loss_pct)

    for bar in future_path.itertuples(index=False):
        hit_tp = pd.notna(bar.high) and bar.high >= take_profit_price
        hit_sl = pd.notna(bar.low) and bar.low <= stop_loss_price

        if hit_tp and hit_sl:
            label, event = _resolve_same_day_barrier_hit(bar.open, take_profit_price, stop_loss_price)
            if label == 1:
                return 1, event, bar.date, float(np.log(bar.close / entry_price))
            event_return = take_profit_pct if label == 2 else -stop_loss_pct
            return label, event, bar.date, float(event_return)

        if hit_tp:
            return 2, "take_profit", bar.date, float(take_profit_pct)

        if hit_sl:
            return 0, "stop_loss", bar.date, float(-stop_loss_pct)

    last_bar = future_path.iloc[-1]
    vertical_return = float(np.log(last_bar["close"] / entry_price))
    return 1, "vertical_barrier", pd.Timestamp(last_bar["date"]), vertical_return


def build_weekly_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    calendar = build_rebalance_calendar(df)
    take_profit_pct = settings.triple_barrier_take_profit_pct
    stop_loss_pct = settings.triple_barrier_stop_loss_pct

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
        suffixes=("", "_hist"),
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
    feature_map = {
        ticker: ticker_df[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)
        for ticker, ticker_df in df.groupby("ticker", sort=False)
    }

    labels: list[int] = []
    label_names: list[str] = []
    hit_dates: list[pd.Timestamp | pd.NaT] = []
    barrier_events: list[str] = []
    realized_returns: list[float] = []

    for row in eligible.itertuples(index=False):
        ticker_path = feature_map[row.ticker]
        future_path = ticker_path[
            (ticker_path["date"] > row.rebalance_date) & (ticker_path["date"] <= row.next_rebalance_date)
        ].copy()

        if future_path.empty:
            labels.append(1)
            label_names.append("Timeout")
            hit_dates.append(pd.NaT)
            barrier_events.append("vertical_barrier")
            realized_returns.append(float(row.next_week_return))
            continue

        label, barrier_event, hit_date, barrier_return = _apply_triple_barrier(
            future_path=future_path,
            entry_price=float(row.close),
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
        )
        labels.append(label)
        label_names.append({0: "StopLoss", 1: "Timeout", 2: "TakeProfit"}[label])
        hit_dates.append(hit_date)
        barrier_events.append(barrier_event)
        realized_returns.append(barrier_return)

    eligible["triple_barrier_take_profit_pct"] = take_profit_pct
    eligible["triple_barrier_stop_loss_pct"] = stop_loss_pct
    eligible["triple_barrier_hit_date"] = hit_dates
    eligible["triple_barrier_event"] = barrier_events
    eligible["triple_barrier_return"] = realized_returns
    eligible["label"] = labels
    eligible["label_name"] = label_names

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
        "triple_barrier_take_profit_pct",
        "triple_barrier_stop_loss_pct",
        "triple_barrier_hit_date",
        "triple_barrier_event",
        "triple_barrier_return",
        "label",
        "label_name",
    ]
    return eligible[keep_cols].sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)
