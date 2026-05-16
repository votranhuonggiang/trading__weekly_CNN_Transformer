from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def engineer_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_df in df.groupby("ticker", sort=False):
        g = ticker_df.sort_values("date").copy()

        g["trading_value"] = g["close"] * g["volume"]
        g["log_return_1d"] = np.log(g["close"] / g["close"].shift(1))
        g["return_5d"] = np.log(g["close"] / g["close"].shift(5))
        g["return_10d"] = np.log(g["close"] / g["close"].shift(10))
        g["return_20d"] = np.log(g["close"] / g["close"].shift(20))
        g["return_60d"] = np.log(g["close"] / g["close"].shift(60))
        g["intraday_return"] = (g["close"] / g["open"]) - 1.0
        g["high_low_range"] = (g["high"] / g["low"]) - 1.0
        g["close_to_high"] = (g["close"] / g["high"]) - 1.0
        g["close_to_low"] = (g["close"] / g["low"]) - 1.0

        g["volatility_5d"] = g["log_return_1d"].rolling(window=5, min_periods=5).std()
        g["volatility_20d"] = g["log_return_1d"].rolling(window=20, min_periods=20).std()
        g["volatility_60d"] = g["log_return_1d"].rolling(window=60, min_periods=60).std()
        downside_returns = g["log_return_1d"].clip(upper=0.0)
        g["downside_volatility_20d"] = downside_returns.rolling(window=20, min_periods=20).std()

        volume_safe = g["volume"].replace(0, np.nan)
        trading_value_safe = g["trading_value"].replace(0, np.nan)
        g["volume_change_1d"] = np.log(volume_safe / volume_safe.shift(1))
        g["avg_volume_5d"] = g["volume"].rolling(window=5, min_periods=5).mean()
        g["avg_volume_20d"] = g["volume"].rolling(window=20, min_periods=20).mean()
        g["volume_spike"] = g["avg_volume_5d"] / g["avg_volume_20d"]
        g["avg_trading_value_20d"] = g["trading_value"].rolling(window=20, min_periods=20).mean()
        g["amihud_20d"] = (
            g["log_return_1d"].abs() / trading_value_safe
        ).rolling(window=20, min_periods=20).mean()

        g["ma_5"] = g["close"].rolling(window=5, min_periods=5).mean()
        g["ma_20"] = g["close"].rolling(window=20, min_periods=20).mean()
        g["ma_60"] = g["close"].rolling(window=60, min_periods=60).mean()
        g["ma_gap_5"] = (g["close"] / g["ma_5"]) - 1.0
        g["ma_gap_20"] = (g["close"] / g["ma_20"]) - 1.0
        g["ma_gap_60"] = (g["close"] / g["ma_60"]) - 1.0
        g["rsi_14"] = _compute_rsi(g["close"], window=14)

        g.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_frames.append(g)

    feature_df = pd.concat(feature_frames, ignore_index=True)
    feature_df = feature_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return feature_df
