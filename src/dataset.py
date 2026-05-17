from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "intraday_return",
    "high_low_range",
    "close_to_high",
    "close_to_low",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_20d",
    "volume_change_1d",
    "avg_volume_5d",
    "avg_volume_20d",
    "volume_spike",
    "trading_value",
    "avg_trading_value_20d",
    "amihud_20d",
    "ma_5",
    "ma_20",
    "ma_60",
    "ma_gap_5",
    "ma_gap_20",
    "ma_gap_60",
    "rsi_14",
]


@dataclass
class DatasetArtifacts:
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    num_features: int
    lookback_window: int


def _zscore_splits(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean.reshape(-1),
        std.reshape(-1),
    )


def build_model_dataset(
    feature_df: pd.DataFrame,
    weekly_labels: pd.DataFrame,
    lookback_window: int = 60,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, DatasetArtifacts]:
    features = feature_df.copy()
    features["date"] = pd.to_datetime(features["date"])
    labels = weekly_labels.copy()
    labels["rebalance_date"] = pd.to_datetime(labels["rebalance_date"])

    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    feature_map = {
        ticker: ticker_df.reset_index(drop=True)
        for ticker, ticker_df in features.groupby("ticker", sort=False)
    }

    samples: list[np.ndarray] = []
    sample_rows: list[dict[str, object]] = []

    for row in labels.itertuples(index=False):
        ticker_df = feature_map.get(row.ticker)
        if ticker_df is None:
            continue

        date_matches = ticker_df.index[ticker_df["date"] == row.rebalance_date]
        if len(date_matches) == 0:
            continue

        end_idx = int(date_matches[0])
        start_idx = end_idx - lookback_window + 1
        if start_idx < 0:
            continue

        window = ticker_df.iloc[start_idx : end_idx + 1][FEATURE_COLUMNS].copy()
        if window.isna().any().any():
            continue

        samples.append(window.to_numpy(dtype=np.float32))
        sample_rows.append(
            {
                "rebalance_date": row.rebalance_date,
                "next_rebalance_date": row.next_rebalance_date,
                "ticker": row.ticker,
                "label": int(row.label),
                "label_name": row.label_name,
                "next_week_return": float(row.next_week_return),
                "return_rank_pct": float(row.return_rank_pct),
                "split": _assign_split(row.rebalance_date),
            }
        )

    metadata = pd.DataFrame(sample_rows)
    if metadata.empty:
        raise RuntimeError("No model samples were created. Check feature coverage and label alignment.")

    x = np.stack(samples)
    y = metadata["label"].to_numpy(dtype=np.int64)

    train_mask = metadata["split"] == "train"
    val_mask = metadata["split"] == "val"
    test_mask = metadata["split"] == "test"

    train_x, val_x, test_x = x[train_mask], x[val_mask], x[test_mask]
    train_y, val_y, test_y = y[train_mask], y[val_mask], y[test_mask]

    train_x, val_x, test_x, feature_mean, feature_std = _zscore_splits(train_x, val_x, test_x)

    arrays = {
        "X_train": train_x.astype(np.float32),
        "y_train": train_y,
        "X_val": val_x.astype(np.float32),
        "y_val": val_y,
        "X_test": test_x.astype(np.float32),
        "y_test": test_y,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }

    artifacts = DatasetArtifacts(
        total_samples=len(metadata),
        train_samples=int(train_mask.sum()),
        val_samples=int(val_mask.sum()),
        test_samples=int(test_mask.sum()),
        num_features=len(FEATURE_COLUMNS),
        lookback_window=lookback_window,
    )
    return arrays, metadata, artifacts


def _assign_split(rebalance_date: pd.Timestamp) -> str:
    if rebalance_date <= pd.Timestamp("2021-12-31"):
        return "train"
    if rebalance_date <= pd.Timestamp("2023-12-31"):
        return "val"
    return "test"
