from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.benchmark import build_vnindex_comparison
from src.data_loader import (
    clean_ohlcv,
    fetch_hose_universe,
    fetch_raw_ohlcv,
    fetch_vnindex,
    save_dataframe,
)
from src.dataset import build_model_dataset
from src.features import engineer_daily_features
from src.labels import build_rebalance_calendar, build_weekly_labels


@dataclass
class PipelineArtifacts:
    universe_rows: int
    raw_rows: int
    cleaned_rows: int
    feature_rows: int
    rebalance_rows: int
    label_rows: int
    dataset_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    benchmark_weeks: int


def run_data_ingestion() -> PipelineArtifacts:
    universe = fetch_hose_universe()
    raw_ohlcv = fetch_raw_ohlcv()
    vnindex = fetch_vnindex()
    cleaned_ohlcv = clean_ohlcv(raw_ohlcv, universe["symbol"])
    daily_features = engineer_daily_features(cleaned_ohlcv)
    rebalance_calendar = build_rebalance_calendar(daily_features)
    weekly_labels = build_weekly_labels(daily_features)
    model_arrays, model_metadata, dataset_artifacts = build_model_dataset(daily_features, weekly_labels)
    benchmark_artifacts = build_vnindex_comparison(rebalance_calendar, weekly_labels, vnindex)

    save_dataframe(universe, settings.data_raw_dir / "hose_universe.csv")
    save_dataframe(raw_ohlcv, settings.data_raw_dir / "hose_ohlcv_daily_raw.parquet")
    save_dataframe(cleaned_ohlcv, settings.data_processed_dir / "hose_ohlcv_daily_clean.parquet")
    save_dataframe(daily_features, settings.data_processed_dir / "daily_features.parquet")
    save_dataframe(rebalance_calendar, settings.data_processed_dir / "rebalance_calendar.parquet")
    save_dataframe(weekly_labels, settings.data_processed_dir / "weekly_labels.parquet")
    save_dataframe(model_metadata, settings.data_processed_dir / "model_dataset_metadata.parquet")
    npz_path = settings.data_processed_dir / "model_dataset_splits.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(npz_path, **model_arrays)

    return PipelineArtifacts(
        universe_rows=len(universe),
        raw_rows=len(raw_ohlcv),
        cleaned_rows=len(cleaned_ohlcv),
        feature_rows=len(daily_features),
        rebalance_rows=len(rebalance_calendar),
        label_rows=len(weekly_labels),
        dataset_samples=dataset_artifacts.total_samples,
        train_samples=dataset_artifacts.train_samples,
        val_samples=dataset_artifacts.val_samples,
        test_samples=dataset_artifacts.test_samples,
        benchmark_weeks=benchmark_artifacts.weekly_rows,
    )
