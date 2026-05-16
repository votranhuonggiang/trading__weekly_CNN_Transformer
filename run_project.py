from __future__ import annotations

from src.pipeline import run_data_ingestion


def main() -> None:
    artifacts = run_data_ingestion()
    print("Data ingestion, feature engineering, weekly labeling, and dataset construction completed.")
    print(f"Universe rows: {artifacts.universe_rows}")
    print(f"Raw OHLCV rows: {artifacts.raw_rows}")
    print(f"Cleaned OHLCV rows: {artifacts.cleaned_rows}")
    print(f"Feature rows: {artifacts.feature_rows}")
    print(f"Rebalance rows: {artifacts.rebalance_rows}")
    print(f"Weekly label rows: {artifacts.label_rows}")
    print(f"Model samples: {artifacts.dataset_samples}")
    print(f"Train samples: {artifacts.train_samples}")
    print(f"Validation samples: {artifacts.val_samples}")
    print(f"Test samples: {artifacts.test_samples}")
    print(f"Benchmark weeks: {artifacts.benchmark_weeks}")


if __name__ == "__main__":
    main()
