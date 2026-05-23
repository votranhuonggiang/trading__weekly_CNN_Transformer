from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import build_model_dataset, resolve_feature_columns
from src.labels import build_rebalance_calendar, build_weekly_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild v2 arrays and Colab shards from processed features.")
    parser.add_argument(
        "--feature-packs",
        default="base",
        help="Comma-separated feature pack names from src.dataset.FEATURE_PACK_MAP. Example: base,cross_sectional_liquidity",
    )
    parser.add_argument(
        "--profile-name",
        default="v2_base",
        help="Short profile tag to save into feature profile metadata.",
    )
    return parser.parse_args()


def write_feature_profile(
    processed_dir: Path,
    colab_dir: Path,
    profile_name: str,
    feature_packs: list[str],
    feature_columns: list[str],
) -> None:
    profile = {
        "profile_name": profile_name,
        "label_mode": "v2_buy_top30_hold_mid40_avoid_bottom30",
        "feature_packs": feature_packs,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
    }
    for out_dir in (processed_dir, colab_dir):
        with (out_dir / "feature_profile.json").open("w", encoding="utf-8") as fp:
            json.dump(profile, fp, indent=2)


def clear_colab_artifacts(colab_dir: Path) -> None:
    for pattern in [
        "X_train_part*.npy",
        "X_val_part*.npy",
        "X_test_part*.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "feature_mean.npy",
        "feature_std.npy",
        "model_dataset_metadata.parquet",
        "feature_profile.json",
    ]:
        for path in colab_dir.glob(pattern):
            path.unlink()


def main() -> None:
    args = parse_args()
    processed_dir = REPO_ROOT / "data" / "processed"
    raw_dir = REPO_ROOT / "data" / "raw"
    colab_dir = REPO_ROOT / "data" / "colab"
    figures_dir = REPO_ROOT / "outputs" / "figures"
    tables_dir = REPO_ROOT / "outputs" / "tables"
    feature_packs = [pack.strip() for pack in args.feature_packs.split(",") if pack.strip()]
    feature_columns = resolve_feature_columns(feature_packs)

    feature_path = processed_dir / "daily_features.parquet"
    comparison_path = REPO_ROOT / "outputs" / "tables" / "vnindex_label_comparison.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature dataset: {feature_path}")
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing VNINDEX comparison dataset: {comparison_path}")

    feature_df = pd.read_parquet(feature_path)
    comparison_df = pd.read_csv(comparison_path)

    rebalance_calendar = build_rebalance_calendar(feature_df)
    weekly_labels = build_weekly_labels(feature_df)
    arrays, metadata, artifacts = build_model_dataset(
        feature_df,
        weekly_labels,
        feature_columns=feature_columns,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    colab_dir.mkdir(parents=True, exist_ok=True)
    clear_colab_artifacts(colab_dir)

    rebalance_calendar.to_parquet(processed_dir / "rebalance_calendar.parquet", index=False)
    weekly_labels.to_parquet(processed_dir / "weekly_labels.parquet", index=False)
    metadata.to_parquet(processed_dir / "model_dataset_metadata.parquet", index=False)
    np.savez_compressed(processed_dir / "model_dataset_splits.npz", **arrays)

    max_shard_bytes = 95 * 1024 * 1024
    for split in ["train", "val", "test"]:
        x = arrays[f"X_{split}"].astype(np.float16)
        y = arrays[f"y_{split}"]
        np.save(colab_dir / f"y_{split}.npy", y)
        shard_rows = max(1, max_shard_bytes // x[0].nbytes)
        for part_idx, start in enumerate(range(0, len(x), shard_rows)):
            end = min(len(x), start + shard_rows)
            np.save(colab_dir / f"X_{split}_part{part_idx:02d}.npy", x[start:end])

    np.save(colab_dir / "feature_mean.npy", arrays["feature_mean"].astype(np.float32))
    np.save(colab_dir / "feature_std.npy", arrays["feature_std"].astype(np.float32))
    metadata.to_parquet(colab_dir / "model_dataset_metadata.parquet", index=False)
    write_feature_profile(
        processed_dir=processed_dir,
        colab_dir=colab_dir,
        profile_name=args.profile_name,
        feature_packs=feature_packs,
        feature_columns=feature_columns,
    )

    counts = metadata["label_name"].value_counts().to_dict()
    ratios = metadata["label_name"].value_counts(normalize=True).round(4).to_dict()
    ranges = metadata.groupby("label_name")["return_rank_pct"].agg(["min", "max"]).round(4)

    print("Rebuilt v2 artifacts from processed features.")
    print(f"Feature profile: {args.profile_name}")
    print(f"Feature packs: {feature_packs}")
    print(f"Num features: {len(feature_columns)}")
    print(f"Samples: total={artifacts.total_samples}, train={artifacts.train_samples}, val={artifacts.val_samples}, test={artifacts.test_samples}")
    print(f"Benchmark weeks available: {len(comparison_df)}")
    print(f"Label counts: {counts}")
    print(f"Label ratios: {ratios}")
    print("Return-rank ranges by label:")
    print(ranges.to_string())


if __name__ == "__main__":
    main()
