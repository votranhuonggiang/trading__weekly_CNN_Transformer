from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


MAX_SHARD_BYTES = 95 * 1024 * 1024


def clear_colab_artifacts(out_dir: Path) -> None:
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
        for path in out_dir.glob(pattern):
            path.unlink()


def main() -> None:
    source = Path("data/processed/model_dataset_splits.npz")
    if not source.exists():
        raise FileNotFoundError(f"Missing source dataset: {source}")

    out_dir = Path("data/colab")
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_colab_artifacts(out_dir)

    arrays = np.load(source)

    for split in ["train", "val", "test"]:
        x = arrays[f"X_{split}"].astype(np.float16)
        y = arrays[f"y_{split}"]
        np.save(out_dir / f"y_{split}.npy", y)
        shard_rows = max(1, MAX_SHARD_BYTES // x[0].nbytes)
        for part_idx, start in enumerate(range(0, len(x), shard_rows)):
            end = min(len(x), start + shard_rows)
            np.save(out_dir / f"X_{split}_part{part_idx:02d}.npy", x[start:end])

    np.save(out_dir / "feature_mean.npy", arrays["feature_mean"].astype(np.float32))
    np.save(out_dir / "feature_std.npy", arrays["feature_std"].astype(np.float32))
    metadata_path = Path("data/processed/model_dataset_metadata.parquet")
    if metadata_path.exists():
        pd.read_parquet(metadata_path).to_parquet(out_dir / "model_dataset_metadata.parquet", index=False)
    feature_profile_path = Path("data/processed/feature_profile.json")
    if feature_profile_path.exists():
        shutil.copy2(feature_profile_path, out_dir / "feature_profile.json")
    print(f"Prepared Colab shards in {out_dir}")


if __name__ == "__main__":
    main()
