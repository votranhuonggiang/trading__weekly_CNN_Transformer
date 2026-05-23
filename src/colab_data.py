from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_sharded_arrays(base_dir: str | Path = "data/colab") -> dict[str, np.ndarray]:
    base_dir = Path(base_dir)
    arrays: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        x_parts = sorted(base_dir.glob(f"X_{split}_part*.npy"))
        y_path = base_dir / f"y_{split}.npy"
        if not x_parts or not y_path.exists():
            raise FileNotFoundError(f"Missing shard files for split '{split}' in {base_dir}")

        arrays[f"X_{split}"] = np.concatenate([np.load(path) for path in x_parts], axis=0).astype(np.float32)
        arrays[f"y_{split}"] = np.load(y_path)

    arrays["feature_mean"] = np.load(base_dir / "feature_mean.npy").astype(np.float32)
    arrays["feature_std"] = np.load(base_dir / "feature_std.npy").astype(np.float32)
    return arrays


def load_feature_profile(base_dir: str | Path = "data/colab") -> dict:
    base_dir = Path(base_dir)
    profile_path = base_dir / "feature_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing feature profile: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
