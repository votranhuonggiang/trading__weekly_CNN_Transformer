from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import CNNTransformerClassifier


def predict_probabilities(
    model: CNNTransformerClassifier,
    x: np.ndarray,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    probs_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            logits = model(x_batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def build_weekly_prediction_table(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    if len(metadata) != len(probabilities):
        raise ValueError("Metadata rows and probability rows must match.")

    table = metadata.copy().reset_index(drop=True)
    table["p_stoploss"] = probabilities[:, 0]
    table["p_timeout"] = probabilities[:, 1]
    table["p_takeprofit"] = probabilities[:, 2]
    table["score"] = table["p_takeprofit"] - table["p_stoploss"]
    table["rank"] = table.groupby("rebalance_date")["score"].rank(method="first", ascending=False).astype(int)
    return table.sort_values(["rebalance_date", "rank", "ticker"]).reset_index(drop=True)


def build_top_k_portfolio(prediction_table: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    top = prediction_table[prediction_table["rank"] <= top_k].copy()
    top["portfolio_weight"] = 1.0 / top_k
    return top.sort_values(["rebalance_date", "rank", "ticker"]).reset_index(drop=True)


def export_prediction_outputs(
    prediction_table: pd.DataFrame,
    output_dir: str | Path,
    top_k: int = 5,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_path = output_dir / "weekly_prediction_scores.csv"
    topk_path = output_dir / f"weekly_top_{top_k}_portfolio.csv"

    prediction_table.to_csv(score_path, index=False)
    build_top_k_portfolio(prediction_table, top_k=top_k).to_csv(topk_path, index=False)
    return score_path, topk_path
