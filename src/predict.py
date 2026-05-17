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
    table["p_avoid"] = probabilities[:, 0]
    table["p_hold"] = probabilities[:, 1]
    table["p_buy"] = probabilities[:, 2]
    table["score"] = table["p_buy"] - table["p_avoid"]
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


def build_portfolio_performance(
    top_k_portfolio: pd.DataFrame,
    vnindex_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    portfolio = top_k_portfolio.copy()
    portfolio["rebalance_date"] = pd.to_datetime(portfolio["rebalance_date"])
    portfolio["next_week_simple_return"] = np.exp(portfolio["next_week_return"]) - 1.0

    weekly = (
        portfolio.groupby("rebalance_date", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "portfolio_simple_return": float(
                        np.sum(g["portfolio_weight"] * g["next_week_simple_return"])
                    ),
                    "portfolio_log_return": float(np.log1p(np.sum(g["portfolio_weight"] * g["next_week_simple_return"]))),
                    "num_holdings": int(len(g)),
                }
            )
        )
        .reset_index(drop=True)
    )
    weekly["portfolio_cumulative"] = (1.0 + weekly["portfolio_simple_return"]).cumprod()
    weekly["split"] = weekly["rebalance_date"].apply(_assign_split_label)

    if vnindex_comparison is not None:
        benchmark = vnindex_comparison.copy()
        benchmark["rebalance_date"] = pd.to_datetime(benchmark["rebalance_date"])
        benchmark["vnindex_simple_return"] = np.exp(benchmark["vnindex_weekly_return"]) - 1.0
        cols = ["rebalance_date", "vnindex_weekly_return", "vnindex_simple_return", "vnindex_cumulative"]
        weekly = weekly.merge(benchmark[cols], on="rebalance_date", how="left")

    return weekly.sort_values("rebalance_date").reset_index(drop=True)


def _assign_split_label(rebalance_date: pd.Timestamp) -> str:
    if rebalance_date <= pd.Timestamp("2021-12-31"):
        return "train"
    if rebalance_date <= pd.Timestamp("2023-12-31"):
        return "validation"
    return "test"
