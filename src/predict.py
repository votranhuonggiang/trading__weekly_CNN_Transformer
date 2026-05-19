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
    if probabilities.shape[1] not in (2, 3):
        raise ValueError(f"Expected two-class or three-class probabilities with shape [n, 2|3], got {probabilities.shape}.")

    table = metadata.copy().reset_index(drop=True)
    if probabilities.shape[1] == 2:
        table["p_notbuy"] = probabilities[:, 0]
        table["p_buy"] = probabilities[:, 1]
        table["score"] = table["p_buy"]
        table["predicted_label"] = np.where(table["p_buy"] >= table["p_notbuy"], "Buy", "NotBuy")
    else:
        table["p_avoid"] = probabilities[:, 0]
        table["p_hold"] = probabilities[:, 1]
        table["p_buy"] = probabilities[:, 2]
        # v2 weekly buy-list ranking uses P(Buy) directly.
        table["score"] = table["p_buy"]
        table["predicted_label"] = np.select(
            [
                (table["p_avoid"] >= table["p_hold"]) & (table["p_avoid"] >= table["p_buy"]),
                (table["p_hold"] >= table["p_avoid"]) & (table["p_hold"] >= table["p_buy"]),
            ],
            ["Avoid", "Hold"],
            default="Buy",
        )
    table["rank"] = table.groupby("rebalance_date")["score"].rank(method="first", ascending=False).astype(int)
    return table.sort_values(["rebalance_date", "rank", "ticker"]).reset_index(drop=True)


def build_top_k_portfolio(prediction_table: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    top = prediction_table[prediction_table["rank"] <= top_k].copy()
    top["portfolio_weight"] = 1.0 / top_k
    return top.sort_values(["rebalance_date", "rank", "ticker"]).reset_index(drop=True)


def build_predicted_buy_portfolio(prediction_table: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    candidates = prediction_table[prediction_table["predicted_label"] == "Buy"].copy()
    candidates["buy_rank"] = candidates.groupby("rebalance_date")["score"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    top = candidates[candidates["buy_rank"] <= top_k].copy()
    top["portfolio_weight"] = 1.0 / top_k
    return top.sort_values(["rebalance_date", "buy_rank", "ticker"]).reset_index(drop=True)


def export_prediction_outputs(
    prediction_table: pd.DataFrame,
    output_dir: str | Path,
    top_k: int = 5,
    buy_only: bool = False,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_path = output_dir / "weekly_prediction_scores.csv"
    suffix = f"weekly_top_{top_k}_buy_only_portfolio.csv" if buy_only else f"weekly_top_{top_k}_portfolio.csv"
    topk_path = output_dir / suffix

    prediction_table.to_csv(score_path, index=False)
    if buy_only:
        build_predicted_buy_portfolio(prediction_table, top_k=top_k).to_csv(topk_path, index=False)
    else:
        build_top_k_portfolio(prediction_table, top_k=top_k).to_csv(topk_path, index=False)
    return score_path, topk_path


def build_portfolio_performance(
    top_k_portfolio: pd.DataFrame,
    vnindex_comparison: pd.DataFrame | None = None,
    fee_rate: float = 0.002,
) -> pd.DataFrame:
    portfolio = top_k_portfolio.copy()
    portfolio["rebalance_date"] = pd.to_datetime(portfolio["rebalance_date"])
    portfolio["next_week_simple_return"] = portfolio["next_week_return"]

    weekly_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for rebalance_date, group in portfolio.groupby("rebalance_date", sort=True):
        gross_simple_return = float(np.sum(group["portfolio_weight"] * group["next_week_simple_return"]))
        invested_weight = float(group["portfolio_weight"].sum())
        # v2 full reset every week: sell all + buy all -> turnover ratio 2.0.
        turnover = 2.0 if invested_weight > 0 else 0.0

        trading_cost = fee_rate * turnover
        net_simple_return = (1.0 - trading_cost) * (1.0 + gross_simple_return) - 1.0

        weekly_rows.append(
            {
                "rebalance_date": rebalance_date,
                "portfolio_gross_simple_return": gross_simple_return,
                "invested_weight": invested_weight,
                "cash_weight": float(max(0.0, 1.0 - invested_weight)),
                "portfolio_turnover": float(turnover),
                "trading_cost": float(trading_cost),
                "portfolio_simple_return": float(net_simple_return),
                "portfolio_log_return": float(np.log1p(net_simple_return)),
                "num_holdings": int(len(group)),
            }
        )
    weekly = pd.DataFrame(weekly_rows)
    weekly["portfolio_cumulative"] = (1.0 + weekly["portfolio_simple_return"]).cumprod()
    weekly["split"] = weekly["rebalance_date"].apply(_assign_split_label)

    if vnindex_comparison is not None:
        benchmark = vnindex_comparison.copy()
        benchmark["rebalance_date"] = pd.to_datetime(benchmark["rebalance_date"])
        benchmark["vnindex_simple_return"] = benchmark["vnindex_weekly_return"]
        cols = ["rebalance_date", "vnindex_weekly_return", "vnindex_simple_return", "vnindex_cumulative"]
        weekly = weekly.merge(benchmark[cols], on="rebalance_date", how="left")

    return weekly.sort_values("rebalance_date").reset_index(drop=True)


def _assign_split_label(rebalance_date: pd.Timestamp) -> str:
    if rebalance_date <= pd.Timestamp("2021-12-31"):
        return "train"
    if rebalance_date <= pd.Timestamp("2023-12-31"):
        return "validation"
    return "test"


def build_v2_buy_list(prediction_table: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Build weekly buy-only top-K portfolio for v2 strategy.

    Every week, select top-K stocks by P(Buy) score.
    No position carryover—all held stocks are exited on Friday.

    Args:
        prediction_table: DataFrame with columns:
            - rebalance_date: Friday of the week
            - ticker: stock symbol
            - p_avoid, p_hold, p_buy: predicted probabilities
            - split: train/val/test
        top_k: Number of stocks to buy (default 5)

    Returns:
        DataFrame with columns:
            - rebalance_date
            - ticker
            - rank: rank by p_buy descending
            - p_buy
            - split
    """
    test_preds = prediction_table[prediction_table['split'] == 'test'].copy()

    test_preds['rank'] = (
        test_preds.groupby('rebalance_date')['p_buy']
        .rank(method='first', ascending=False)
    )

    top_list = test_preds[test_preds['rank'] <= top_k].copy()

    return top_list[['rebalance_date', 'ticker', 'rank', 'p_buy', 'split']]


def calculate_v2_weekly_returns(
    top_list: pd.DataFrame,
    metadata: pd.DataFrame,
    vnindex_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculate v2 portfolio weekly returns: buy on entry price,
    sell on exit price (both at week rebalance close for simplicity).

    Args:
        top_list: DataFrame from build_v2_buy_list()
        metadata: Model dataset metadata with next_week_return
        vnindex_data: Optional VNINDEX returns for comparison

    Returns:
        DataFrame with columns:
            - rebalance_date
            - portfolio_simple_return (equal-weighted avg of top-k)
            - portfolio_gross_return (before costs)
            - vnindex_simple_return (if provided)
            - split
    """
    top_list = top_list.copy()
    top_list['rebalance_date'] = pd.to_datetime(top_list['rebalance_date']).dt.normalize()
    metadata = metadata.copy()
    metadata['rebalance_date'] = pd.to_datetime(metadata['rebalance_date']).dt.normalize()

    merged = top_list.merge(
        metadata[['rebalance_date', 'ticker', 'next_week_return', 'split']],
        on=['rebalance_date', 'ticker', 'split'],
        how='left'
    )

    if merged['next_week_return'].isna().all():
        raise ValueError("No matching returns data in metadata for top_list tickers/dates")

    weekly_returns = merged.groupby('rebalance_date').agg({
        'next_week_return': 'mean',
    }).reset_index()

    weekly_returns.rename(columns={'next_week_return': 'portfolio_simple_return'}, inplace=True)

    if vnindex_data is not None:
        vnindex_data = vnindex_data[['rebalance_date', 'vnindex_simple_return']].copy()
        vnindex_data['rebalance_date'] = pd.to_datetime(vnindex_data['rebalance_date']).dt.normalize()
        weekly_returns = weekly_returns.merge(vnindex_data, on='rebalance_date', how='left')

    return weekly_returns


def apply_transaction_costs_v2(
    weekly_returns: pd.DataFrame,
    fee_rate: float = 0.002,
) -> pd.DataFrame:
    """
    Apply transaction costs to v2 returns.

    V2 has full turnover every week (exit all, buy top-k fresh).
    Turnover ratio = (k + k) / k = 2.0 (100% in + 100% out of each slot)

    Args:
        weekly_returns: DataFrame from calculate_v2_weekly_returns()
        fee_rate: Transaction cost as fraction (e.g., 0.002 = 0.2%)

    Returns:
        DataFrame with added columns:
            - turnover_ratio (fixed at 2.0 for v2)
            - trading_cost
            - portfolio_net_return (after costs)
    """
    result = weekly_returns.copy()

    result['turnover_ratio'] = 2.0
    result['trading_cost'] = fee_rate * result['turnover_ratio']
    result['portfolio_net_return'] = (
        (1 + result['portfolio_simple_return']) * (1 - result['trading_cost']) - 1
    )

    return result
