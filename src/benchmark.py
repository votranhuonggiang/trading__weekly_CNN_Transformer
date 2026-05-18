from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class BenchmarkArtifacts:
    weekly_rows: int
    figure_paths: list[str]
    table_paths: list[str]


def build_vnindex_comparison(
    rebalance_calendar: pd.DataFrame,
    weekly_labels: pd.DataFrame,
    vnindex_df: pd.DataFrame,
) -> BenchmarkArtifacts:
    if vnindex_df.empty:
        raise RuntimeError("VNINDEX data is empty. Cannot build benchmark comparison.")

    calendar = rebalance_calendar.copy()
    labels = weekly_labels.copy()
    vnindex = vnindex_df.copy()

    calendar["rebalance_date"] = pd.to_datetime(calendar["rebalance_date"])
    calendar["next_rebalance_date"] = pd.to_datetime(calendar["next_rebalance_date"])
    labels["rebalance_date"] = pd.to_datetime(labels["rebalance_date"])
    vnindex["date"] = pd.to_datetime(vnindex["date"])

    benchmark = calendar[calendar["next_rebalance_date"].notna()].copy()
    benchmark = benchmark.merge(
        vnindex[["date", "close"]].rename(columns={"date": "rebalance_date", "close": "vnindex_close"}),
        on="rebalance_date",
        how="left",
    )
    benchmark = benchmark.merge(
        vnindex[["date", "close"]].rename(columns={"date": "next_rebalance_date", "close": "vnindex_next_close"}),
        on="next_rebalance_date",
        how="left",
    )
    benchmark["vnindex_weekly_return"] = np.log(
        benchmark["vnindex_next_close"] / benchmark["vnindex_close"]
    )
    benchmark["vnindex_cumulative"] = np.exp(benchmark["vnindex_weekly_return"].fillna(0.0).cumsum())

    label_perf = (
        labels.groupby(["rebalance_date", "label_name"], as_index=False)["next_week_return"]
        .mean()
        .pivot(index="rebalance_date", columns="label_name", values="next_week_return")
        .reset_index()
    )

    comparison = benchmark.merge(label_perf, on="rebalance_date", how="left")
    label_cols = [col for col in ["NotBuy", "Buy"] if col in comparison.columns]
    for col in label_cols:
        comparison[f"{col}_cumulative"] = np.exp(comparison[col].fillna(0.0).cumsum())

    settings.outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_figures_dir.mkdir(parents=True, exist_ok=True)

    table_path = settings.outputs_tables_dir / "vnindex_label_comparison.csv"
    comparison.to_csv(table_path, index=False)

    fig1_path = settings.outputs_figures_dir / "vnindex_vs_label_groups.png"
    _plot_cumulative_comparison(comparison, fig1_path)

    fig2_path = settings.outputs_figures_dir / "label_return_distribution.png"
    _plot_label_distribution(labels, fig2_path)

    return BenchmarkArtifacts(
        weekly_rows=len(comparison),
        figure_paths=[str(fig1_path), str(fig2_path)],
        table_paths=[str(table_path)],
    )


def _plot_cumulative_comparison(comparison: pd.DataFrame, output_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        comparison["rebalance_date"],
        comparison["vnindex_cumulative"],
        label="VNINDEX",
        linewidth=2.2,
        color="#1f4e79",
    )

    color_map = {
        "NotBuy_cumulative": "#7f8c8d",
        "Buy_cumulative": "#117a65",
    }
    label_map = {
        "NotBuy_cumulative": "NotBuy Group (look-ahead)",
        "Buy_cumulative": "Buy Group (look-ahead)",
    }

    for col in ["NotBuy_cumulative", "Buy_cumulative"]:
        if col in comparison.columns:
            ax.plot(
                comparison["rebalance_date"],
                comparison[col],
                label=label_map[col],
                linewidth=1.8,
                color=color_map[col],
            )

    ax.set_title("VNINDEX vs Weekly Return Label Group Performance")
    ax.set_ylabel("Cumulative Growth of 1.0")
    ax.set_xlabel("Rebalance Date")
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "Label groups are ex-post and not tradable. Use only for diagnostic analysis.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_label_distribution(labels: pd.DataFrame, output_path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    order = ["NotBuy", "Buy"]
    data = [labels.loc[labels["label_name"] == name, "next_week_return"].dropna().values for name in order]
    ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)

    colors = ["#d5dbdb", "#a3e4d7"]
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)

    ax.set_title("Next-Week Return Distribution by Label")
    ax.set_ylabel("Next-Week Log Return")
    ax.set_xlabel("Label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
