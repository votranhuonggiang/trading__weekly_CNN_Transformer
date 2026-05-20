# Buy-Only Results (From `outputs/predictions/only_buy`)

## Files Used
- `outputs/predictions/only_buy/weekly_prediction_scores.csv`
- `outputs/predictions/only_buy/weekly_top_5_portfolio.csv`
- `outputs/predictions/only_buy/weekly_top_5_v2_full_reset_performance.csv`
- `outputs/predictions/only_buy/weekly_top_5_v2_full_reset_cost_sensitivity.csv`

## Test Window
- Start: `2024-01-01`
- Weeks: `122`

## Transaction-Cost Sensitivity (Top-5 Weekly Reset)

| fee_rate | portfolio_total_return_test | portfolio_sharpe_test | vnindex_sharpe_test | avg_turnover_test | avg_trading_cost_test |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 1.429971 | 1.642753 | 1.342727 | 0.878689 | 0.000000 |
| 0.0015 | 1.068803 | 1.365884 | 1.342727 | 0.878689 | 0.001318 |
| 0.0025 | 0.858140 | 1.181421 | 1.342727 | 0.878689 | 0.002197 |
| 0.0035 | 0.668766 | 0.997063 | 1.342727 | 0.878689 | 0.003075 |

## Base Fee Scenario (`fee_rate = 0.0025`)
- Portfolio total return (test): `0.858140`
- VNINDEX total return (test): `0.663465`

## Latest Weekly Top-5 Portfolio Snapshot
Rebalance date: `2026-05-15` (next rebalance: `2026-05-19`)

| rank | ticker | portfolio_weight | p_buy | score | next_week_return |
|---:|:---|---:|---:|---:|---:|
| 1 | NVL | 0.2 | 0.484780 | 0.232726 | -0.035298 |
| 2 | FIR | 0.2 | 0.419932 | 0.212832 | -0.045462 |
| 3 | TDH | 0.2 | 0.389631 | 0.180811 | 0.012837 |
| 4 | VRE | 0.2 | 0.434811 | 0.154611 | -0.014815 |
| 5 | HII | 0.2 | 0.442091 | 0.154472 | -0.072817 |

## Notes
- Portfolio holdings are in `weekly_top_5_portfolio.csv` (5 stocks per rebalance week).
- The current `score` column in the exported table is the model ranking score from the pipeline run that generated these files.
