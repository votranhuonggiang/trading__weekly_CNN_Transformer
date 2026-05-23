# Model Exp 02 Results (From `outputs/predictions/model_exp_02_multiscale_cnn_base_only_predictions`)

## Files Used
- `outputs/predictions/model_exp_02_multiscale_cnn_base_only_predictions/weekly_prediction_scores.csv`
- `outputs/predictions/model_exp_02_multiscale_cnn_base_only_predictions/weekly_top_5_portfolio.csv`
- `outputs/predictions/model_exp_02_multiscale_cnn_base_only_predictions/weekly_top_5_model_exp_02_multiscale_cnn_base_only_full_reset_performance.csv`
- `outputs/predictions/model_exp_02_multiscale_cnn_base_only_predictions/weekly_top_5_model_exp_02_multiscale_cnn_base_only_full_reset_cost_sensitivity.csv`

## Test Window
- Start: `2024-01-01`
- Weeks: `122`

## Base Fee Scenario (`fee_rate = 0.0025`)
- Portfolio total return (test): `0.832393`
- Portfolio Sharpe (test): `1.216773`
- VNINDEX Sharpe (test): `1.329580`
- Average turnover (test): `2.0`
- Average trading cost (test): `0.005`

## Cost Sensitivity
| fee_rate | portfolio_total_return_test | portfolio_sharpe_test | vnindex_sharpe_test | avg_turnover_test | avg_trading_cost_test |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 2.377558 | 2.324329 | 1.329580 | 2.0 | 0.0 |
| 0.0015 | 1.341058 | 1.661129 | 1.329580 | 2.0 | 0.003 |
| 0.0025 | 0.832393 | 1.216773 | 1.329580 | 2.0 | 0.005 |
| 0.0035 | 0.433544 | 0.770628 | 1.329580 | 2.0 | 0.007 |

## Latest Weekly Top-5 Portfolio Snapshot
Rebalance date: `2026-05-15` (next rebalance: `2026-05-20`)

| rank | ticker | portfolio_weight | p_buy | score | next_week_return |
|---:|:---|---:|---:|---:|---:|
| 1 | SJD | 0.2 | 0.371798 | 0.257571 | 0.007143 |
| 2 | PC1 | 0.2 | 0.485739 | 0.235740 | 0.056022 |
| 3 | TLD | 0.2 | 0.338313 | 0.199623 | 0.002381 |
| 4 | ADS | 0.2 | 0.364975 | 0.197352 | -0.020856 |
| 5 | TSA | 0.2 | 0.348976 | 0.186789 | 0.000000 |

## Notes
- This is the selected model candidate from the model phase: `model_exp_02_multiscale_cnn_base_only`.
- Features are `base` only.
- Labels are `v2 30/40/30`.
- Execution remains `top-5`, score `P(Buy) - P(Avoid)`, weekly full reset.
- Compared with the baseline in `results.md`, this candidate improves Sharpe slightly but trails baseline total return at `fee_rate = 0.0025`.
