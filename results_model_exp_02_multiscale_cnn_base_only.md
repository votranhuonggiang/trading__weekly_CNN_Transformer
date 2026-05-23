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

## What Changed vs Base Model
This candidate does **not** change labels, features, execution, or the transformer head logic. The only intentional change is the **CNN stem** at the front of the model.

### Baseline CNN-Transformer
The baseline model uses a **single-scale CNN stem** before the transformer:

```python
Conv1d(num_features -> conv_channels, kernel_size=5)
-> GELU
-> Dropout
-> Conv1d(conv_channels -> d_model, kernel_size=5)
-> GELU
-> TransformerEncoder
-> mean pooling
-> MLP head
```

Interpretation:
- the model looks at local time patterns with only **one receptive field**
- that receptive field is controlled mainly by `kernel_size=5`
- every local pattern must be represented through that one scale

### `model_exp_02_multiscale_cnn`
Candidate 02 replaces the single CNN stem with a **multi-scale CNN stem**:

```python
branch 1: Conv1d(num_features -> conv_channels, kernel_size=3)
branch 2: Conv1d(num_features -> conv_channels, kernel_size=5)
branch 3: Conv1d(num_features -> conv_channels, kernel_size=9)

concat(branch_1, branch_2, branch_3)
-> Conv1d(merged_channels -> d_model, kernel_size=1)
-> GELU
-> Dropout
-> Conv1d(d_model -> d_model, kernel_size=3)
-> GELU
-> TransformerEncoder
-> mean pooling
-> MLP head
```

Interpretation:
- the model now looks at local patterns through **three window sizes at once**
- `kernel_size=3` focuses more on short, sharp changes
- `kernel_size=5` stays close to the baseline local scale
- `kernel_size=9` sees broader short-to-medium patterns
- the branch outputs are concatenated, then projected back into `d_model`

### What stays the same
To isolate the CNN change, these parts stay unchanged versus the base model:
- same `base` feature set
- same label definition `Avoid/Hold/Buy = 30/40/30`
- same transformer depth and feedforward width
- same positional encoding
- same `mean pooling`
- same classifier head
- same top-5 weekly full-reset backtest logic
- same score: `P(Buy) - P(Avoid)`

### Why this change can help
Weekly stock selection can depend on several kinds of local patterns at the same time:
- short bursts in return or volume
- 1-2 week continuation patterns
- slightly broader setup structure before breakout or reversal

The baseline single-scale CNN has to compress all of that through one kernel width.  
The multi-scale version gives the model separate paths for:
- short-horizon signals
- medium local patterns
- broader local context

That is the main reason this candidate performed better than the other model experiments in the model phase.

### Practical summary
`model_exp_02_multiscale_cnn_base_only` is best understood as:
- **same base pipeline**
- **same transformer**
- **same backtest**
- only the **front CNN feature extractor** is made richer by using multiple kernel sizes instead of one

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
