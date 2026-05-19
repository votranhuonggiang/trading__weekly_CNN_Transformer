# CNN-Transformer Weekly Stock Selection Architecture for HOSE
# Version 2: Weekly Buy List — Full Reset Every Week

> **Upgrade summary**: This version reframes the system as a pure **weekly buy recommendation engine**. Every Friday close, all positions are exited. Every Monday open, the model's top-5 ranked stocks are bought fresh. There is no shorting, no Avoid action, and no carryover of positions. The model still trains on 3 classes for a richer ranking signal, but in production only the **Buy score** matters for selecting the 5 stocks to trade next week.

---

## 1. Research Objective

Build a price-volume-based weekly stock recommendation system for the Vietnamese equity market (HOSE) using daily OHLCV data from 2016 to April 2026.

Every week the system answers one question:

> **Which 5 stocks should I buy at Monday open and sell at Friday close next week?**

The model outputs a ranked list of all eligible HOSE stocks by predicted outperformance probability. The investor acts only on the top 5.

Core research question:

> Can a CNN-Transformer model using only historical OHLCV data generate a weekly top-5 buy list that outperforms the VNINDEX on a risk-adjusted basis after transaction costs?

---

## 2. Data Scope

### 2.1 Input Data

Daily OHLCV data for HOSE-listed stocks from 2016 to April 2026.

Required columns:

```text
date
ticker
open
high
low
close
volume
```

Optional columns if available:

```text
adjusted_close
exchange
industry
market_cap
```

If adjusted close is not available, use close price and note this limitation.

### 2.2 Frequency

```text
Raw data frequency:     daily
Recommendation output:  weekly (every Friday after market close)
Entry:                  Monday open of the following week
Exit:                   Friday close of the following week
Holding period:         Monday open → Friday close (one full trading week)
```

---

## 3. Universe Construction

At each weekly rebalancing date `t` (Friday close), define the eligible universe using only information available up to date `t`.

A stock is eligible if:

```text
Number of valid trading days in the past 60 trading days >= 45
```

AND:

```text
Average daily trading value over the past 20 trading days
is above the cross-sectional 30th percentile of the eligible universe
```

Trading value:

```text
trading_value = close * volume
```

Rules:
- No look-ahead bias. Do not use future liquidity or future index membership.
- Dynamic filter only — no fixed VND threshold.

---

## 4. Feature Engineering

For each ticker and each trading date, generate OHLCV-derived features.

### 4.1 Return Features

```text
log_return_1d   = log(close_t / close_t-1)
return_5d       = log(close_t / close_t-5)
return_10d      = log(close_t / close_t-10)
return_20d      = log(close_t / close_t-20)
return_60d      = log(close_t / close_t-60)
intraday_return = close_t / open_t - 1
high_low_range  = high_t / low_t - 1
close_to_high   = close_t / high_t - 1
close_to_low    = close_t / low_t - 1
```

### 4.2 Volatility Features

```text
volatility_5d           = rolling_std(log_return_1d, 5)
volatility_20d          = rolling_std(log_return_1d, 20)
volatility_60d          = rolling_std(log_return_1d, 60)
downside_volatility_20d = rolling_std(min(log_return_1d, 0), 20)
```

### 4.3 Volume and Liquidity Features

```text
volume_change_1d      = log(volume_t / volume_t-1)
avg_volume_5d         = rolling_mean(volume, 5)
avg_volume_20d        = rolling_mean(volume, 20)
volume_spike          = avg_volume_5d / avg_volume_20d
trading_value         = close * volume
avg_trading_value_20d = rolling_mean(trading_value, 20)
amihud_20d            = rolling_mean(abs(log_return_1d) / trading_value, 20)
```

Handle zero volume or zero trading value safely. Replace infinite values with NaN and impute or drop.

### 4.4 Technical Indicator Features

```text
ma_5      = rolling_mean(close, 5)
ma_20     = rolling_mean(close, 20)
ma_60     = rolling_mean(close, 60)
ma_gap_5  = close / ma_5 - 1
ma_gap_20 = close / ma_20 - 1
ma_gap_60 = close / ma_60 - 1
rsi_14    = RSI(close, 14)
```

---

## 5. Weekly Target Construction

The task is relative stock ranking within the eligible universe, not exact price prediction.

At each Friday rebalance date `t`, calculate each eligible stock's next-week return:

```text
next_week_return_i_t = close_i_{t+1} / close_i_t - 1
```

Where `t+1` is the following Friday close.

Assign cross-sectional labels within the eligible universe of that week:

```text
Buy   = top 30% of next-week returns    → label 2
Hold  = middle 40% of next-week returns → label 1
Avoid = bottom 30% of next-week returns → label 0
```

The model trains on all three classes. This gives a richer gradient signal than binary classification and produces well-calibrated Buy probabilities needed for ranking.

In production (inference), only the Buy probability is used to rank stocks and generate the weekly buy list.

---

## 6. Dataset Construction

For each stock-week observation, construct a rolling input window using the past 60 trading days up to the rebalance date.

```text
Lookback window T = 60 trading days
```

Each sample shape:

```text
X_i_t : [60, F]   — 60 days of engineered features for stock i at week t
y_i_t : {0, 1, 2} — Avoid / Hold / Buy label
```

Rules:
1. No random splitting of samples.
2. Only use information available up to date `t`.
3. Use next-week return only for label assignment, never as a feature.
4. Drop observations with fewer than 60 days of history.
5. Standardize features using training-period statistics only.

---

## 7. Train, Validation, and Test Split

Time-based split only. No random splitting.

```text
Training:   2016-01-01 to 2021-12-31  (287 weeks)
Validation: 2022-01-01 to 2023-12-31  (103 weeks)
Testing:    2024-01-01 to 2026-04-30  (approx 121 weeks)
```

---

## 8. Model Architecture

### 8.1 Overview

```text
Input: [batch_size, 60, F]
        ↓
1D CNN — captures short-term local price-volume patterns
        ↓
Token embeddings + Positional embeddings
        ↓
Transformer encoder — captures long-term dependencies across 60 days
        ↓
Mean pooling
        ↓
MLP classification head
        ↓
Output: P(Avoid), P(Hold), P(Buy)   ← all three used during training
                                        only P(Buy) used for weekly ranking
```

### 8.2 CNN Block

```text
Conv1D input channels  = F (number of features)
Conv1D output channels = 64 or 128
Kernel size            = 5 or 10
Stride                 = 1 or 2
Activation             = GELU
Dropout                = 0.1 to 0.3
```

### 8.3 Positional Embedding

Learnable or sinusoidal positional embeddings added before the Transformer to preserve temporal order.

### 8.4 Transformer Encoder

```text
Embedding dimension      = 128
Number of encoder layers = 2 to 4
Number of attention heads = 4
Feedforward dimension    = 256 or 512
Dropout                  = 0.1 to 0.3
```

### 8.5 MLP Head

```text
Linear(128, 128) → GELU → Dropout(0.2) → Linear(128, 3) → Softmax
```

Training loss: CrossEntropyLoss over all three classes.

---

## 9. Stock Ranking Score

During inference, compute the Buy score for every eligible stock at each Friday rebalance:

```text
score_i_t = P(Buy)_i_t
```

This is simpler than `P(Buy) - P(Avoid)` from v1 because in a buy-only market there is no benefit to modeling the short side. Using raw `P(Buy)` directly ranks stocks by how confidently the model believes they will outperform next week.

Optionally, you can still use `P(Buy) - P(Avoid)` as a sensitivity check — it tends to produce similar rankings but penalizes high-uncertainty stocks more.

At each rebalance date, rank all eligible stocks by `score_i_t` descending.

---

## 10. Weekly Buy List and Portfolio Construction

### 10.1 Weekly Recommendation Output

Every Friday after market close, the system outputs:

```text
Rank | Ticker | Score (P(Buy)) | Suggested Action
-----|--------|----------------|------------------
1    | XXX    | 0.82           | BUY Monday open
2    | YYY    | 0.79           | BUY Monday open
3    | ZZZ    | 0.74           | BUY Monday open
4    | AAA    | 0.71           | BUY Monday open
5    | BBB    | 0.68           | BUY Monday open
```

All other stocks: no action.

### 10.2 Execution Rules

```text
1. Exit all current holdings at Friday close (end of week t).
2. On Monday open of week t+1, buy the top-5 stocks by score.
3. Allocate equal weight: 20% of portfolio per stock.
4. Hold all 5 positions for the full week — no intra-week changes.
5. Repeat at next Friday close.
```

No short selling. No partial positions. No intra-week rebalancing.

### 10.3 Portfolio Return Calculation

Weekly gross return:

```text
portfolio_gross_return_t = (1/5) * sum(simple_return_i for i in top5_t)

simple_return_i_t = close_i_friday_t+1 / close_i_monday_t+1 - 1
```

Note: use Monday open as entry and Friday close as exit for realistic simulation. If Monday open is unavailable, use Friday close as both entry and exit benchmark (less realistic but simpler).

### 10.4 Transaction Cost

Every week the full portfolio is replaced — 100% turnover each week.

```text
turnover_t       = 2.0   (100% sell + 100% buy)
trading_cost_t   = transaction_cost_rate * turnover_t
net_return_t     = gross_return_t - trading_cost_t
```

Test the following cost assumptions:

```text
0.00%   (no cost)
0.15%   (low estimate)
0.25%   (mid estimate)
0.35%   (conservative estimate)
```

At 0.15% one-way cost, the round-trip cost per week is 0.30%. The strategy needs to generate at least 0.30% gross alpha per week to break even after costs.

---

## 11. Baseline Strategies

Compare the model against:

```text
1. VNINDEX buy-and-hold
2. Equal-weighted eligible HOSE universe (weekly reset)
3. Momentum top-5: buy top-5 by past 20-day return, weekly reset
4. Low-volatility top-5: buy lowest-vol 5 stocks, weekly reset
5. XGBoost classifier using same features, same top-5 execution
6. CNN-only model, same top-5 execution
7. Transformer-only model, same top-5 execution
8. Random top-5, repeated 100 times and averaged
```

---

## 12. Evaluation Metrics

### 12.1 Prediction Metrics

```text
Buy-class precision    — of stocks predicted Buy, how many actually outperformed?
Buy-class recall       — of stocks that outperformed, how many did we catch?
Top-5 hit rate         — among weekly top-5 picks, % with positive next-week return
Top-5 beat rate        — % of top-5 picks that beat VNINDEX that week
Rank IC                — Spearman correlation of score vs realized return per week,
                         averaged over all test weeks
```

### 12.2 Portfolio Metrics

```text
Total return (gross and net)
Annualized return
Annualized volatility
Sharpe ratio (zero risk-free rate)
Maximum drawdown
Calmar ratio
Win rate (% of weeks with positive net return)
Average weekly gross return
Average weekly net return
Average weekly trading cost
Information ratio vs VNINDEX
```

Annualization:

```text
annualized_return     = (1 + mean_weekly_net_return) ^ 52 - 1
annualized_volatility = std(weekly_net_return) * sqrt(52)
sharpe_ratio          = annualized_return / annualized_volatility
```

---

## 13. Implementation Steps

### Step 1: Load and clean OHLCV data

```text
- Parse and sort by ticker and date.
- Remove duplicate ticker-date rows.
- Handle missing OHLCV values.
- Ensure all price and volume columns are numeric.
```

### Step 2: Engineer daily features

Generate return, volatility, volume, liquidity, and technical features per ticker per day.

### Step 3: Create weekly rebalance calendar

Use the last available trading day of each week as the rebalance date (Friday close).

### Step 4: Build weekly eligible universe

Apply liquidity and valid-history filters using only past information at each rebalance date.

### Step 5: Construct labels

For each rebalance date:

```text
- Calculate next-week simple return for each eligible stock.
- Rank cross-sectionally.
- Assign Buy (top 30%), Hold (middle 40%), Avoid (bottom 30%).
```

### Step 6: Construct rolling windows

For each ticker and rebalance date:

```text
- Extract past 60 trading days of features → X tensor [60, F]
- Store target label → y scalar {0, 1, 2}
```

### Step 7: Split by time

```text
Training:   2016–2021
Validation: 2022–2023
Testing:    2024–April 2026
```

### Step 8: Standardize features

Fit scaler on training data only. Apply same scaler to validation and test.

### Step 9: Train model

```text
Loss:           CrossEntropyLoss
Optimizer:      AdamW
Batch size:     64 or 128
Learning rate:  1e-4 to 1e-3
Early stopping: based on validation Rank IC or Buy-class F1
Max epochs:     100
```

### Step 10: Generate weekly buy list

```python
# Every Friday after close:
scores = model.predict_proba(X_eligible_t)   # shape: [N, 3]
p_buy  = scores[:, 2]                         # P(Buy) for each stock
ranked = argsort(p_buy, descending=True)
top5   = tickers[ranked[:5]]
weekly_buy_list[t] = top5
```

### Step 11: Backtest

```python
portfolio_value = 1.0
results = []

for t in test_rebalance_dates:
    top5 = weekly_buy_list[t]

    # Compute next-week return for each pick
    # Entry: Monday open t+1, Exit: Friday close t+1
    returns = [next_week_return(ticker, t) for ticker in top5]
    gross   = mean(returns)
    cost    = 2 * transaction_cost_rate        # full round-trip
    net     = gross - cost

    portfolio_value *= (1 + net)
    results.append({
        'week':          t,
        'top5':          top5,
        'gross_return':  gross,
        'net_return':    net,
        'portfolio_value': portfolio_value,
    })
```

### Step 12: Compare with baselines

Run identical backtest for all baseline strategies.

### Step 13: Report results

```text
- Weekly buy list output sample (last 4 weeks of test period)
- Prediction metrics table
- Portfolio metrics table (all strategies side-by-side)
- Cumulative return chart
- Drawdown chart
- Weekly return distribution histogram
- Transaction cost sensitivity table
- Rank IC over time (rolling 12-week average)
```

---

## 14. Important Bias Controls

### 14.1 Look-Ahead Bias

Never use any information from week `t+1` or later when generating the week-`t` buy list.

### 14.2 Survivorship Bias

Include delisted and suspended stocks if available in historical data. If unavailable, state this as a limitation.

### 14.3 Liquidity Bias

Dynamic liquidity filter prevents selecting stocks that cannot realistically be traded.

### 14.4 Corporate Action Bias

Use adjusted prices where available. If unavailable, document the limitation.

### 14.5 Data Snooping

Keep the test set (2024–April 2026) completely untouched until final evaluation. All hyperparameter tuning must be done on validation set only.

### 14.6 Execution Timing

```text
- Friday close scores → Monday open entry (realistic 2-day lag).
- Never use Monday or intra-week data to adjust the buy list mid-week.
- If Monday open is unavailable, use Friday close as entry price and
  clearly state this approximation.
```

---

## 15. Suggested File Structure

```text
project/
│
├── data/
│   ├── raw/
│   │   └── hose_ohlcv_daily.csv
│   ├── processed/
│   │   ├── daily_features.parquet
│   │   ├── weekly_labels.parquet
│   │   └── model_dataset.parquet
│
├── src/
│   ├── config.py          ← K=5, cost rates, date splits
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py         ← outputs weekly ranked buy list
│   ├── backtest.py        ← full weekly reset, top-5, equal weight
│   ├── metrics.py
│   └── baselines.py
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── outputs/
│   ├── weekly_buy_lists/      ← weekly CSV: date, rank, ticker, score
│   ├── portfolios/
│   ├── figures/
│   └── tables/
│
├── architecture_v2.md
└── README.md
```

---

## 16. Expected Final Outputs

```text
1.  Cleaned daily feature dataset
2.  Weekly label dataset
3.  Train, validation, and test tensors
4.  Trained CNN-Transformer model
5.  Weekly prediction scores for all eligible stocks (all splits)
6.  Weekly top-5 buy list — one CSV per rebalance date
7.  Backtest gross and net returns per week
8.  Benchmark strategy returns
9.  Performance summary table (all strategies)
10. Cumulative return and drawdown charts
11. Transaction cost sensitivity table
12. Sample weekly output — what the system would have recommended
```

---

## 17. Recommended Configuration

```text
Data:            Daily OHLCV of HOSE stocks, 2016–April 2026
Frequency:       Weekly recommendation, every Friday after close
Entry:           Monday open of following week
Exit:            Friday close of same week
Lookback window: 60 trading days
Universe filter: >= 45 valid days in past 60, avg 20d value above 30th pct
Labels:          Top 30% = Buy, Middle 40% = Hold, Bottom 30% = Avoid
Model:           CNN + Transformer + MLP (3-class softmax)
Ranking score:   P(Buy)
Portfolio:       Top 5 stocks, equal weight (20% each), full reset weekly
Costs:           0%, 0.15%, 0.25%, 0.35% one-way
Benchmarks:      VNINDEX, equal-weight HOSE, momentum top-5,
                 XGBoost, CNN-only, Transformer-only, random top-5
```

---

## 18. Research Positioning

This system is designed for the Vietnamese retail and institutional investor who can only take long positions. The model does not predict price direction in absolute terms — it identifies which 5 stocks are most likely to outperform the rest of the eligible HOSE universe in the coming week.

The 3-class training target (Buy/Hold/Avoid) is kept because it produces richer gradient signals and better-calibrated probabilities than binary classification. In production, only the Buy probability is needed to generate the ranked weekly list.

The final evaluation focuses on whether this weekly buy list produces consistently positive risk-adjusted returns after realistic transaction costs — the only criterion that matters for a real investor in the Vietnamese market.
