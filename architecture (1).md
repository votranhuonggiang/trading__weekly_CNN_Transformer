# CNN-Transformer Weekly Stock Selection Architecture for HOSE

## 1. Research Objective

Build a price-volume-based stock selection system for the Vietnamese equity market using daily OHLCV data of HOSE-listed stocks from 2016 to April 2026.

The model should support weekly portfolio management by predicting which stocks are likely to outperform in the following week. The final output of the model is not only a prediction label, but also a stock ranking score used to construct and rebalance a portfolio every week.

Core research question:

> Can a CNN-Transformer model using only historical OHLCV data improve weekly stock selection and portfolio performance in the Vietnamese stock market?

---

## 2. Data Scope

### 2.1 Input Data

Use daily OHLCV data for HOSE-listed stocks from 2016 to April 2026.

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

Optional but useful columns if available:

```text
adjusted_close
exchange
industry
market_cap
```

If adjusted close is not available, use close price, but note this limitation clearly because corporate actions such as dividends, stock splits, and rights issues may affect return calculation.

### 2.2 Frequency

Raw data frequency: daily  
Portfolio decision frequency: weekly  
Rebalancing date: last trading day of each week, usually Friday

---

## 3. Universe Construction

The model should avoid illiquid or inactive stocks.

At each weekly rebalancing date `t`, define the eligible universe using only information available up to date `t`.

A stock is eligible if:

```text
Number of valid trading days in the past 60 trading days >= 45
```

and:

```text
Average daily trading value over the past 20 trading days is above the cross-sectional 30th percentile
```

Trading value is calculated as:

```text
trading_value = close * volume
```

This dynamic liquidity filter avoids using a fixed VND threshold that may become inappropriate across different market periods.

Avoid look-ahead bias. Do not use future liquidity, future market capitalization, or future index membership to define the investment universe.

---

## 4. Feature Engineering

For each ticker and each trading date, generate OHLCV-derived features.

### 4.1 Return Features

```text
log_return_1d = log(close_t / close_t-1)
return_5d = log(close_t / close_t-5)
return_10d = log(close_t / close_t-10)
return_20d = log(close_t / close_t-20)
return_60d = log(close_t / close_t-60)
intraday_return = close_t / open_t - 1
high_low_range = high_t / low_t - 1
close_to_high = close_t / high_t - 1
close_to_low = close_t / low_t - 1
```

### 4.2 Volatility Features

```text
volatility_5d = rolling_std(log_return_1d, 5)
volatility_20d = rolling_std(log_return_1d, 20)
volatility_60d = rolling_std(log_return_1d, 60)
downside_volatility_20d = rolling_std(min(log_return_1d, 0), 20)
```

### 4.3 Volume and Liquidity Features

```text
volume_change_1d = log(volume_t / volume_t-1)
avg_volume_5d = rolling_mean(volume, 5)
avg_volume_20d = rolling_mean(volume, 20)
volume_spike = avg_volume_5d / avg_volume_20d
trading_value = close * volume
avg_trading_value_20d = rolling_mean(trading_value, 20)
amihud_20d = rolling_mean(abs(log_return_1d) / trading_value, 20)
```

Handle zero volume or zero trading value safely. Replace infinite values with missing values and impute or drop them later.

### 4.4 Technical Indicator Features

```text
ma_5 = rolling_mean(close, 5)
ma_20 = rolling_mean(close, 20)
ma_60 = rolling_mean(close, 60)
ma_gap_5 = close / ma_5 - 1
ma_gap_20 = close / ma_20 - 1
ma_gap_60 = close / ma_60 - 1
rsi_14 = RSI(close, 14)
```

Optional indicators:

```text
macd
bollinger_band_position
stochastic_oscillator
atr_14
```

For the first implementation, prioritize clean and interpretable features rather than too many technical indicators.

---

## 5. Weekly Target Construction

The task should be designed as a stock selection problem, not an exact price forecasting problem.

At each weekly rebalancing date `t`, calculate each stock's next-week return:

```text
next_week_return_i_t = log(close_i_next_rebalance / close_i_current_rebalance)
```

Then assign labels cross-sectionally within the eligible universe of that week.

Recommended 3-class target:

```text
Buy   = stocks in the top 30% of next-week returns
Hold  = stocks in the middle 40% of next-week returns
Avoid = stocks in the bottom 30% of next-week returns
```

Numerical label mapping:

```text
Buy   = 2
Hold  = 1
Avoid = 0
```

This design is better than simple positive/negative classification because portfolio management depends on relative stock ranking.

---

## 6. Dataset Construction

For each stock-week observation, construct a rolling input window using the past `T` trading days before or up to the rebalance date.

Recommended lookback window:

```text
T = 60 trading days
```

Each sample should have the following shape:

```text
X_i_t: [60, F]
y_i_t: class label for next-week relative return
```

Where:

```text
60 = historical trading days
F = number of engineered OHLCV features
```

Example:

```text
Ticker: FPT
Rebalance date: 2021-06-25
Input window: past 60 trading days up to 2021-06-25
Target: FPT's next-week return rank among eligible HOSE stocks
Label: Buy, Hold, or Avoid
```

Important rules:

1. Do not randomly split samples.
2. Use only past information to construct features.
3. Use next-week return only for label construction.
4. Remove observations with insufficient history.
5. Standardize features using training-period statistics only.

---

## 7. Train, Validation, and Test Split

Use a time-based split.

Recommended split:

```text
Training:   2016-01-01 to 2021-12-31
Validation: 2022-01-01 to 2023-12-31
Testing:    2024-01-01 to 2026-04-30
```

Do not use random train-test splitting because this causes look-ahead bias in financial time series.

Optional advanced version:

Use walk-forward validation after the first implementation is stable.

---

## 8. Model Architecture

The model is inspired by the CNN-Transformer time series forecasting approach.

### 8.1 Architecture Overview

```text
Input: [batch_size, 60, F]
        ↓
1D CNN temporal feature extractor
        ↓
Token embeddings
        ↓
Positional embeddings
        ↓
Transformer encoder
        ↓
Pooling layer
        ↓
MLP classification head
        ↓
Output: P(Avoid), P(Hold), P(Buy)
```

### 8.2 CNN Block

Purpose:

Capture short-term local price-volume patterns such as short-term momentum, reversal, volatility spikes, and volume surges.

Suggested configuration:

```text
Conv1D input channels = F
Conv1D output channels = 64 or 128
Kernel size = 5 or 10
Stride = 1 or 2
Activation = ReLU or GELU
Dropout = 0.1 to 0.3
```

The CNN should operate along the time dimension.

### 8.3 Positional Embedding

Add learnable or sinusoidal positional embeddings to the token sequence before passing it to the Transformer.

Purpose:

Allow the Transformer to understand temporal order.

### 8.4 Transformer Encoder

Purpose:

Capture longer dependencies across the 60-day historical window.

Suggested configuration:

```text
Embedding dimension = 128
Number of encoder layers = 2 to 4
Number of attention heads = 4
Feedforward dimension = 256 or 512
Dropout = 0.1 to 0.3
```

### 8.5 Pooling Layer

Use one of the following:

```text
Mean pooling across time tokens
CLS token pooling
Last-token pooling
```

Recommended first version:

```text
Mean pooling
```

### 8.6 MLP Head

The MLP head maps the final time-series representation to class probabilities.

Suggested structure:

```text
Linear(embedding_dim, 128)
ReLU or GELU
Dropout(0.2)
Linear(128, 3)
Softmax
```

Training should use cross-entropy loss.

---

## 9. Model Output and Stock Ranking Score

The model outputs three probabilities:

```text
P(Avoid), P(Hold), P(Buy)
```

Define the stock selection score as:

```text
score_i_t = P(Buy)_i_t - P(Avoid)_i_t
```

Higher score means the model believes the stock has a higher chance of being in the top next-week return group and a lower chance of being in the bottom group.

At each weekly rebalance date, rank all eligible stocks by this score.

---

## 10. Portfolio Construction

### 10.1 Main Strategy

At each weekly rebalance date:

```text
1. Generate model scores for all eligible HOSE stocks.
2. Rank stocks by score_i_t = P(Buy) - P(Avoid).
3. Select the top K stocks.
4. Allocate equal weight to selected stocks.
5. Hold for one week.
6. Rebalance at the next weekly rebalance date.
```

Recommended first setting:

```text
K = 20
weight_i = 1 / K
```

### 10.2 Portfolio Return

Gross portfolio return:

```text
portfolio_return_t+1 = sum(weight_i_t * next_week_return_i_t+1)
```

### 10.3 Transaction Cost Adjustment

Turnover:

```text
turnover_t = sum(abs(weight_i_t - weight_i_t_minus_1))
```

Net return:

```text
net_portfolio_return_t = gross_portfolio_return_t - transaction_cost_rate * turnover_t
```

Test transaction cost assumptions:

```text
0.00%
0.15%
0.25%
0.35%
```

---

## 11. Baseline Strategies

The model must be compared against both market and rule-based benchmarks.

Recommended baselines:

```text
1. VNINDEX buy-and-hold
2. Equal-weighted eligible HOSE universe
3. Momentum top 20 strategy based on past 20-day return
4. Low-volatility top 20 strategy based on past 20-day volatility
5. XGBoost classifier or ranker using the same features
6. CNN-only model
7. Transformer-only model
8. Random top 20 strategy, repeated multiple times
```

The most important baselines for the first version are:

```text
VNINDEX
Equal-weighted HOSE universe
Momentum top 20
XGBoost
CNN-only
Transformer-only
```

---

## 12. Evaluation Metrics

Evaluate the system at two levels:

1. Prediction performance
2. Portfolio performance

### 12.1 Prediction Metrics

```text
Accuracy
Macro F1-score
Buy-class precision
Buy-class recall
Confusion matrix
Top-K hit ratio
Rank IC
```

Top-K hit ratio:

```text
Among the selected top K stocks, what percentage actually produce positive next-week returns or outperform the market?
```

Rank IC:

```text
Spearman correlation between predicted stock scores and realized next-week returns within each week.
```

Then report the average Rank IC across all test weeks.

### 12.2 Portfolio Metrics

```text
Cumulative return
Annualized return
Annualized volatility
Sharpe ratio
Maximum drawdown
Calmar ratio
Win rate
Average weekly return
Turnover
Transaction-cost-adjusted return
Information ratio against VNINDEX
```

Annualization for weekly returns:

```text
annualized_return = (1 + average_weekly_return) ^ 52 - 1
annualized_volatility = weekly_return_std * sqrt(52)
sharpe_ratio = annualized_return / annualized_volatility
```

If using risk-free rate is not available, report Sharpe ratio with zero risk-free rate and clearly state this assumption.

---

## 13. Implementation Steps

### Step 1: Load and clean OHLCV data

Tasks:

```text
- Parse date column.
- Sort by ticker and date.
- Remove duplicated ticker-date rows.
- Handle missing OHLCV values.
- Ensure close, open, high, low, and volume are numeric.
```

### Step 2: Engineer daily features

Generate return, volatility, volume, liquidity, and technical indicator features for each ticker.

### Step 3: Create weekly rebalance calendar

Use the last available trading day of each week as the rebalance date.

### Step 4: Build weekly eligible universe

Apply liquidity and valid-history filters using only past information.

### Step 5: Construct labels

For each rebalance date:

```text
- Calculate next-week return for each eligible stock.
- Rank stocks cross-sectionally by next-week return.
- Assign Buy, Hold, and Avoid labels.
```

### Step 6: Construct rolling windows

For each ticker and rebalance date:

```text
- Extract past 60 trading days of features.
- Store input tensor X.
- Store target label y.
```

### Step 7: Split by time

Use the defined training, validation, and test periods.

### Step 8: Standardize features

Fit scaler only on training data.

Apply the same scaler to validation and test sets.

### Step 9: Train CNN-Transformer model

Use:

```text
Loss: CrossEntropyLoss
Optimizer: AdamW
Batch size: 64 or 128
Learning rate: 1e-4 to 1e-3
Early stopping based on validation loss or validation Rank IC
Epochs: maximum 100
```

### Step 10: Generate weekly predictions

For each test week:

```text
- Predict probabilities for all eligible stocks.
- Calculate score = P(Buy) - P(Avoid).
- Rank stocks by score.
```

### Step 11: Backtest portfolio

Construct top-K weekly portfolio and calculate gross and net returns.

### Step 12: Compare with baselines

Run the same backtesting logic for benchmark strategies.

### Step 13: Report results

Report:

```text
- Prediction metrics
- Portfolio metrics
- Cumulative return plot
- Drawdown plot
- Weekly return distribution
- Turnover analysis
- Transaction cost sensitivity
```

---

## 14. Important Bias Controls

The implementation must explicitly prevent the following biases.

### 14.1 Look-Ahead Bias

Do not use future information in features, universe construction, scaling, or portfolio weights.

### 14.2 Survivorship Bias

Do not include only currently listed stocks if historical delisted/suspended stocks are available. If unavailable, state survivorship bias as a limitation.

### 14.3 Liquidity Bias

Use dynamic liquidity filters. Do not assume all stocks can be traded equally.

### 14.4 Corporate Action Bias

Prefer adjusted prices. If unavailable, state that unadjusted prices may distort returns around corporate actions.

### 14.5 Data Snooping

Keep the test set untouched until final evaluation.

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
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── backtest.py
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
│   ├── predictions/
│   ├── portfolios/
│   ├── figures/
│   └── tables/
│
├── architecture.md
└── README.md
```

---

## 16. Expected Final Outputs

The implementation should produce:

```text
1. Cleaned daily feature dataset
2. Weekly label dataset
3. Train, validation, and test tensors
4. Trained CNN-Transformer model
5. Weekly stock prediction scores
6. Weekly top-K portfolio holdings
7. Gross and net portfolio returns
8. Benchmark strategy returns
9. Performance summary table
10. Cumulative return and drawdown charts
```

---

## 17. Recommended First Version

Use this configuration for the first working prototype:

```text
Data: Daily OHLCV of HOSE stocks, 2016-April 2026
Frequency: Weekly rebalancing
Rebalance day: Last trading day of each week
Lookback window: 60 trading days
Universe filter: valid trading days >= 45 in past 60 days and avg trading value 20d above weekly 30th percentile
Target: Top 30% = Buy, middle 40% = Hold, bottom 30% = Avoid
Model: CNN + Transformer + MLP
Score: P(Buy) - P(Avoid)
Portfolio: Top 20 stocks, equal-weighted
Training: 2016-2021
Validation: 2022-2023
Testing: 2024-April 2026
Costs: 0%, 0.15%, 0.25%, 0.35%
Main benchmarks: VNINDEX, equal-weighted HOSE universe, momentum top 20, XGBoost, CNN-only, Transformer-only
```

---

## 18. Research Positioning

This project extends CNN-Transformer-based financial time series forecasting from short-horizon price direction prediction to weekly stock selection and portfolio construction in an emerging market.

Unlike the original intraday forecasting setup, this implementation focuses on whether model predictions can be converted into a practical weekly rebalanced portfolio. The final evaluation should therefore emphasize not only classification accuracy, but also risk-adjusted portfolio performance after transaction costs.

The main contribution is a practical deep-learning stock selection framework for HOSE-listed stocks using only OHLCV data.
