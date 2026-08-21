# TailDiff Authentic Historical Market Datasets

This directory contains clean, verified, and authoritative 10-year historical daily OHLCV market datasets (2015-01-05 to 2026-02-27):

1. **`csi300_daily.csv`**:
   - **Target**: CSI 300 Index (沪深300, 000300.SS / sh000300)
   - **Trading Days**: 2708 days
   - **Date Range**: 2015-01-05 to 2026-02-27
   - **Columns**: `Date,Open,High,Low,Close,Volume`
   - **Data Quality**: 100% verified, 0 missing values, perfect K-line OHLC geometry.

2. **`sp500_daily.csv`**:
   - **Target**: S&P 500 Index (^GSPC / .INX)
   - **Trading Days**: 2806 days
   - **Date Range**: 2015-01-02 to 2026-02-27
   - **Columns**: `Date,Open,High,Low,Close,Volume`
   - **Data Quality**: 100% verified, 0 missing values, perfect K-line OHLC geometry.

### Machine Learning & Risk Modeling Specs:
- **Lookback Trajectory Window L**: 32 trading days
- **Forward Risk Horizon H**: 10 trading days
- **Tail Crisis Labeling**: Forward Drawdown at Risk (DaR 97.5% quantile, positive crash rate ≈ 2.5% ~ 5.0%)
- **Conditioning Vector c**: 20-day Realized Volatility ($RV_{20}$), Microstructure Illiquidity, Drawdown Depth
