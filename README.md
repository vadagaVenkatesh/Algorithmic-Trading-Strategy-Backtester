# Algorithmic-Trading-Strategy-Backtester

A comprehensive Python-based backtesting framework for quantitative trading strategies, including statistical arbitrage (pairs trading, mean reversion), momentum-based algorithms, transaction cost modeling, slippage assumptions, and thorough performance validation.

## Project Overview

This project provides a robust backtesting environment for developing and evaluating quantitative trading strategies in Python. It supports:
- Transaction cost and slippage modeling
- Strategy implementation for statistical arbitrage & momentum
- Performance metrics: Sharpe ratio, max drawdown, alpha, beta
- Walk-forward optimization & out-of-sample testing for robustness

## Requirements
- Python >= 3.8
- pandas
- numpy
- matplotlib (for plotting)
- scipy (for advanced stats)

Install dependencies:
```bash
pip install pandas numpy matplotlib scipy
```

## Usage Example

Below is a basic workflow to backtest a mean reversion strategy with transaction costs and performance metrics:

```python
import pandas as pd
import numpy as np

def mean_reversion_strategy(prices, lookback=20, tc=0.001):
    returns = prices.pct_change()
    rolling_mean = prices.rolling(lookback).mean()
    signal = (prices < rolling_mean).astype(int)
    strat_returns = signal.shift(1) * returns - tc * np.abs(signal.diff())
    return strat_returns

data = pd.read_csv('historical_data.csv', index_col='Date', parse_dates=True)
prices = data['Close']
strat_returns = mean_reversion_strategy(prices)

# Performance metrics
sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
max_drawdown = (strat_returns.cumsum() - strat_returns.cumsum().cummax()).min()
print(f"Sharpe Ratio: {sharpe:.2f}, Max Drawdown: {max_drawdown:.2%}")
```

## Advanced Features
- **Pairs Trading**: Identify cointegrated pairs, trade mean reversion spread
- **Momentum**: Use rolling returns to compute momentum signals
- **Walk-Forward Optimization**: Split data, optimize parameters in sample, test out-of-sample

```python
# Example walk-forward split and testing
split = int(len(prices) * 0.7)
train, test = prices[:split], prices[split:]
optimal_lb = max(range(10, 30), key=lambda lb: mean_reversion_strategy(train, lb).mean())
strat_returns_test = mean_reversion_strategy(test, optimal_lb)
print(f"Out-of-Sample Sharpe: {strat_returns_test.mean() / strat_returns_test.std() * np.sqrt(252):.2f}")
```

## Output & Validation
- Metrics: Sharpe ratio, max drawdown, alpha, beta, turnover, slippage
- Plots: Equity curve, drawdown plot, rolling Sharpe ratio

## Extending
Add new strategies by defining signal logic. For advanced transaction cost or market impact modeling, extend the `tc` and slippage components in your function.

## Reference
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Quantitative Trading by Ernest P. Chan]

Feel free to contribute new strategies, optimizers, and performance plots!
