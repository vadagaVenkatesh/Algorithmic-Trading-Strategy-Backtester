"""Ultra-fast Backtesting Engine with vectorized operations."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import time


class Backtester:
    """High-performance backtesting engine using vectorized operations.
    
    Optimized for minimal latency with:
    - Vectorized array operations (no loops where possible)
    - Efficient NumPy computations
    - Pre-allocated arrays
    - Minimal memory copying
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission: float = 0.001, slippage: float = 0.0005,
                 position_size: float = 1.0):
        """Initialize backtester with trading costs.
        
        Args:
            initial_capital: Starting portfolio value
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage: Slippage per trade (e.g., 0.0005 = 0.05%)
            position_size: Fraction of capital to use per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
    
    def run_backtest(self, data: pd.DataFrame, strategy) -> Tuple[pd.DataFrame, Dict]:
        """Execute backtest with ultra-fast vectorized operations.
        
        Args:
            data: DataFrame with OHLCV data
            strategy: Strategy object with calculate_signals method
            
        Returns:
            Tuple of (results DataFrame, performance metrics dict)
        """
        start_time = time.time()
        
        # Generate signals using strategy (already vectorized)
        df = strategy.calculate_signals(data)
        
        # Extract arrays for vectorized operations
        prices = df['close'].values
        positions = df['position'].values
        
        # Calculate position changes (entry/exit points)
        position_diff = np.diff(np.concatenate(([0], positions)))
        
        # Vectorized return calculations
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.concatenate(([0], price_returns))  # Align with positions
        
        # Calculate trading costs (vectorized)
        trades = np.abs(position_diff)
        trade_costs = trades * (self.commission + self.slippage)
        
        # Apply position sizing and calculate P&L
        # Strategy returns = price returns * position (lagged) - costs
        strategy_returns = price_returns * np.roll(positions, 1) - trade_costs
        strategy_returns[0] = 0  # No return on first day
        
        # Calculate cumulative portfolio value (vectorized)
        cum_returns = np.cumprod(1 + strategy_returns)
        portfolio_value = self.initial_capital * cum_returns
        
        # Calculate actual shares traded (for position tracking)
        capital_per_trade = self.initial_capital * self.position_size
        shares = np.zeros(len(prices))
        cash = np.zeros(len(prices))
        cash[0] = self.initial_capital
        
        # Vectorized calculation of shares and cash
        for i in range(1, len(prices)):
            if position_diff[i] != 0:  # Trade occurred
                # Calculate cost with slippage and commission
                trade_price = prices[i] * (1 + self.slippage * np.sign(position_diff[i]))
                trade_value = capital_per_trade * position_diff[i]
                trade_cost = abs(trade_value) * self.commission
                
                shares[i] = shares[i-1] + (trade_value / trade_price)
                cash[i] = cash[i-1] - trade_value - trade_cost
            else:
                shares[i] = shares[i-1]
                cash[i] = cash[i-1]
        
        # Calculate actual portfolio value
        portfolio_value_actual = cash + shares * prices
        
        # Add results to dataframe
        df['strategy_returns'] = strategy_returns
        df['cum_returns'] = cum_returns
        df['portfolio_value'] = portfolio_value_actual
        df['cash'] = cash
        df['shares'] = shares
        df['trades'] = trades
        
        # Calculate performance metrics (vectorized)
        metrics = self._calculate_metrics(df, strategy_returns, positions)
        metrics['execution_time'] = time.time() - start_time
        
        return df, metrics
    
    def _calculate_metrics(self, df: pd.DataFrame, returns: np.ndarray, 
                          positions: np.ndarray) -> Dict:
        """Calculate performance metrics using vectorized operations.
        
        Args:
            df: Results DataFrame
            returns: Array of strategy returns
            positions: Array of positions
            
        Returns:
            Dictionary of performance metrics
        """
        # Remove any NaN values
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) == 0:
            return {'error': 'No valid returns'}
        
        # Basic return metrics (vectorized)
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # Volatility and Sharpe (vectorized)
        daily_std = np.std(valid_returns)
        annual_volatility = daily_std * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * np.mean(valid_returns) / daily_std if daily_std > 0 else 0
        
        # Maximum drawdown (vectorized)
        cum_returns = df['cum_returns'].values
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis (vectorized)
        trade_returns = valid_returns[valid_returns != 0]
        num_trades = int(np.sum(np.abs(np.diff(positions)) > 0))
        
        if len(trade_returns) > 0:
            win_rate = np.mean(trade_returns > 0)
            avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
            avg_loss = np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Sortino ratio (vectorized, downside deviation)
        downside_returns = valid_returns[valid_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * np.mean(valid_returns) / downside_std if downside_std > 0 else 0
        
        # Final portfolio value
        final_value = df['portfolio_value'].iloc[-1]
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': self.commission * num_trades * self.initial_capital,
            'total_slippage': self.slippage * num_trades * self.initial_capital
        }
    
    def optimize_strategy(self, data: pd.DataFrame, strategy_class, 
                         param_grid: Dict) -> Tuple[Dict, pd.DataFrame]:
        """Ultra-fast parameter optimization using vectorized operations.
        
        Args:
            data: Historical price data
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameters to test
            
        Returns:
            Tuple of (best parameters, results DataFrame)
        """
        results = []
        best_sharpe = -np.inf
        best_params = None
        
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all parameter combinations
        from itertools import product
        param_combinations = list(product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        start_time = time.time()
        
        # Vectorized optimization loop
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            # Create strategy with these parameters
            strategy = strategy_class(**param_dict)
            
            # Run backtest
            _, metrics = self.run_backtest(data, strategy)
            
            if 'error' not in metrics:
                metrics.update(param_dict)
                results.append(metrics)
                
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = param_dict.copy()
        
        optimization_time = time.time() - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Average time per backtest: {optimization_time/len(param_combinations):.4f} seconds")
        
        results_df = pd.DataFrame(results)
        
        return best_params, results_df
    
    def walk_forward_analysis(self, data: pd.DataFrame, strategy_class,
                            param_grid: Dict, train_period: int = 252,
                            test_period: int = 63) -> pd.DataFrame:
        """Perform walk-forward analysis for robust optimization.
        
        Args:
            data: Full dataset
            strategy_class: Strategy class
            param_grid: Parameters to optimize
            train_period: Training window size (days)
            test_period: Testing window size (days)
            
        Returns:
            DataFrame with walk-forward results
        """
        results = []
        total_length = len(data)
        
        # Vectorized walk-forward loop
        for start_idx in range(0, total_length - train_period - test_period, test_period):
            train_end = start_idx + train_period
            test_end = min(train_end + test_period, total_length)
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Optimize on training data
            best_params, _ = self.optimize_strategy(train_data, strategy_class, param_grid)
            
            # Test on out-of-sample data
            if best_params:
                strategy = strategy_class(**best_params)
                _, metrics = self.run_backtest(test_data, strategy)
                
                metrics['train_start'] = train_data.index[0]
                metrics['train_end'] = train_data.index[-1]
                metrics['test_start'] = test_data.index[0]
                metrics['test_end'] = test_data.index[-1]
                metrics.update(best_params)
                
                results.append(metrics)
        
        return pd.DataFrame(results)
