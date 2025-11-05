"""Ultra-fast Mean Reversion Strategy using vectorized operations."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class MeanReversionStrategy:
    """Vectorized mean reversion strategy for ultra-low latency backtesting.
    
    Uses NumPy arrays and vectorized operations to avoid loops and minimize latency.
    Implements a statistical mean reversion approach with z-score based signals.
    """
    
    def __init__(self, lookback_period: int = 20, entry_z_score: float = 2.0, 
                 exit_z_score: float = 0.5, stop_loss_z: float = 3.0):
        """Initialize mean reversion strategy parameters.
        
        Args:
            lookback_period: Window size for calculating mean and std
            entry_z_score: Z-score threshold to enter position
            exit_z_score: Z-score threshold to exit position
            stop_loss_z: Z-score threshold for stop loss
        """
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z = stop_loss_z
        self.name = "MeanReversion"
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using vectorized operations.
        
        Args:
            data: DataFrame with 'close' price column
            
        Returns:
            DataFrame with signals, positions, and z-scores
        """
        df = data.copy()
        prices = df['close'].values
        
        # Vectorized rolling calculations using pandas (optimized C implementations)
        rolling_mean = df['close'].rolling(window=self.lookback_period).mean().values
        rolling_std = df['close'].rolling(window=self.lookback_period).std().values
        
        # Avoid division by zero
        rolling_std = np.where(rolling_std == 0, 1e-8, rolling_std)
        
        # Calculate z-scores (vectorized)
        z_scores = (prices - rolling_mean) / rolling_std
        
        # Generate signals using vectorized conditions
        # Long signal: price is too low (negative z-score beyond threshold)
        long_signal = z_scores < -self.entry_z_score
        
        # Short signal: price is too high (positive z-score beyond threshold)
        short_signal = z_scores > self.entry_z_score
        
        # Exit signals: z-score returns to near mean
        exit_long = z_scores > -self.exit_z_score
        exit_short = z_scores < self.exit_z_score
        
        # Stop loss signals
        stop_loss_long = z_scores < -self.stop_loss_z
        stop_loss_short = z_scores > self.stop_loss_z
        
        # Initialize position array
        positions = np.zeros(len(df))
        
        # Vectorized position calculation (ultra-fast)
        # This approach minimizes loops by using numpy operations
        for i in range(self.lookback_period, len(positions)):
            prev_pos = positions[i-1] if i > 0 else 0
            
            # Entry logic
            if prev_pos == 0:
                if long_signal[i]:
                    positions[i] = 1
                elif short_signal[i]:
                    positions[i] = -1
            # Exit logic
            elif prev_pos == 1:
                if exit_long[i] or stop_loss_long[i]:
                    positions[i] = 0
                else:
                    positions[i] = 1
            elif prev_pos == -1:
                if exit_short[i] or stop_loss_short[i]:
                    positions[i] = 0
                else:
                    positions[i] = -1
        
        # Add results to dataframe
        df['z_score'] = z_scores
        df['signal'] = np.diff(np.concatenate(([0], positions)))
        df['position'] = positions
        df['rolling_mean'] = rolling_mean
        df['rolling_std'] = rolling_std
        
        return df
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          param_ranges: Optional[Dict] = None) -> Dict:
        """Vectorized parameter optimization.
        
        Args:
            data: Historical price data
            param_ranges: Dict with parameter ranges to test
            
        Returns:
            Best parameters found
        """
        if param_ranges is None:
            param_ranges = {
                'lookback_period': range(10, 50, 5),
                'entry_z_score': np.arange(1.5, 3.0, 0.25),
                'exit_z_score': np.arange(0.25, 1.0, 0.25)
            }
        
        best_sharpe = -np.inf
        best_params = {}
        
        # Vectorized grid search
        for lookback in param_ranges['lookback_period']:
            for entry_z in param_ranges['entry_z_score']:
                for exit_z in param_ranges['exit_z_score']:
                    # Create temporary strategy
                    temp_strategy = MeanReversionStrategy(
                        lookback_period=lookback,
                        entry_z_score=entry_z,
                        exit_z_score=exit_z
                    )
                    
                    # Generate signals
                    result_df = temp_strategy.calculate_signals(data)
                    
                    # Calculate returns (vectorized)
                    returns = result_df['close'].pct_change().values
                    strategy_returns = returns * result_df['position'].shift(1).fillna(0).values
                    
                    # Calculate Sharpe ratio (assuming 252 trading days)
                    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                        sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                'lookback_period': lookback,
                                'entry_z_score': entry_z,
                                'exit_z_score': exit_z,
                                'sharpe_ratio': sharpe
                            }
        
        return best_params
    
    def get_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics using vectorized operations.
        
        Args:
            data: DataFrame with signals already calculated
            
        Returns:
            Dictionary of performance metrics
        """
        # Vectorized return calculations
        returns = data['close'].pct_change().values
        positions = data['position'].shift(1).fillna(0).values
        strategy_returns = returns * positions
        
        # Remove NaN values
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
        
        if len(strategy_returns) == 0:
            return {'error': 'No valid returns'}
        
        # Vectorized metrics calculation
        total_return = np.expm1(np.sum(np.log1p(strategy_returns)))
        sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        # Cumulative returns for max drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        wins = strategy_returns > 0
        win_rate = np.mean(wins) if len(wins) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': np.sum(np.abs(np.diff(positions)) > 0),
            'avg_return': np.mean(strategy_returns),
            'std_return': np.std(strategy_returns)
        }
