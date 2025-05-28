"""
Standalone backtesting functionality for demonstration and testing.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import uuid
import json

logger = logging.getLogger(__name__)


class MockMarketData:
    """Mock market data for standalone testing."""
    
    @staticmethod
    def generate_price_series(symbol: str, start_date: datetime, end_date: datetime, 
                            initial_price: float = 50000.0) -> pd.DataFrame:
        """Generate realistic crypto price data."""
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        n_periods = len(dates)
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 2**32)  # Reproducible per symbol
        
        # Random walk with trend and volatility
        returns = np.random.normal(0, 0.02, n_periods)  # 2% hourly volatility
        returns[0] = 0  # First return is 0
        
        # Add some trend and mean reversion
        trend = np.linspace(-0.001, 0.001, n_periods)  # Slight upward trend
        returns += trend
        
        # Calculate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate open, high, low based on close
            open_price = prices[i-1] if i > 0 else close
            high = max(open_price, close) * (1 + np.random.uniform(0, 0.01))
            low = min(open_price, close) * (1 - np.random.uniform(0, 0.01))
            volume = np.random.uniform(1000000, 5000000)  # Random volume
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


class SimpleStrategy:
    """Simple moving average crossover strategy for demonstration."""
    
    def __init__(self, fast_ma: int = 12, slow_ma: int = 26):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data and generate signals."""
        if len(data) < self.slow_ma:
            return {'signal': 'HOLD', 'strength': 0.0}
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(window=self.fast_ma).mean().iloc[-1]
        slow_ma = data['close'].rolling(window=self.slow_ma).mean().iloc[-1]
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Simple signal logic
        current_price = data['close'].iloc[-1]
        signal_strength = (fast_ma - slow_ma) / current_price
        
        if fast_ma > slow_ma and rsi < 70:
            return {
                'signal': 'BUY',
                'strength': min(abs(signal_strength), 1.0),
                'price': current_price,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi': rsi
            }
        elif fast_ma < slow_ma and rsi > 30:
            return {
                'signal': 'SELL',
                'strength': min(abs(signal_strength), 1.0),
                'price': current_price,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi': rsi
            }
        else:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'price': current_price,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi': rsi
            }


class StandaloneBacktest:
    """Standalone backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        self.strategy = SimpleStrategy()
        
    def execute_trade(self, symbol: str, signal: str, quantity: float, price: float, timestamp: datetime):
        """Execute a trade."""
        commission_cost = quantity * price * self.commission
        
        if signal == 'BUY':
            total_cost = quantity * price + commission_cost
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                trade = {
                    'id': str(uuid.uuid4()),
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': price,
                    'commission': commission_cost,
                    'total_cost': total_cost
                }
                self.trades.append(trade)
                return True
                
        elif signal == 'SELL':
            if self.positions.get(symbol, 0) >= quantity:
                proceeds = quantity * price - commission_cost
                self.cash += proceeds
                self.positions[symbol] -= quantity
                
                trade = {
                    'id': str(uuid.uuid4()),
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'commission': commission_cost,
                    'proceeds': proceeds
                }
                self.trades.append(trade)
                return True
        
        return False
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            self.positions.get(symbol, 0) * price
            for symbol, price in prices.items()
        )
        return self.cash + position_value
    
    def run_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtest on multiple symbols."""
        logger.info(f"Running standalone backtest from {start_date} to {end_date}")
        
        # Generate market data for all symbols
        market_data = {}
        for symbol in symbols:
            data = MockMarketData.generate_price_series(symbol, start_date, end_date)
            market_data[symbol] = data
        
        # Get all timestamps (union of all data)
        all_timestamps = set()
        for data in market_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(all_timestamps)
        
        # Run backtest
        for timestamp in timestamps:
            current_prices = {}
            
            # Get current market data and generate signals
            for symbol in symbols:
                if timestamp in market_data[symbol].index:
                    current_data = market_data[symbol].loc[:timestamp]
                    current_prices[symbol] = current_data['close'].iloc[-1]
                    
                    # Generate signal
                    if len(current_data) >= 30:  # Need enough data for analysis
                        analysis = self.strategy.analyze(current_data.tail(50))
                        
                        if analysis['signal'] == 'BUY':
                            # Calculate position size (simple allocation)
                            allocation = 0.1  # 10% per position
                            dollar_amount = self.calculate_portfolio_value(current_prices) * allocation
                            quantity = dollar_amount / current_prices[symbol]
                            
                            if quantity > 0:
                                self.execute_trade(symbol, 'BUY', quantity, current_prices[symbol], timestamp)
                                
                        elif analysis['signal'] == 'SELL':
                            # Sell entire position
                            current_position = self.positions.get(symbol, 0)
                            if current_position > 0:
                                self.execute_trade(symbol, 'SELL', current_position, current_prices[symbol], timestamp)
            
            # Record portfolio snapshot
            if current_prices:
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'positions': dict(self.positions)
                })
        
        # Calculate final results
        final_prices = {symbol: data['close'].iloc[-1] for symbol, data in market_data.items()}
        final_value = self.calculate_portfolio_value(final_prices)
        
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate additional metrics
        portfolio_values = [snap['portfolio_value'] for snap in self.portfolio_history]
        if len(portfolio_values) > 1:
            returns = pd.Series(portfolio_values).pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365)  # Annualized (hourly data)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(24 * 365) if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = pd.Series(portfolio_values)
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade statistics
        buy_trades = [t for t in self.trades if t['side'] == 'BUY']
        sell_trades = [t for t in self.trades if t['side'] == 'SELL']
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'portfolio_history': self.portfolio_history,
            'trades': self.trades
        }
        
        return results


def demo_backtest():
    """Demonstrate standalone backtesting."""
    logger.info("=== Standalone Backtesting Demo ===")
    
    # Configuration
    symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    initial_capital = 100000.0
    
    # Run backtest
    backtest = StandaloneBacktest(initial_capital=initial_capital)
    results = backtest.run_backtest(symbols, start_date, end_date)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Buy Trades: {results['buy_trades']}")
    print(f"Sell Trades: {results['sell_trades']}")
    
    # Show some trade examples
    if results['trades']:
        print(f"\nFirst 5 Trades:")
        for i, trade in enumerate(results['trades'][:5]):
            print(f"  {i+1}. {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{trade['side']} {trade['quantity']:.4f} {trade['symbol']} "
                  f"at ${trade['price']:.2f}")
    
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_backtest()
