# Crypto Pulse V3 Backtesting System

## Overview

The Crypto Pulse V3 Backtesting System provides comprehensive historical strategy validation and performance testing capabilities. It includes:

- **Core Backtesting Engine**: Historical simulation with realistic market conditions
- **Parameter Optimization**: Grid search, random search, and Bayesian optimization
- **Walk-Forward Analysis**: Time-series validation with out-of-sample testing
- **Comprehensive Reporting**: HTML reports with charts, JSON exports, and CSV data
- **CLI Tools**: Command-line interface for easy backtesting operations

## Quick Start

### 1. Run Demo Backtest

```bash
cd /workspaces/crypto-pulse-v3
PYTHONPATH=src python backtest_cli.py demo
```

### 2. Quick Test (7-day backtest)

```bash
PYTHONPATH=src python backtest_cli.py quick --days 7 --symbols "BTC/USD,ETH/USD"
```

### 3. Custom Backtest

```bash
PYTHONPATH=src python backtest_cli.py run \
  --start-date "2024-01-01" \
  --end-date "2024-03-31" \
  --symbols "BTC/USD,ETH/USD,ADA/USD" \
  --initial-capital 50000 \
  --commission 0.001 \
  --output-format detailed
```

## System Architecture

### Core Components

1. **BacktestEngine** (`src/backtesting/engine.py`)
   - Main orchestrator for historical simulation
   - Manages portfolio state, trade execution, and market analysis
   - Integrates with existing trading infrastructure

2. **BacktestConfig** (`src/backtesting/config.py`)
   - Configuration management for backtest parameters
   - Supports optimization parameter ranges
   - Flexible time periods and trading settings

3. **BacktestResults** (`src/backtesting/results.py`)
   - Comprehensive performance metrics calculation
   - Trade tracking and portfolio snapshots
   - Statistical analysis and risk metrics

4. **ParameterOptimizer** (`src/backtesting/optimization.py`)
   - Multi-method parameter optimization
   - Walk-forward analysis capabilities
   - Parallel processing support

5. **ReportGenerator** (`src/backtesting/reports.py`)
   - HTML reports with interactive charts
   - JSON exports for further analysis
   - CSV data exports

### Key Features

#### Portfolio Management
- **Multi-asset portfolio tracking**
- **Position sizing based on risk management**
- **Commission and slippage simulation**
- **Cash management and margin requirements**

#### Strategy Integration
- **Technical analysis indicators**
- **ML model predictions**
- **Sentiment analysis signals**
- **Risk-based position sizing**

#### Performance Analysis
- **Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)**
- **Drawdown analysis and recovery metrics**
- **Trading statistics (win rate, profit factor)**
- **Advanced risk metrics (VaR, Ulcer Index)**

## Configuration Examples

### Basic Backtest Configuration

```python
from datetime import datetime
from decimal import Decimal
from backtesting.config import BacktestConfig

config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=Decimal('100000'),
    symbols=['BTC/USD', 'ETH/USD', 'ADA/USD'],
    max_portfolio_allocation=0.20,
    commission_rate=0.001,
    analysis_interval_minutes=60
)
```

### Parameter Optimization Configuration

```python
from backtesting.config import OptimizationConfig

optimization_config = OptimizationConfig(
    method='random_search',
    objective='sharpe_ratio',
    max_iterations=100,
    parameter_ranges={
        'max_portfolio_allocation': [0.10, 0.15, 0.20, 0.25],
        'commission_rate': [0.0005, 0.001, 0.0015],
        'atr_stop_multiplier': [2.0, 3.0, 4.0],
        'min_prediction_confidence': [0.60, 0.65, 0.70]
    },
    max_parallel_jobs=4
)
```

## Usage Examples

### 1. Simple Backtest

```python
import asyncio
from datetime import datetime
from decimal import Decimal
from backtesting.config import BacktestConfig
from backtesting.engine import BacktestEngine

async def run_simple_backtest():
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD']
    )
    
    engine = BacktestEngine(config)
    results = await engine.run_backtest()
    
    print(f"Total Return: {results.total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {results.max_drawdown * 100:.2f}%")

asyncio.run(run_simple_backtest())
```

### 2. Parameter Optimization

```python
from backtesting.optimization import ParameterOptimizer

async def run_optimization():
    base_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD']
    )
    
    optimization_config = OptimizationConfig(
        method='random_search',
        objective='sharpe_ratio',
        max_iterations=50,
        parameter_ranges={
            'max_portfolio_allocation': [0.15, 0.20, 0.25],
            'commission_rate': [0.001, 0.0015]
        }
    )
    
    optimizer = ParameterOptimizer(base_config, optimization_config)
    results = await optimizer.optimize()
    
    print(f"Best Score: {results.best_score:.4f}")
    print(f"Best Parameters: {results.best_parameters}")

asyncio.run(run_optimization())
```

### 3. Walk-Forward Analysis

```python
from backtesting.optimization import WalkForwardOptimizer

async def run_walk_forward():
    base_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD']
    )
    
    optimization_config = OptimizationConfig(
        method='random_search',
        objective='sharpe_ratio',
        max_iterations=20,
        parameter_ranges={
            'max_portfolio_allocation': [0.15, 0.20, 0.25]
        },
        train_period_days=90,
        test_period_days=30,
        step_days=30
    )
    
    wf_optimizer = WalkForwardOptimizer(base_config, optimization_config)
    results = await wf_optimizer.run_walk_forward_optimization()
    
    print(f"Average OOS Return: {results['average_oos_return']:.4f}")
    print(f"OOS Sharpe Ratio: {results['oos_sharpe_ratio']:.4f}")

asyncio.run(run_walk_forward())
```

### 4. Generate Reports

```python
from backtesting.reports import BacktestReportGenerator

# After running a backtest and getting results
report_generator = BacktestReportGenerator(results)
report_files = report_generator.generate_full_report()

print(f"HTML Report: {report_files['html_report']}")
print(f"JSON Report: {report_files['json_report']}")
print(f"CSV Exports: {report_files['csv_export_dir']}")
```

## CLI Reference

### Available Commands

#### Demo
Run a demonstration backtest with predefined parameters:
```bash
PYTHONPATH=src python backtest_cli.py demo
```

#### Quick Test
Run a quick backtest for recent periods:
```bash
PYTHONPATH=src python backtest_cli.py quick \
  --symbols "BTC/USD,ETH/USD" \
  --days 30
```

#### Custom Backtest
Run a fully customized backtest:
```bash
PYTHONPATH=src python backtest_cli.py run \
  --start-date "2024-01-01" \
  --end-date "2024-06-30" \
  --symbols "BTC/USD,ETH/USD,ADA/USD" \
  --initial-capital 100000 \
  --commission 0.001 \
  --output-format detailed \
  --output-file results.json
```

### Command Options

- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--symbols`: Comma-separated list of trading symbols
- `--initial-capital`: Starting capital amount (default: 100000)
- `--commission`: Commission rate as decimal (default: 0.001)
- `--output-format`: Output format (summary, detailed, json)
- `--output-file`: Save results to specified file
- `--verbose`: Enable verbose logging

## Performance Metrics

### Return Metrics
- **Total Return**: Overall portfolio return percentage
- **Annualized Return**: Return adjusted for time period
- **CAGR**: Compound Annual Growth Rate

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (return/volatility)
- **Sortino Ratio**: Downside deviation adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win/Loss**: Average profit per winning/losing trade
- **Maximum Consecutive Wins/Losses**: Longest winning/losing streaks

## Integration with Trading System

The backtesting system integrates seamlessly with the existing Crypto Pulse V3 infrastructure:

### Data Sources
- **Historical market data** from database
- **Technical indicators** from technical_analysis module
- **ML predictions** from ensemble module
- **Sentiment analysis** from sentiment module

### Risk Management
- **Kelly Criterion** position sizing
- **Correlation analysis** for diversification
- **ATR-based stop losses**
- **Drawdown controls**

### Execution Simulation
- **Realistic commission rates**
- **Market impact modeling**
- **Slippage simulation**
- **Order book depth consideration**

## Best Practices

### 1. Data Quality
- Ensure sufficient historical data (minimum 1 year)
- Use consistent time intervals
- Account for market holidays and gaps

### 2. Parameter Selection
- Start with reasonable parameter ranges
- Use walk-forward analysis for time-series data
- Avoid over-optimization (curve fitting)

### 3. Risk Management
- Set appropriate maximum drawdown limits
- Use position sizing based on Kelly Criterion
- Monitor correlation between positions

### 4. Validation
- Use out-of-sample testing
- Perform walk-forward analysis
- Compare against buy-and-hold benchmarks

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Set PYTHONPATH correctly
export PYTHONPATH=/workspaces/crypto-pulse-v3/src
python your_script.py
```

#### Memory Issues
- Reduce the number of symbols or time period
- Increase system memory allocation
- Use data sampling for initial tests

#### Performance Optimization
- Use parallel processing for optimization
- Limit the number of parameter combinations
- Cache historical data for repeated runs

### Debug Mode
Enable verbose logging for troubleshooting:
```bash
PYTHONPATH=src python backtest_cli.py run --verbose [other options]
```

## Future Enhancements

### Planned Features
1. **Monte Carlo Simulation**: Portfolio risk assessment
2. **Multi-timeframe Analysis**: Different analysis intervals
3. **Advanced Order Types**: Stop-loss, take-profit orders
4. **Benchmark Comparison**: Against market indices
5. **Real-time Validation**: Paper trading mode

### Performance Improvements
1. **Data Caching**: Faster repeated backtests
2. **Parallel Processing**: Multi-core optimization
3. **Memory Optimization**: Large dataset handling
4. **GPU Acceleration**: ML model inference

## Support and Documentation

- **Examples**: See `examples/run_backtest.py`
- **CLI Help**: `python backtest_cli.py --help`
- **Test Suite**: Run `python test_backtesting.py`
- **Standalone Demo**: Run `python src/backtesting/standalone.py`

For additional support, check the logging output with `--verbose` flag or review the generated reports for detailed analysis.
