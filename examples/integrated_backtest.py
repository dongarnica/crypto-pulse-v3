#!/usr/bin/env python3
"""
Example script demonstrating integrated backtesting with live strategy integration.
This shows how to run backtests using real market data and live trading strategies.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestConfig, 
    IntegratedBacktestEngine,
    BacktestValidator,
    BacktestReportGenerator
)
from config.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_integrated_backtest():
    """Run a comprehensive integrated backtest."""
    logger.info("Starting Integrated Backtesting Demo")
    
    try:
        # Configuration for integrated backtest
        config = BacktestConfig(
            # Test period - last 30 days
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow() - timedelta(days=1),
            
            # Portfolio settings
            initial_capital=Decimal('100000'),  # $100K starting capital
            
            # Trading symbols from settings
            symbols=settings.trading.trading_pairs[:5],  # Test with top 5 pairs
            
            # Risk management
            max_position_size=Decimal('0.15'),  # 15% max position
            stop_loss_pct=Decimal('0.05'),      # 5% stop loss
            take_profit_pct=Decimal('0.10'),    # 10% take profit
            
            # Costs
            commission_rate=Decimal('0.001'),   # 0.1% commission
            slippage_rate=Decimal('0.0005')     # 0.05% slippage
        )
        
        logger.info(f"Backtest Configuration:")
        logger.info(f"  Period: {config.start_date} to {config.end_date}")
        logger.info(f"  Capital: ${config.initial_capital:,}")
        logger.info(f"  Symbols: {config.symbols}")
        
        # Create integrated backtest engine
        engine = IntegratedBacktestEngine(config)
        
        # Run backtest
        logger.info("Running integrated backtest...")
        start_time = datetime.utcnow()
        
        results = await engine.run_backtest()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Backtest completed in {duration:.1f} seconds")
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Annualized Return: {results.annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        
        logger.info(f"\nTrading Statistics:")
        logger.info(f"  Total Trades: {results.total_trades}")
        logger.info(f"  Winning Trades: {results.winning_trades}")
        logger.info(f"  Losing Trades: {results.losing_trades}")
        logger.info(f"  Average Trade: {results.avg_trade_return:.2%}")
        logger.info(f"  Best Trade: {results.best_trade:.2%}")
        logger.info(f"  Worst Trade: {results.worst_trade:.2%}")
        
        # Generate HTML report
        report_generator = BacktestReportGenerator()
        report_path = await report_generator.generate_html_report(
            results, 
            output_file="integrated_backtest_report.html"
        )
        logger.info(f"\nDetailed HTML report generated: {report_path}")
        
        # Validate against live signals
        logger.info("\nValidating against live trading signals...")
        validator = BacktestValidator()
        validation_results = await validator.validate_against_live_signals(
            results, config.start_date, config.end_date
        )
        
        if validation_results:
            logger.info("Signal Validation Results:")
            for symbol, validation in validation_results.items():
                logger.info(f"  {symbol}:")
                logger.info(f"    Signal Accuracy: {validation.get('signal_accuracy', 0):.2%}")
                logger.info(f"    Timing Accuracy: {validation.get('timing_accuracy', 0):.2%}")
                logger.info(f"    Signal Count Ratio: {validation.get('signal_count_ratio', 0):.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in integrated backtest: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def run_strategy_comparison():
    """Compare different strategy configurations."""
    logger.info("\nRunning Strategy Comparison")
    
    strategies = [
        {
            'name': 'Conservative',
            'max_position_size': Decimal('0.10'),
            'stop_loss_pct': Decimal('0.03'),
            'take_profit_pct': Decimal('0.08')
        },
        {
            'name': 'Moderate',
            'max_position_size': Decimal('0.15'),
            'stop_loss_pct': Decimal('0.05'),
            'take_profit_pct': Decimal('0.10')
        },
        {
            'name': 'Aggressive',
            'max_position_size': Decimal('0.20'),
            'stop_loss_pct': Decimal('0.07'),
            'take_profit_pct': Decimal('0.15')
        }
    ]
    
    comparison_results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy['name']} strategy...")
        
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=14),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('50000'),
            symbols=settings.trading.trading_pairs[:3],
            **{k: v for k, v in strategy.items() if k != 'name'}
        )
        
        engine = IntegratedBacktestEngine(config)
        results = await engine.run_backtest()
        
        comparison_results[strategy['name']] = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate
        }
    
    # Display comparison
    logger.info("\n" + "="*60)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*60)
    
    for name, metrics in comparison_results.items():
        logger.info(f"\n{name} Strategy:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")


async def main():
    """Main execution function."""
    try:
        # Run integrated backtest
        results = await run_integrated_backtest()
        
        if results:
            # Run strategy comparison
            await run_strategy_comparison()
            
            logger.info("\n" + "="*60)
            logger.info("INTEGRATION DEMO COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info("\nKey Features Demonstrated:")
            logger.info("✓ Database integration for historical data")
            logger.info("✓ Live strategy integration (ML, technical analysis, sentiment)")
            logger.info("✓ Real risk management integration")
            logger.info("✓ Signal validation against live trading")
            logger.info("✓ Comprehensive performance reporting")
            logger.info("✓ Strategy comparison capabilities")
            
        else:
            logger.error("Integrated backtest failed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
