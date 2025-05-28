#!/usr/bin/env python3
"""
Command-line interface for integrated backtesting with database connectivity.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestConfig,
    IntegratedBacktestEngine,
    BacktestValidator,
    BacktestReportGenerator,
    database_data_provider
)
from src.core.database import db_manager
from config.settings import settings

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def check_database_connection():
    """Check if database is accessible."""
    try:
        await db_manager.initialize()
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


async def check_data_availability(symbols: list, start_date: datetime, end_date: datetime):
    """Check data availability for symbols in the date range."""
    logger.info("Checking data availability...")
    
    available_symbols = []
    for symbol in symbols:
        try:
            df = await database_data_provider.get_historical_data(
                symbol, start_date, end_date
            )
            if not df.empty:
                available_symbols.append(symbol)
                logger.info(f"✓ {symbol}: {len(df)} data points available")
            else:
                logger.warning(f"✗ {symbol}: No data available")
        except Exception as e:
            logger.error(f"✗ {symbol}: Error checking data - {e}")
    
    return available_symbols


async def run_integrated_backtest(args):
    """Run integrated backtest with database connectivity."""
    logger.info("Starting Integrated Backtest")
    
    # Check database connection
    if not await check_database_connection():
        logger.error("Cannot proceed without database connection")
        return False
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    # Determine symbols
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = settings.trading.trading_pairs[:args.max_symbols]
    
    # Check data availability
    available_symbols = await check_data_availability(symbols, start_date, end_date)
    
    if not available_symbols:
        logger.error("No symbols with available data. Cannot proceed.")
        return False
    
    logger.info(f"Proceeding with {len(available_symbols)} symbols: {available_symbols}")
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        symbols=available_symbols,
        max_position_size=Decimal(str(args.max_position_size)),
        stop_loss_pct=Decimal(str(args.stop_loss_pct)),
        take_profit_pct=Decimal(str(args.take_profit_pct)),
        commission_rate=Decimal(str(args.commission_rate)),
        slippage_rate=Decimal(str(args.slippage_rate))
    )
    
    logger.info(f"Backtest Configuration:")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Initial Capital: ${config.initial_capital:,}")
    logger.info(f"  Symbols: {len(available_symbols)} pairs")
    logger.info(f"  Max Position Size: {config.max_position_size:.1%}")
    
    try:
        # Create and run integrated backtest
        engine = IntegratedBacktestEngine(config)
        
        logger.info("Running integrated backtest...")
        start_time = datetime.utcnow()
        
        results = await engine.run_backtest()
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Backtest completed in {duration:.1f} seconds")
        
        # Display results
        print("\n" + "="*60)
        print("INTEGRATED BACKTEST RESULTS")
        print("="*60)
        
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Duration: {(end_date - start_date).days} days")
        print(f"Symbols: {len(available_symbols)}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {results.total_return:.2%}")
        print(f"  Annualized Return: {results.annualized_return:.2%}")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {results.max_drawdown:.2%}")
        print(f"  Calmar Ratio: {results.calmar_ratio:.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Win Rate: {results.win_rate:.2%}")
        print(f"  Profit Factor: {results.profit_factor:.2f}")
        print(f"  Average Trade: {results.avg_trade_return:.2%}")
        print(f"  Best Trade: {results.best_trade:.2%}")
        print(f"  Worst Trade: {results.worst_trade:.2%}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        print(f"  Volatility: {results.volatility:.2%}")
        print(f"  VaR (95%): {results.var_95:.2%}")
        print(f"  CVaR (95%): {results.cvar_95:.2%}")
        print(f"  Ulcer Index: {results.ulcer_index:.2f}")
        
        # Generate reports if requested
        if args.generate_report:
            logger.info("\nGenerating detailed reports...")
            
            report_generator = BacktestReportGenerator()
            
            # HTML report
            html_path = await report_generator.generate_html_report(
                results, output_file=f"integrated_backtest_{start_date.strftime('%Y%m%d')}.html"
            )
            print(f"HTML Report: {html_path}")
            
            # JSON export
            json_path = await report_generator.export_json(
                results, output_file=f"integrated_backtest_{start_date.strftime('%Y%m%d')}.json"
            )
            print(f"JSON Export: {json_path}")
            
            # CSV export
            csv_path = await report_generator.export_csv(
                results, output_file=f"integrated_backtest_{start_date.strftime('%Y%m%d')}.csv"
            )
            print(f"CSV Export: {csv_path}")
        
        # Validate against live signals if requested
        if args.validate_signals:
            logger.info("\nValidating against live trading signals...")
            
            validator = BacktestValidator()
            validation_results = await validator.validate_against_live_signals(
                results, start_date, end_date
            )
            
            if validation_results:
                print(f"\nSignal Validation Results:")
                for symbol, validation in validation_results.items():
                    print(f"  {symbol}:")
                    print(f"    Signal Accuracy: {validation.get('signal_accuracy', 0):.2%}")
                    print(f"    Timing Accuracy: {validation.get('timing_accuracy', 0):.2%}")
                    print(f"    Live Signals: {validation.get('live_signals_count', 0)}")
                    print(f"    Backtest Signals: {validation.get('backtest_signals_count', 0)}")
        
        print("\n" + "="*60)
        print("INTEGRATED BACKTEST COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running integrated backtest: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_database_data(args):
    """Test database data availability and quality."""
    logger.info("Testing Database Data Quality")
    
    if not await check_database_connection():
        return False
    
    # Test period
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)  # Last week
    
    symbols = settings.trading.trading_pairs[:10]  # Test top 10
    
    print("\n" + "="*60)
    print("DATABASE DATA QUALITY TEST")
    print("="*60)
    
    total_records = 0
    symbols_with_data = 0
    
    for symbol in symbols:
        try:
            df = await database_data_provider.get_historical_data(
                symbol, start_date, end_date
            )
            
            if not df.empty:
                symbols_with_data += 1
                total_records += len(df)
                
                # Check for gaps
                expected_hours = int((end_date - start_date).total_seconds() / 3600)
                coverage = len(df) / expected_hours * 100
                
                print(f"  {symbol}: {len(df)} records ({coverage:.1f}% coverage)")
                
                # Check data quality
                if df['close'].isna().any():
                    print(f"    ⚠ Warning: Missing close prices")
                if (df['volume'] == 0).any():
                    print(f"    ⚠ Warning: Zero volume periods")
            else:
                print(f"  {symbol}: No data available")
                
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    print(f"\nSummary:")
    print(f"  Symbols tested: {len(symbols)}")
    print(f"  Symbols with data: {symbols_with_data}")
    print(f"  Total records: {total_records:,}")
    print(f"  Data availability: {symbols_with_data/len(symbols)*100:.1f}%")
    
    return True


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Integrated Backtesting CLI with Database Connectivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 30-day backtest with default settings
  python integrated_backtest_cli.py run --days 30
  
  # Run backtest for specific period and symbols
  python integrated_backtest_cli.py run --start-date 2024-01-01 --end-date 2024-01-31 --symbols BTCUSDT,ETHUSDT
  
  # Run backtest with custom risk parameters
  python integrated_backtest_cli.py run --days 14 --max-position-size 0.2 --stop-loss-pct 0.03
  
  # Test database connectivity and data quality
  python integrated_backtest_cli.py test-data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run backtest command
    run_parser = subparsers.add_parser('run', help='Run integrated backtest')
    run_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    run_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    run_parser.add_argument('--days', type=int, default=30, help='Number of days back from today (default: 30)')
    run_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (default: from settings)')
    run_parser.add_argument('--max-symbols', type=int, default=5, help='Maximum number of symbols to test (default: 5)')
    run_parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    run_parser.add_argument('--max-position-size', type=float, default=0.15, help='Max position size as fraction (default: 0.15)')
    run_parser.add_argument('--stop-loss-pct', type=float, default=0.05, help='Stop loss percentage (default: 0.05)')
    run_parser.add_argument('--take-profit-pct', type=float, default=0.10, help='Take profit percentage (default: 0.10)')
    run_parser.add_argument('--commission-rate', type=float, default=0.001, help='Commission rate (default: 0.001)')
    run_parser.add_argument('--slippage-rate', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    run_parser.add_argument('--generate-report', action='store_true', help='Generate detailed reports')
    run_parser.add_argument('--validate-signals', action='store_true', help='Validate against live signals')
    
    # Test data command
    test_parser = subparsers.add_parser('test-data', help='Test database data availability')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            success = asyncio.run(run_integrated_backtest(args))
        elif args.command == 'test-data':
            success = asyncio.run(test_database_data(args))
        else:
            parser.print_help()
            return
        
        if success:
            logger.info("Command completed successfully")
        else:
            logger.error("Command failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
