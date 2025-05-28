#!/usr/bin/env python3
"""
Command-line interface for running backtests.
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.standalone import StandaloneBacktest, demo_backtest

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")

def run_custom_backtest(args):
    """Run a custom backtest with specified parameters."""
    logger = logging.getLogger(__name__)
    
    # Parse dates
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    logger.info(f"Running custom backtest:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Initial Capital: ${args.initial_capital:,.2f}")
    logger.info(f"  Commission: {args.commission * 100:.3f}%")
    
    # Create and run backtest
    backtest = StandaloneBacktest(
        initial_capital=args.initial_capital,
        commission=args.commission
    )
    
    results = backtest.run_backtest(symbols, start_date, end_date)
    
    # Print results
    print_results(results, args.output_format)
    
    # Save results if requested
    if args.output_file:
        save_results(results, args.output_file, args.output_format)
        logger.info(f"Results saved to: {args.output_file}")

def print_results(results: dict, output_format: str = 'summary'):
    """Print backtest results in specified format."""
    if output_format == 'json':
        # Convert datetime objects to strings for JSON serialization
        results_copy = results.copy()
        for item in results_copy['portfolio_history']:
            item['timestamp'] = item['timestamp'].isoformat()
        for trade in results_copy['trades']:
            trade['timestamp'] = trade['timestamp'].isoformat()
        
        print(json.dumps(results_copy, indent=2, default=str))
        
    elif output_format == 'detailed':
        print_detailed_results(results)
        
    else:  # summary
        print_summary_results(results)

def print_summary_results(results: dict):
    """Print summary results."""
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Absolute Return: ${results['final_value'] - results['initial_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Volatility: {results['volatility']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Buy Trades: {results['buy_trades']}")
    print(f"  Sell Trades: {results['sell_trades']}")
    print(f"{'='*60}")

def print_detailed_results(results: dict):
    """Print detailed results including trade history."""
    print_summary_results(results)
    
    if results['trades']:
        print(f"\nTRADE HISTORY:")
        print(f"{'Date':<20} {'Side':<4} {'Symbol':<10} {'Qty':<12} {'Price':<12} {'Commission':<12}")
        print("-" * 80)
        
        for trade in results['trades'][-20:]:  # Show last 20 trades
            print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{trade['side']:<4} "
                  f"{trade['symbol']:<10} "
                  f"{trade['quantity']:<12.6f} "
                  f"${trade['price']:<11.2f} "
                  f"${trade['commission']:<11.2f}")
    
    if results['portfolio_history']:
        print(f"\nPORTFOLIO SNAPSHOTS (Last 10):")
        print(f"{'Date':<20} {'Portfolio Value':<20} {'Cash':<15}")
        print("-" * 55)
        
        for snapshot in results['portfolio_history'][-10:]:
            print(f"{snapshot['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"${snapshot['portfolio_value']:<19,.2f} "
                  f"${snapshot['cash']:<14,.2f}")

def save_results(results: dict, filename: str, output_format: str):
    """Save results to file."""
    # Convert datetime objects to strings for serialization
    results_copy = results.copy()
    for item in results_copy['portfolio_history']:
        item['timestamp'] = item['timestamp'].isoformat()
    for trade in results_copy['trades']:
        trade['timestamp'] = trade['timestamp'].isoformat()
    
    if output_format == 'json':
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
    else:
        # Save as text summary
        with open(filename, 'w') as f:
            # Redirect print to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            print_detailed_results(results)
            sys.stdout = old_stdout

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Crypto Pulse V3 Backtesting CLI')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo backtest')
    demo_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Custom backtest command
    custom_parser = subparsers.add_parser('run', help='Run custom backtest')
    custom_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    custom_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    custom_parser.add_argument('--symbols', required=True, help='Comma-separated symbols (e.g., BTC/USD,ETH/USD)')
    custom_parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital (default: 100000)')
    custom_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    custom_parser.add_argument('--output-format', choices=['summary', 'detailed', 'json'], default='summary', help='Output format')
    custom_parser.add_argument('--output-file', help='Save results to file')
    custom_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Quick test command
    quick_parser = subparsers.add_parser('quick', help='Run quick test backtest')
    quick_parser.add_argument('--symbols', default='BTC/USD,ETH/USD', help='Symbols to test')
    quick_parser.add_argument('--days', type=int, default=30, help='Number of days to test')
    quick_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(getattr(args, 'verbose', False))
    
    try:
        if args.command == 'demo':
            demo_backtest()
            
        elif args.command == 'run':
            run_custom_backtest(args)
            
        elif args.command == 'quick':
            # Quick test with recent dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            symbols = [s.strip() for s in args.symbols.split(',')]
            
            print(f"Running quick backtest:")
            print(f"  Period: {start_date.date()} to {end_date.date()}")
            print(f"  Symbols: {symbols}")
            
            backtest = StandaloneBacktest()
            results = backtest.run_backtest(symbols, start_date, end_date)
            print_summary_results(results)
            
    except Exception as e:
        logging.error(f"Error: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
