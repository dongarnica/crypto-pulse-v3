# Crypto Pulse V3 - Medium-Frequency Trading System

A sophisticated medium-frequency cryptocurrency trading system that captures profitable opportunities through strategic 1-4 hour trading cycles, operating in the optimal zone between rapid scalping and long-term position holding.

## Architecture Overview

### Trading Strategy
- **Medium-Frequency Focus**: 1-4 hour trading cycles capturing meaningful price movements
- **Multi-Timeframe Analysis**: Comprehensive technical analysis across multiple timeframes
- **Sentiment Integration**: Perplexity AI sentiment analysis every 2 hours
- **Decision Cycles**: 30-minute comprehensive analysis cycles

### Machine Learning Framework
- **Three-Tier Ensemble**: Random Forest + LSTM + Transformer models
- **Feature Engineering**: 4-hour aggregated technical indicators (RSI, MACD, Bollinger Bands)
- **Sequential Processing**: 30+ days of market history for pattern recognition
- **Regime Detection**: Transformer-based attention for market structure breaks

### Risk Management
- **Kelly Criterion**: Dynamic position sizing (8-15% allocation)
- **Volatility-Adjusted Stops**: 3-5% stop losses based on ATR
- **Correlation Limits**: Prevent overconcentration across positions
- **Dynamic Scaling**: Exposure adjustment based on market conditions

### Infrastructure
- **Data Sources**: Binance WebSockets + Alpaca Markets execution
- **Database**: PostgreSQL with TimescaleDB for time-series optimization
- **Cloud-Native**: Scalable, reliable infrastructure prioritizing analytical depth
- **Monitoring**: 15-25 major cryptocurrency pairs with $50M+ daily volume

### Performance Targets
- **Annual Returns**: >28% with <15% maximum drawdown
- **Win Rate**: >65% on 4-hour directional predictions
- **Profit Factor**: >2.2
- **Sharpe Ratio**: >1.8

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python scripts/init_db.py

# Start the trading system
python main.py
```

## Project Structure

```
crypto-pulse-v3/
├── src/
│   ├── core/           # Core trading engine
│   ├── data/           # Data ingestion and processing
│   ├── ml/             # Machine learning models
│   ├── risk/           # Risk management
│   ├── execution/      # Trade execution
│   └── monitoring/     # System monitoring
├── config/             # Configuration files
├── scripts/            # Utility scripts
├── tests/              # Test suites
├── docs/               # Documentation
└── notebooks/          # Jupyter notebooks for analysis
```

## Configuration

The system uses environment variables for configuration. Key settings include:

- **API Keys**: Binance, Alpaca, Perplexity AI
- **Database**: PostgreSQL connection settings
- **Risk Parameters**: Position sizing, stop losses, exposure limits
- **Trading Pairs**: Monitored cryptocurrency pairs
- **Performance Thresholds**: Risk and return targets

## Development

```bash
# Run tests
pytest

# Code formatting
black src/

# Type checking
mypy src/

# Start development server with hot reload
uvicorn src.api.main:app --reload
```

## Monitoring

The system includes comprehensive monitoring:

- **Performance Metrics**: Real-time P&L, Sharpe ratio, drawdown tracking
- **Risk Monitoring**: Position sizing, correlation, exposure limits
- **System Health**: Database connections, API latency, model performance
- **Alerts**: Telegram notifications for significant events

## License

Proprietary - Internal Trading System