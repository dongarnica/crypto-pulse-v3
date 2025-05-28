# Docker Deployment Guide for Crypto Pulse V3

This guide covers how to deploy the Crypto Pulse V3 trading system using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+ 
- Docker Compose 2.0+
- At least 4GB RAM
- 10GB free disk space

## Quick Start

### Development Environment

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd crypto-pulse-v3
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Start Development Environment**
   ```bash
   ./scripts/docker-deploy.sh dev
   ```

4. **Access Services**
   - API Server: http://localhost:8000
   - Database: localhost:5432
   - Redis: localhost:6379

### Production Environment

1. **Configure Production Environment**
   ```bash
   cp .env.production.example .env.production
   # Edit .env.production with your production settings
   ```

2. **Start Production Environment**
   ```bash
   ./scripts/docker-deploy.sh prod
   ```

3. **Access Services**
   - API Server: http://localhost:8000
   - Monitoring Dashboard: http://localhost:3000

## Architecture

The Docker setup includes the following services:

### Core Services
- **app**: Main trading application
- **postgres**: PostgreSQL 15 with TimescaleDB extension
- **redis**: Redis cache and message broker

### Optional Services (Production)
- **monitoring**: Performance monitoring dashboard
- **backtesting**: Dedicated backtesting service

### Utility Services
- **db-migrate**: One-time database initialization

## Configuration

### Environment Variables

#### Required for Production
```env
# API Keys
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Database
POSTGRES_PASSWORD=secure_password
SECRET_KEY=32_character_secret_key
```

#### Optional Configuration
```env
# Trading Parameters
MAX_PORTFOLIO_ALLOCATION=0.15
MAX_DRAWDOWN_THRESHOLD=0.10
ATR_STOP_MULTIPLIER=3.5

# Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Volume Mounts

- **postgres_data**: Database persistence
- **redis_data**: Redis persistence  
- **logs**: Application logs
- **backtest_cache**: Backtesting cache
- **data**: Market data storage

## Management Commands

Use the `docker-deploy.sh` script for common operations:

```bash
# Development
./scripts/docker-deploy.sh dev          # Start development
./scripts/docker-deploy.sh stop         # Stop development

# Production  
./scripts/docker-deploy.sh prod         # Start production
./scripts/docker-deploy.sh stop prod    # Stop production

# Monitoring
./scripts/docker-deploy.sh status       # Check service status
./scripts/docker-deploy.sh logs         # View logs
./scripts/docker-deploy.sh logs prod app # View production app logs

# Database
./scripts/docker-deploy.sh db-init      # Initialize database
./scripts/docker-deploy.sh db-backup    # Backup database

# Backtesting
./scripts/docker-deploy.sh backtest     # Run backtesting

# Maintenance
./scripts/docker-deploy.sh cleanup      # Clean up resources
```

## Manual Docker Commands

### Build and Start
```bash
# Build images
docker-compose build

# Start development
docker-compose up -d

# Start production
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

### Service Management
```bash
# View running services
docker-compose ps

# View logs
docker-compose logs -f app

# Restart a service
docker-compose restart app

# Scale services (if needed)
docker-compose up -d --scale app=2
```

### Database Operations
```bash
# Initialize database
docker-compose run --rm db-migrate

# Access database shell
docker-compose exec postgres psql -U postgres -d crypto_pulse_v3

# Backup database
docker-compose exec postgres pg_dump -U postgres crypto_pulse_v3 > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres -d crypto_pulse_v3 < backup.sql
```

### Backtesting
```bash
# Run advanced backtesting
docker-compose run --rm backtesting python working_advanced_cli.py demo

# Run specific backtest
docker-compose run --rm backtesting python working_advanced_cli.py monte-carlo --symbol BTCUSDT --scenarios 1000
```

## Monitoring and Maintenance

### Health Checks
All services include health checks:
- **PostgreSQL**: Connection and query test
- **Redis**: Ping test
- **Application**: API endpoint test

### Log Management
Logs are configured with rotation:
- Maximum size: 10MB per file
- Maximum files: 3-5 files retained
- Format: JSON for structured logging

### Performance Monitoring
```bash
# View resource usage
docker stats

# View detailed service info
docker-compose ps
docker-compose top

# Monitor logs in real-time
docker-compose logs -f --tail=100
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :8000
   # Change ports in docker-compose.yml if needed
   ```

2. **Database Connection Issues**
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Verify database is ready
   docker-compose exec postgres pg_isready -U postgres
   ```

3. **API Key Configuration**
   ```bash
   # Verify environment variables
   docker-compose exec app env | grep API_KEY
   ```

4. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats --no-stream
   
   # Increase Docker memory if needed
   # Docker Desktop: Settings > Resources > Memory
   ```

### Debug Mode
```bash
# Run with debug logging
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml docker-compose up

# Access container shell
docker-compose exec app bash

# Run individual components
docker-compose run --rm app python -c "from src.core.database import test_connection; test_connection()"
```

### Cleanup and Reset
```bash
# Stop and remove everything
./scripts/docker-deploy.sh cleanup

# Remove all volumes (CAUTION: deletes all data)
docker-compose down -v
docker volume prune -f

# Reset to clean state
docker system prune -a -f
```

## Security Considerations

### Production Security
1. **Environment Variables**: Never commit `.env.production` to version control
2. **API Keys**: Use secure, production API keys
3. **Database**: Use strong passwords and limit connections
4. **Network**: Consider using Docker secrets for sensitive data
5. **Firewall**: Restrict access to necessary ports only

### Network Security
```bash
# View Docker networks
docker network ls

# Inspect network configuration
docker network inspect crypto-pulse-v3_crypto-pulse-network
```

## Backup and Recovery

### Automated Backups
```bash
# Setup automated daily backups (add to crontab)
0 2 * * * /path/to/crypto-pulse-v3/scripts/docker-deploy.sh db-backup prod
```

### Disaster Recovery
1. **Database**: Regular PostgreSQL dumps
2. **Configuration**: Backup `.env` files
3. **Code**: Version control with Git
4. **Data**: Backup volume mounts

### Backup Script Example
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/crypto-pulse-v3"
mkdir -p $BACKUP_DIR

# Database backup
docker-compose exec postgres pg_dump -U postgres crypto_pulse_v3 > $BACKUP_DIR/db_$DATE.sql

# Configuration backup
cp .env.production $BACKUP_DIR/env_$DATE.backup

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete
```

## Performance Optimization

### Resource Allocation
- **CPU**: 2+ cores recommended
- **Memory**: 4GB minimum, 8GB recommended
- **Storage**: SSD recommended for database
- **Network**: Stable internet for API connections

### Database Optimization
- TimescaleDB for time-series data
- Automatic compression for old data
- Retention policies for data management
- Performance indexes for common queries

### Application Optimization
- Redis caching for frequent data
- Connection pooling for database
- Async processing for heavy operations
- Monitoring and alerting for performance issues

## Advanced Configuration

### Custom Docker Compose
Create `docker-compose.local.yml` for local overrides:
```yaml
version: '3.8'
services:
  app:
    ports:
      - "8080:8000"  # Custom port
    environment:
      - CUSTOM_SETTING=value
```

Use with:
```bash
docker-compose -f docker-compose.yml -f docker-compose.local.yml up
```

### Multi-Stage Deployment
```bash
# Build for different environments
docker build --target development -t crypto-pulse-v3:dev .
docker build --target production -t crypto-pulse-v3:prod .
docker build --target backtesting -t crypto-pulse-v3:backtest .
```

This completes the comprehensive Docker deployment setup for Crypto Pulse V3. The system is now fully containerized and ready for both development and production deployment.
