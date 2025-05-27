from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Database Configuration
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/crypto_pulse_v3", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # API Keys
    binance_api_key: str = Field(default="development_binance_key", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="development_binance_secret", env="BINANCE_SECRET_KEY")
    alpaca_api_key: str = Field(default="development_alpaca_key", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="development_alpaca_secret", env="ALPACA_SECRET_KEY")
    perplexity_api_key: str = Field(default="development_perplexity_key", env="PERPLEXITY_API_KEY")
    
    # Trading Configuration
    environment: str = Field(default="development", env="ENVIRONMENT")
    max_portfolio_allocation: float = Field(default=0.15, env="MAX_PORTFOLIO_ALLOCATION")
    min_portfolio_allocation: float = Field(default=0.08, env="MIN_PORTFOLIO_ALLOCATION")
    max_drawdown_threshold: float = Field(default=0.15, env="MAX_DRAWDOWN_THRESHOLD")
    min_daily_volume_usd: int = Field(default=50000000, env="MIN_DAILY_VOLUME_USD")
    min_daily_volatility: float = Field(default=0.02, env="MIN_DAILY_VOLATILITY")
    
    # Trading pairs to monitor
    trading_pairs: List[str] = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT", "TRXUSDT",
        "BNBUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT",
        "ALGOUSDT", "ATOMUSDT", "FILUSDT", "UNIUSDT", "AAVEUSDT",
        "SUSHIUSDT", "COMPUSDT", "MKRUSDT", "YFIUSDT", "SNXUSDT"
    ]
    
    # Risk Management
    atr_stop_multiplier: float = Field(default=3.5, env="ATR_STOP_MULTIPLIER")
    max_correlation_threshold: float = Field(default=0.7, env="MAX_CORRELATION_THRESHOLD")
    max_sector_exposure: float = Field(default=0.4, env="MAX_SECTOR_EXPOSURE")
    volatility_scaling_factor: float = Field(default=0.5, env="VOLATILITY_SCALING_FACTOR")
    
    # System Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    analysis_interval_minutes: int = Field(default=30, env="ANALYSIS_INTERVAL_MINUTES")
    sentiment_interval_hours: int = Field(default=2, env="SENTIMENT_INTERVAL_HOURS")
    
    # Notification Settings
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    enable_notifications: bool = Field(default=False, env="ENABLE_NOTIFICATIONS")
    analysis_interval_minutes: int = Field(default=30, env="ANALYSIS_INTERVAL_MINUTES")
    sentiment_interval_hours: int = Field(default=2, env="SENTIMENT_INTERVAL_HOURS")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Performance Targets
    target_annual_return: float = Field(default=0.28, env="TARGET_ANNUAL_RETURN")
    target_sharpe_ratio: float = Field(default=1.8, env="TARGET_SHARPE_RATIO")
    target_win_rate: float = Field(default=0.65, env="TARGET_WIN_RATE")
    target_profit_factor: float = Field(default=2.2, env="TARGET_PROFIT_FACTOR")
    
    # Convenience aliases
    @property
    def API_HOST(self) -> str:
        return self.api_host
    
    @property
    def API_PORT(self) -> int:
        return self.api_port
    
    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level.upper()
    
    @property
    def ENVIRONMENT(self) -> str:
        return self.environment
    
    def validate_config(self):
        """Validate configuration settings."""
        # In development mode, we can skip API key validation
        if self.environment == "development":
            return True
        
        # In production, validate all required API keys
        required_keys = [
            self.binance_api_key,
            self.binance_secret_key,
            self.alpaca_api_key,
            self.alpaca_secret_key,
            self.perplexity_api_key
        ]
        
        if any(key.startswith("development_") for key in required_keys):
            raise ValueError("Production environment requires real API keys")
        
        return True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
