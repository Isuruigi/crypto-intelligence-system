"""
Configuration management using pydantic-settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Groq API Configuration
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    
    # Reddit Configuration
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "CryptoIntelligenceBot/1.0"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/signals"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Rate Limits (requests per minute)
    REDDIT_RATE_LIMIT: int = 60
    CCXT_RATE_LIMIT: int = 1200
    BLOCKCHAIN_RATE_LIMIT: int = 20
    
    # Cache TTL (seconds)
    CACHE_WHALE_DATA: int = 60
    CACHE_ORDERBOOK: int = 30
    CACHE_SENTIMENT: int = 300
    
    # Application Configuration
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    DEMO_MODE: bool = True  # Skip slow API calls, use fast mock data
    
    # Exchange Configuration
    DEFAULT_EXCHANGE: str = "binance"
    DEFAULT_SYMBOL: str = "BTC/USDT"
    
    # Signal Generation Parameters
    MIN_CONFIDENCE_THRESHOLD: int = 30
    WHALE_TRANSACTION_MIN_USD: float = 1000000.0
    ORDERBOOK_DEPTH_LEVELS: int = 20
    SENTIMENT_LOOKBACK_HOURS: int = 24
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
