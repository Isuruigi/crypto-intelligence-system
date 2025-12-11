"""
Configuration Management for Crypto Intelligence System
Uses Pydantic settings for type-safe configuration
"""
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    GROQ_API_KEY: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "CryptoIntelligence/1.0"
    NEWS_API_KEY: str = ""
    
    # LangChain Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "crypto-intelligence"
    
    # Database
    REDIS_URL: str = "redis://localhost:6379"
    POSTGRES_URL: str = "postgresql://user:password@localhost:5432/crypto_intel"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Model Settings
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "mixtral-8x7b-32768"
    
    # Rate Limits
    GROQ_RATE_LIMIT: int = 30  # requests per minute
    REDDIT_RATE_LIMIT: int = 60  # requests per minute
    CCXT_RATE_LIMIT: int = 1200  # requests per minute
    BLOCKCHAIN_RATE_LIMIT: int = 20  # requests per minute
    NEWS_API_RATE_LIMIT: int = 100  # requests per day
    
    # Cache TTL (seconds)
    CACHE_WHALE_DATA: int = 60
    CACHE_ORDERBOOK: int = 30
    CACHE_SENTIMENT: int = 300
    CACHE_NEWS: int = 21600  # 6 hours
    CACHE_SIGNAL: int = 60
    
    # Default Exchange
    DEFAULT_EXCHANGE: str = "binance"
    FALLBACK_EXCHANGE: str = "coinbase"
    
    # Agent Configuration
    AGENT_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Signal Weights
    WEIGHT_SENTIMENT: float = 0.40
    WEIGHT_ONCHAIN: float = 0.30
    WEIGHT_WHALE: float = 0.20
    WEIGHT_NEWS: float = 0.10
    
    # Backtesting
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    DEFAULT_COMMISSION: float = 0.001
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
