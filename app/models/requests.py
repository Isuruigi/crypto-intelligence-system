"""
Pydantic request models for API validation
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class SignalRequest(BaseModel):
    """Request model for generating trading signals"""
    
    symbol: str = Field(
        default="BTC/USDT",
        description="Trading pair symbol",
        min_length=3,
        max_length=20,
        examples=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    
    timeframe: Literal["1h", "4h", "24h"] = Field(
        default="4h",
        description="Analysis timeframe"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "timeframe": "4h"
            }
        }


class BacktestRequest(BaseModel):
    """Request model for running backtests"""
    
    symbol: str = Field(
        default="BTC/USDT",
        description="Trading pair symbol"
    )
    
    start_date: str = Field(
        description="Start date in YYYY-MM-DD format",
        examples=["2024-01-01"]
    )
    
    end_date: str = Field(
        description="End date in YYYY-MM-DD format",
        examples=["2024-11-23"]
    )
    
    initial_capital: float = Field(
        default=10000.0,
        description="Initial capital in USD",
        gt=0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "start_date": "2024-01-01",
                "end_date": "2024-11-23",
                "initial_capital": 10000.0
            }
        }
