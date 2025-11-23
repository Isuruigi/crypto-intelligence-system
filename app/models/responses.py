"""
Pydantic response models for API validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime


class SignalResponse(BaseModel):
    """Response model for trading signals"""
    
    signal: str = Field(
        description="Trading signal: BUY, SELL, or HOLD",
        examples=["BUY", "SELL", "HOLD"]
    )
    
    confidence: int = Field(
        description="Confidence level (0-100)",
        ge=0,
        le=100
    )
    
    timeframe: str = Field(
        description="Expected timeframe for signal",
        examples=["2-6 hours", "6-24 hours", "24-48 hours"]
    )
    
    thesis: str = Field(
        description="Trading thesis and reasoning"
    )
    
    key_factors: List[str] = Field(
        description="Key factors supporting the signal"
    )
    
    risk_factors: List[str] = Field(
        description="Risk factors to consider"
    )
    
    historical_pattern: str = Field(
        description="Similar historical patterns"
    )
    
    position_sizing: str = Field(
        description="Recommended position sizing"
    )
    
    symbol: str = Field(
        description="Trading pair symbol"
    )
    
    price: float = Field(
        description="Current market price"
    )
    
    generated_at: str = Field(
        description="Signal generation timestamp (ISO 8601)"
    )
    
    underlying_data: Dict[str, Any] = Field(
        description="Underlying data from all agents"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "signal": "BUY",
                "confidence": 75,
                "timeframe": "6-24 hours",
                "thesis": "Strong whale accumulation with retail fear creates bullish divergence",
                "key_factors": [
                    "Whale accumulation score: +85",
                    "Retail sentiment: Extreme fear (15/100)",
                    "Order book shows strong support"
                ],
                "risk_factors": [
                    "High volatility expected",
                    "Macro uncertainty"
                ],
                "historical_pattern": "Similar to March 2024 bottom",
                "position_sizing": "Conservative: 2% | Moderate: 5% | Aggressive: 10%",
                "symbol": "BTC/USDT",
                "price": 97500.50,
                "generated_at": "2024-11-23T13:30:00Z",
                "underlying_data": {}
            }
        }


class HistoricalSignalsResponse(BaseModel):
    """Response model for historical signals"""
    
    signals: List[Dict[str, Any]] = Field(
        description="List of historical signals"
    )
    
    count: int = Field(
        description="Number of signals returned"
    )
    
    symbol: str = Field(
        default=None,
        description="Filtered symbol (if applicable)"
    )


class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    
    symbol: str = Field(description="Trading pair symbol")
    start_date: str = Field(description="Backtest start date")
    end_date: str = Field(description="Backtest end date")
    
    # Performance metrics
    total_trades: int = Field(description="Total number of trades")
    win_rate: float = Field(description="Win rate (0-1)")
    profit_factor: float = Field(description="Profit factor (gross profit / gross loss)")
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown")
    total_return: float = Field(description="Total return (decimal)")
    
    avg_win: float = Field(description="Average winning trade")
    avg_loss: float = Field(description="Average losing trade")
    
    initial_capital: float = Field(description="Starting capital")
    final_capital: float = Field(description="Ending capital")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "start_date": "2024-01-01",
                "end_date": "2024-11-23",
                "total_trades": 45,
                "win_rate": 0.72,
                "profit_factor": 2.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.15,
                "total_return": 0.45,
                "avg_win": 0.08,
                "avg_loss": -0.03,
                "initial_capital": 10000.0,
                "final_capital": 14500.0
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
