"""
Pydantic Models and Schemas for Crypto Intelligence System
Type-safe data models for all components
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================

class SignalType(str, Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    HOLD = "HOLD"


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Timeframe(str, Enum):
    """Trading timeframes"""
    MINUTES_15 = "15m"
    HOUR_1 = "1h"
    HOURS_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


# =============================================================================
# Base Models
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    class Config:
        from_attributes = True
        extra = "allow"


# =============================================================================
# Sentiment Models
# =============================================================================

class SentimentResult(BaseSchema):
    """Result of sentiment analysis for a single text"""
    text: str = Field(..., description="Analyzed text")
    sentiment: SentimentLabel = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Score breakdown by sentiment type"
    )
    compound_score: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Compound score from -1 to 1"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v):
        return round(v, 4)


class AggregatedSentiment(BaseSchema):
    """Aggregated sentiment from multiple texts"""
    overall_sentiment: SentimentLabel
    avg_confidence: float = Field(..., ge=0, le=1)
    weighted_compound: float = Field(..., ge=-1, le=1)
    distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each sentiment type"
    )
    num_texts: int = Field(..., ge=0)
    positive_pct: float = Field(..., ge=0, le=100)
    negative_pct: float = Field(..., ge=0, le=100)
    neutral_pct: float = Field(..., ge=0, le=100)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SocialSentiment(BaseSchema):
    """Social media sentiment analysis result"""
    overall_sentiment: SentimentLabel
    sentiment_score: float = Field(..., ge=-1, le=1)
    post_count: int = Field(..., ge=0)
    avg_sentiment: float = Field(..., ge=-1, le=1)
    trending_topics: Dict[str, int] = Field(default_factory=dict)
    engagement_score: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="reddit")


# =============================================================================
# Whale and On-Chain Models
# =============================================================================

class WhaleTransaction(BaseSchema):
    """Single whale transaction"""
    tx_hash: str
    amount: float
    currency: str
    from_address: str = Field(default="unknown")
    to_address: str = Field(default="unknown")
    timestamp: datetime
    usd_value: Optional[float] = None
    transaction_type: str = Field(default="transfer")


class WhaleMovement(BaseSchema):
    """Aggregated whale movement analysis"""
    net_flow: float = Field(
        ...,
        description="Positive = accumulation, Negative = distribution"
    )
    large_transfers: List[WhaleTransaction] = Field(default_factory=list)
    classification: str = Field(default="neutral")
    confidence: float = Field(default=0.5, ge=0, le=1)
    accumulation_score: float = Field(
        default=0.0,
        ge=-100,
        le=100,
        description="Score from -100 to +100"
    )
    total_volume: float = Field(default=0.0, ge=0)
    transaction_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OrderbookMetrics(BaseSchema):
    """Order book analysis metrics"""
    symbol: str
    imbalance: float = Field(..., ge=-1, le=1)
    bid_volume: float = Field(..., ge=0)
    ask_volume: float = Field(..., ge=0)
    spread: float = Field(..., ge=0)
    mid_price: float = Field(..., ge=0)
    market_depth_score: float = Field(default=50, ge=0, le=100)
    spoofing_detected: bool = Field(default=False)
    whale_walls: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Signal Models
# =============================================================================

class SignalComponents(BaseSchema):
    """Component scores that make up a trading signal"""
    sentiment: float = Field(default=0.0, ge=-1, le=1)
    on_chain: float = Field(default=0.0, ge=-1, le=1)
    whale: float = Field(default=0.0, ge=-1, le=1)
    news: float = Field(default=0.0, ge=-1, le=1)
    technical: float = Field(default=0.0, ge=-1, le=1)


class TradingSignal(BaseSchema):
    """Complete trading signal"""
    signal: SignalType = Field(..., description="Signal type")
    score: float = Field(..., ge=-1, le=1, description="Overall score")
    confidence: float = Field(..., ge=0, le=100, description="Confidence 0-100")
    components: SignalComponents = Field(default_factory=SignalComponents)
    symbol: str = Field(default="BTC/USDT")
    price: float = Field(default=0.0, ge=0)
    thesis: str = Field(default="", description="Human-readable explanation")
    timeframe: str = Field(default="4h")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    generated_at: str = Field(default="")
    
    @field_validator("generated_at", mode="before")
    @classmethod
    def set_generated_at(cls, v):
        return v or datetime.utcnow().isoformat() + "Z"


class HistoricalSignal(TradingSignal):
    """Historical signal with performance tracking"""
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    was_profitable: Optional[bool] = None
    duration_hours: Optional[float] = None


# =============================================================================
# Risk Models
# =============================================================================

class RiskFactors(BaseSchema):
    """Individual risk factors"""
    volatility_risk: float = Field(default=0.0, ge=0, le=1)
    liquidity_risk: float = Field(default=0.0, ge=0, le=1)
    sentiment_risk: float = Field(default=0.0, ge=0, le=1)
    concentration_risk: float = Field(default=0.0, ge=0, le=1)


class RiskAssessment(BaseSchema):
    """Complete risk assessment"""
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    risk_score: float = Field(default=50, ge=0, le=100)
    factors: RiskFactors = Field(default_factory=RiskFactors)
    recommended_position_size: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="Recommended position as fraction of portfolio"
    )
    stop_loss_pct: float = Field(default=0.05, ge=0, le=1)
    take_profit_pct: float = Field(default=0.10, ge=0, le=1)
    max_loss: float = Field(default=0.02, ge=0, le=1)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Analysis Result Models
# =============================================================================

class AgentResult(BaseSchema):
    """Result from a single agent"""
    agent_name: str
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalysisResult(BaseSchema):
    """Complete multi-agent analysis result"""
    signal: TradingSignal
    risk: RiskAssessment
    reasoning: str = Field(default="")
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
    llm_explanation: str = Field(default="")
    execution_time_ms: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# API Models
# =============================================================================

class SignalRequest(BaseSchema):
    """API request for signal generation"""
    symbol: str = Field(default="BTC/USDT", description="Trading pair")
    timeframe: str = Field(default="4h", description="Analysis timeframe")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    include_reasoning: bool = Field(default=True)


class SignalResponse(BaseSchema):
    """API response for signal generation"""
    success: bool = True
    signal: Optional[TradingSignal] = None
    risk: Optional[RiskAssessment] = None
    reasoning: str = Field(default="")
    error: Optional[str] = None
    execution_time_ms: float = Field(default=0.0)


class HealthResponse(BaseSchema):
    """API health check response"""
    status: str = Field(default="ok")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
    components: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Backtesting Models
# =============================================================================

class Trade(BaseSchema):
    """Single trade in backtesting"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    signal_type: SignalType
    duration_hours: float = Field(default=0.0)


class BacktestMetrics(BaseSchema):
    """Backtesting performance metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int = Field(default=0, description="Days")
    win_rate: float = Field(..., ge=0, le=1)
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float = Field(default=0.0, description="Hours")
    num_trades: int
    num_winning: int
    num_losing: int


class BacktestResult(BaseSchema):
    """Complete backtesting result"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics
    trades: List[Trade] = Field(default_factory=list)
    equity_curve: List[float] = Field(default_factory=list)
