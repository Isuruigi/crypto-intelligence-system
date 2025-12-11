"""
Signal Generation Model for Crypto Intelligence System
Combines multiple data sources into trading signals
"""
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from src.config import get_settings
from src.utils.logger import get_logger
from src.models.schemas import (
    SignalType, 
    TradingSignal, 
    SignalComponents,
    RiskLevel,
    RiskAssessment,
    RiskFactors
)

logger = get_logger(__name__)


@dataclass
class SignalInput:
    """Input data for signal generation"""
    sentiment_score: float  # -1 to 1
    onchain_score: float  # -1 to 1 (orderbook imbalance)
    whale_score: float  # -1 to 1 (accumulation/distribution)
    news_score: float  # -1 to 1
    fear_greed_index: int = 50  # 0-100
    price_change_24h: float = 0.0  # percentage
    volatility: float = 0.0  # percentage


class SignalGenerator:
    """
    Trading signal generator from multi-modal inputs
    
    Features:
    - Weighted combination of signals
    - Configurable thresholds
    - Divergence detection
    - Risk-adjusted outputs
    """
    
    def __init__(self):
        """Initialize signal generator with settings"""
        self.settings = get_settings()
        
        # Configurable weights
        self.weights = {
            "sentiment": self.settings.WEIGHT_SENTIMENT,  # 0.40
            "onchain": self.settings.WEIGHT_ONCHAIN,  # 0.30
            "whale": self.settings.WEIGHT_WHALE,  # 0.20
            "news": self.settings.WEIGHT_NEWS,  # 0.10
        }
        
        # Signal thresholds
        self.thresholds = {
            "strong_buy": 0.6,
            "buy": 0.25,
            "sell": -0.25,
            "strong_sell": -0.6,
        }
        
        logger.info("signal_generator_initialized", weights=self.weights)
    
    def generate(
        self,
        sentiment: float = 0.0,
        onchain: float = 0.0,
        whale: float = 0.0,
        news: float = 0.0,
        fear_greed: int = 50,
        symbol: str = "BTC/USDT",
        price: float = 0.0
    ) -> TradingSignal:
        """
        Generate trading signal from component scores
        
        Args:
            sentiment: Social sentiment score (-1 to 1)
            onchain: On-chain/orderbook score (-1 to 1) 
            whale: Whale activity score (-1 to 1)
            news: News sentiment score (-1 to 1)
            fear_greed: Fear & Greed Index (0-100)
            symbol: Trading pair
            price: Current price
            
        Returns:
            TradingSignal object
        """
        # Normalize inputs
        sentiment = self._normalize(sentiment)
        onchain = self._normalize(onchain)
        whale = self._normalize(whale)
        news = self._normalize(news)
        
        # Calculate weighted score
        score = (
            self.weights["sentiment"] * sentiment +
            self.weights["onchain"] * onchain +
            self.weights["whale"] * whale +
            self.weights["news"] * news
        )
        
        # Apply divergence bonus
        divergence_bonus = self._calculate_divergence_bonus(
            whale, sentiment, fear_greed
        )
        score = self._apply_divergence(score, divergence_bonus)
        
        # Classify signal
        signal_type = self._classify_signal(score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            sentiment, onchain, whale, news, score
        )
        
        # Generate thesis
        thesis = self._generate_thesis(
            signal_type, score, confidence,
            sentiment, onchain, whale, news,
            fear_greed
        )
        
        components = SignalComponents(
            sentiment=round(sentiment, 4),
            on_chain=round(onchain, 4),
            whale=round(whale, 4),
            news=round(news, 4)
        )
        
        return TradingSignal(
            signal=signal_type,
            score=round(score, 4),
            confidence=round(confidence, 2),
            components=components,
            symbol=symbol,
            price=price,
            thesis=thesis,
            timestamp=datetime.utcnow()
        )
    
    def generate_from_input(
        self,
        input_data: SignalInput,
        symbol: str = "BTC/USDT",
        price: float = 0.0
    ) -> TradingSignal:
        """Generate signal from SignalInput dataclass"""
        return self.generate(
            sentiment=input_data.sentiment_score,
            onchain=input_data.onchain_score,
            whale=input_data.whale_score,
            news=input_data.news_score,
            fear_greed=input_data.fear_greed_index,
            symbol=symbol,
            price=price
        )
    
    def _normalize(self, value: float) -> float:
        """Normalize value to -1 to 1 range"""
        return max(-1.0, min(1.0, value))
    
    def _classify_signal(self, score: float) -> SignalType:
        """Classify score into signal type"""
        if score >= self.thresholds["strong_buy"]:
            return SignalType.STRONG_BUY
        elif score >= self.thresholds["buy"]:
            return SignalType.BUY
        elif score <= self.thresholds["strong_sell"]:
            return SignalType.STRONG_SELL
        elif score <= self.thresholds["sell"]:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL
    
    def _calculate_divergence_bonus(
        self,
        whale: float,
        retail_sentiment: float,
        fear_greed: int
    ) -> float:
        """
        Calculate divergence bonus for contrarian signals
        
        The key insight: When whales are accumulating but retail is fearful,
        this often precedes upward moves (and vice versa).
        """
        # Normalize fear/greed to -1 to 1 scale
        fg_normalized = (fear_greed - 50) / 50
        
        # Calculate divergence between whale and retail
        # Positive divergence = whales bullish, retail bearish
        divergence = whale - retail_sentiment
        
        # Strong divergence (>0.5) provides signal boost
        if abs(divergence) > 0.5:
            if whale > 0 and retail_sentiment < 0:
                # Whales buying, retail fearful = bullish contrarian signal
                return 0.15
            elif whale < 0 and retail_sentiment > 0:
                # Whales selling, retail greedy = bearish contrarian signal
                return -0.15
        
        return 0.0
    
    def _apply_divergence(self, score: float, bonus: float) -> float:
        """Apply divergence bonus with capping"""
        new_score = score + bonus
        return max(-1.0, min(1.0, new_score))
    
    def _calculate_confidence(
        self,
        sentiment: float,
        onchain: float,
        whale: float,
        news: float,
        score: float
    ) -> float:
        """
        Calculate confidence score (0-100)
        
        Confidence is higher when:
        - All signals align (same direction)
        - Score magnitude is high
        - Individual signals are strong
        """
        signals = [sentiment, onchain, whale, news]
        
        # Check alignment (all positive or all negative)
        positive_count = sum(1 for s in signals if s > 0.1)
        negative_count = sum(1 for s in signals if s < -0.1)
        
        alignment = max(positive_count, negative_count) / len(signals)
        
        # Signal strength (average magnitude)
        avg_magnitude = np.mean([abs(s) for s in signals])
        
        # Score magnitude contribution
        score_contribution = abs(score) * 0.5
        
        # Calculate confidence
        confidence = (
            alignment * 40 +
            avg_magnitude * 30 +
            score_contribution * 30
        )
        
        return min(95, max(20, confidence))  # Cap between 20-95
    
    def _generate_thesis(
        self,
        signal: SignalType,
        score: float,
        confidence: float,
        sentiment: float,
        onchain: float,
        whale: float,
        news: float,
        fear_greed: int
    ) -> str:
        """Generate human-readable trading thesis"""
        # Determine main drivers
        drivers = []
        
        if abs(sentiment) > 0.3:
            direction = "bullish" if sentiment > 0 else "bearish"
            drivers.append(f"social sentiment is {direction} ({sentiment:.2f})")
        
        if abs(whale) > 0.3:
            action = "accumulating" if whale > 0 else "distributing"
            drivers.append(f"whales are {action}")
        
        if abs(onchain) > 0.3:
            imbalance = "buy pressure" if onchain > 0 else "sell pressure"
            drivers.append(f"orderbook shows {imbalance}")
        
        # Fear/Greed context
        if fear_greed < 25:
            fg_context = "extreme fear in the market"
        elif fear_greed < 45:
            fg_context = "fear in the market"
        elif fear_greed > 75:
            fg_context = "extreme greed in the market"
        elif fear_greed > 55:
            fg_context = "greed in the market"
        else:
            fg_context = "neutral market sentiment"
        
        # Build thesis
        if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            action = "bullish"
        elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            action = "bearish"
        else:
            action = "neutral"
        
        driver_text = ", ".join(drivers) if drivers else "mixed signals"
        
        thesis = (
            f"Signal is {action} (score: {score:.2f}, confidence: {confidence:.0f}%). "
            f"Key factors: {driver_text}. "
            f"Market context: {fg_context} (F&G: {fear_greed})."
        )
        
        return thesis


class RiskManager:
    """
    Risk assessment and position sizing
    """
    
    def __init__(self):
        """Initialize risk manager"""
        self.settings = get_settings()
        
        # Base position sizes by risk level
        self.position_sizes = {
            RiskLevel.LOW: 0.03,
            RiskLevel.MEDIUM: 0.02,
            RiskLevel.HIGH: 0.01,
            RiskLevel.VERY_HIGH: 0.005,
        }
        
        logger.info("risk_manager_initialized")
    
    def assess(
        self,
        signal: TradingSignal,
        volatility: float = 0.0,
        fear_greed: int = 50
    ) -> RiskAssessment:
        """
        Assess risk for a trading signal
        
        Args:
            signal: Trading signal to assess
            volatility: Current volatility percentage
            fear_greed: Fear & Greed Index
            
        Returns:
            RiskAssessment object
        """
        # Calculate individual risk factors
        volatility_risk = self._assess_volatility_risk(volatility)
        sentiment_risk = self._assess_sentiment_risk(fear_greed)
        confidence_risk = self._assess_confidence_risk(signal.confidence)
        
        factors = RiskFactors(
            volatility_risk=volatility_risk,
            sentiment_risk=sentiment_risk,
            liquidity_risk=0.2,  # Default moderate
            concentration_risk=0.3  # Default moderate
        )
        
        # Calculate overall risk score
        risk_score = (
            volatility_risk * 30 +
            sentiment_risk * 25 +
            confidence_risk * 25 +
            0.2 * 20  # Base risk
        ) * 100
        
        # Determine risk level
        risk_level = self._classify_risk(risk_score)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal, risk_level, risk_score
        )
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exit_levels(
            signal, volatility, risk_level
        )
        
        # Generate warnings
        warnings = self._generate_warnings(
            signal, volatility, fear_greed, risk_score
        )
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=round(risk_score, 2),
            factors=factors,
            recommended_position_size=round(position_size, 4),
            stop_loss_pct=round(stop_loss, 4),
            take_profit_pct=round(take_profit, 4),
            max_loss=round(position_size * stop_loss, 4),
            warnings=warnings
        )
    
    def _assess_volatility_risk(self, volatility: float) -> float:
        """Assess risk from volatility (0-1)"""
        if volatility > 5:
            return 0.9
        elif volatility > 3:
            return 0.7
        elif volatility > 2:
            return 0.5
        elif volatility > 1:
            return 0.3
        return 0.1
    
    def _assess_sentiment_risk(self, fear_greed: int) -> float:
        """Assess risk from extreme sentiment (0-1)"""
        if fear_greed < 20 or fear_greed > 80:
            return 0.8  # Extreme sentiment = high risk
        elif fear_greed < 35 or fear_greed > 65:
            return 0.5
        return 0.2
    
    def _assess_confidence_risk(self, confidence: float) -> float:
        """Assess risk from signal confidence (0-1)"""
        if confidence > 70:
            return 0.2
        elif confidence > 50:
            return 0.4
        elif confidence > 30:
            return 0.6
        return 0.8
    
    def _classify_risk(self, risk_score: float) -> RiskLevel:
        """Classify risk score into level"""
        if risk_score < 30:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 70:
            return RiskLevel.HIGH
        return RiskLevel.VERY_HIGH
    
    def _calculate_position_size(
        self,
        signal: TradingSignal,
        risk_level: RiskLevel,
        risk_score: float
    ) -> float:
        """Calculate recommended position size as fraction of portfolio"""
        base_size = self.position_sizes[risk_level]
        
        # Adjust based on confidence
        confidence_multiplier = signal.confidence / 100
        
        # Stronger signals allow larger positions
        signal_multiplier = 1.0
        if signal.signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            signal_multiplier = 1.2
        elif signal.signal == SignalType.NEUTRAL:
            signal_multiplier = 0.5
        
        position = base_size * confidence_multiplier * signal_multiplier
        
        # Cap at 5% of portfolio
        return min(0.05, max(0.005, position))
    
    def _calculate_exit_levels(
        self,
        signal: TradingSignal,
        volatility: float,
        risk_level: RiskLevel
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit percentages"""
        # Base levels
        base_stop_loss = 0.02  # 2%
        base_take_profit = 0.04  # 4%
        
        # Adjust for volatility
        volatility_adj = max(1.0, volatility / 2)
        
        # Risk level adjustment
        risk_adj = {
            RiskLevel.LOW: 0.8,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 1.2,
            RiskLevel.VERY_HIGH: 1.5
        }[risk_level]
        
        stop_loss = base_stop_loss * volatility_adj * risk_adj
        take_profit = base_take_profit * volatility_adj
        
        # Ensure take profit > stop loss
        take_profit = max(take_profit, stop_loss * 1.5)
        
        return (min(0.15, stop_loss), min(0.30, take_profit))
    
    def _generate_warnings(
        self,
        signal: TradingSignal,
        volatility: float,
        fear_greed: int,
        risk_score: float
    ) -> list:
        """Generate risk warnings"""
        warnings = []
        
        if volatility > 5:
            warnings.append("High volatility - consider smaller position size")
        
        if fear_greed < 20:
            warnings.append("Extreme fear - potential for sharp moves")
        elif fear_greed > 80:
            warnings.append("Extreme greed - potential for correction")
        
        if signal.confidence < 40:
            warnings.append("Low confidence signal - exercise caution")
        
        if risk_score > 70:
            warnings.append("High overall risk - consider waiting for better setup")
        
        return warnings


# Global instances
_signal_generator: Optional[SignalGenerator] = None
_risk_manager: Optional[RiskManager] = None


def get_signal_generator() -> SignalGenerator:
    """Get global signal generator instance"""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SignalGenerator()
    return _signal_generator


def get_risk_manager() -> RiskManager:
    """Get global risk manager instance"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager
