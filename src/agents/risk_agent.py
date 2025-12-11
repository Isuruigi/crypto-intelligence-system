"""
Risk Assessment Agent for Crypto Intelligence System
Evaluates risk and recommends position sizing
"""
from typing import Dict, Any
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.signal_model import get_risk_manager
from src.models.schemas import TradingSignal, SignalType, SignalComponents
from src.utils.logger import get_logger
from src.utils.metrics import timed

logger = get_logger(__name__)


class RiskAgent(BaseAgent):
    """
    Assesses risk and provides position sizing recommendations
    
    Evaluates:
    - Market volatility
    - Sentiment extremes
    - Signal confidence
    - Portfolio concentration
    
    Output:
    - Risk level classification
    - Recommended position size
    - Stop loss / take profit levels
    - Warning alerts
    """
    
    def __init__(self):
        """Initialize risk agent"""
        super().__init__("risk_assessment")
        self.risk_manager = get_risk_manager()
    
    @timed("risk_agent_analyze")
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Assess risk for a trading decision
        
        Args:
            symbol: Trading pair
            **kwargs: Risk assessment inputs:
                - signal: Signal dict from signal agent
                - volatility: Current volatility %
                - fear_greed: Fear & Greed Index
                
        Returns:
            Risk assessment with recommendations
        """
        # Extract inputs
        signal_data = kwargs.get("signal", {})
        volatility = kwargs.get("volatility", 2.0)
        fear_greed = kwargs.get("fear_greed", 50)
        
        logger.info(
            "risk_assessment_started",
            symbol=symbol,
            volatility=volatility,
            fear_greed=fear_greed
        )
        
        try:
            # Convert signal dict to TradingSignal
            signal_type_str = signal_data.get("signal", "NEUTRAL")
            
            # Handle both string and enum
            if isinstance(signal_type_str, str):
                try:
                    signal_type = SignalType(signal_type_str)
                except ValueError:
                    signal_type = SignalType.NEUTRAL
            else:
                signal_type = signal_type_str
            
            components = signal_data.get("components", {})
            
            signal = TradingSignal(
                signal=signal_type,
                score=signal_data.get("score", 0.0),
                confidence=signal_data.get("confidence", 50.0),
                components=SignalComponents(
                    sentiment=components.get("sentiment", 0.0),
                    on_chain=components.get("onchain", 0.0),
                    whale=components.get("whale", 0.0),
                    news=components.get("news", 0.0)
                ),
                symbol=symbol,
                price=signal_data.get("price", 0.0)
            )
            
            # Perform risk assessment
            assessment = self.risk_manager.assess(
                signal=signal,
                volatility=volatility,
                fear_greed=fear_greed
            )
            
            result = {
                "success": True,
                "symbol": symbol,
                
                # Risk classification
                "risk_level": assessment.risk_level.value,
                "risk_score": assessment.risk_score,
                
                # Position sizing
                "recommended_position_size": assessment.recommended_position_size,
                "position_size_pct": round(assessment.recommended_position_size * 100, 2),
                
                # Exit levels
                "stop_loss_pct": round(assessment.stop_loss_pct * 100, 2),
                "take_profit_pct": round(assessment.take_profit_pct * 100, 2),
                "max_loss_pct": round(assessment.max_loss * 100, 2),
                
                # Risk factors
                "risk_factors": {
                    "volatility_risk": assessment.factors.volatility_risk,
                    "sentiment_risk": assessment.factors.sentiment_risk,
                    "liquidity_risk": assessment.factors.liquidity_risk,
                    "concentration_risk": assessment.factors.concentration_risk
                },
                
                # Warnings
                "warnings": assessment.warnings,
                "warning_count": len(assessment.warnings),
                
                # Context
                "volatility": volatility,
                "fear_greed": fear_greed,
                "signal_confidence": signal.confidence,
                
                "timestamp": assessment.timestamp.isoformat()
            }
            
            logger.info(
                "risk_assessment_complete",
                symbol=symbol,
                risk_level=assessment.risk_level.value,
                position_size=assessment.recommended_position_size
            )
            
            return result
            
        except Exception as e:
            logger.error(f"risk_assessment_error: {e}")
            return self._get_default_risk_result(symbol)
    
    def assess_from_signal(
        self,
        signal_result: Dict[str, Any],
        chain_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronous risk assessment from agent results
        
        Used directly by the coordinator
        """
        volatility = chain_result.get("volatility", 2.0)
        fear_greed = chain_result.get("fear_greed_index", 50)
        
        # Build signal object
        signal_type_str = signal_result.get("signal", "NEUTRAL")
        try:
            signal_type = SignalType(signal_type_str)
        except ValueError:
            signal_type = SignalType.NEUTRAL
        
        components = signal_result.get("components", {})
        
        signal = TradingSignal(
            signal=signal_type,
            score=signal_result.get("score", 0.0),
            confidence=signal_result.get("confidence", 50.0),
            components=SignalComponents(
                sentiment=components.get("sentiment", 0.0),
                on_chain=components.get("onchain", 0.0),
                whale=components.get("whale", 0.0),
                news=components.get("news", 0.0)
            ),
            symbol=signal_result.get("symbol", "BTC/USDT"),
            price=signal_result.get("price", 0.0)
        )
        
        # Assess
        assessment = self.risk_manager.assess(
            signal=signal,
            volatility=volatility,
            fear_greed=fear_greed
        )
        
        return {
            "success": True,
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.risk_score,
            "recommended_position_size": assessment.recommended_position_size,
            "stop_loss_pct": round(assessment.stop_loss_pct * 100, 2),
            "take_profit_pct": round(assessment.take_profit_pct * 100, 2),
            "warnings": assessment.warnings,
            "timestamp": assessment.timestamp.isoformat()
        }
    
    def _get_default_risk_result(self, symbol: str) -> Dict[str, Any]:
        """Return conservative default risk assessment"""
        return {
            "success": True,
            "is_fallback": True,
            "symbol": symbol,
            "risk_level": "high",
            "risk_score": 70.0,
            "recommended_position_size": 0.01,
            "position_size_pct": 1.0,
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "max_loss_pct": 0.5,
            "risk_factors": {
                "volatility_risk": 0.5,
                "sentiment_risk": 0.5,
                "liquidity_risk": 0.3,
                "concentration_risk": 0.3
            },
            "warnings": ["Using default conservative risk assessment"],
            "warning_count": 1,
            "timestamp": datetime.utcnow().isoformat()
        }
