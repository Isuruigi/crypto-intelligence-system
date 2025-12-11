"""
Signal Generation Agent for Crypto Intelligence System
Generates trading signals from multi-modal inputs
"""
from typing import Dict, Any
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.signal_model import SignalGenerator, SignalInput
from src.utils.logger import get_logger
from src.utils.metrics import timed

logger = get_logger(__name__)


class SignalAgent(BaseAgent):
    """
    Generates trading signals by combining inputs from all other agents
    
    Input Sources:
    - Social sentiment score
    - News sentiment score
    - On-chain/orderbook score
    - Whale accumulation score
    
    Output:
    - Signal type (BUY/SELL/HOLD)
    - Confidence score
    - Component breakdown
    """
    
    def __init__(self):
        """Initialize signal agent"""
        super().__init__("signal_generator")
        self.generator = SignalGenerator()
    
    @timed("signal_agent_generate")
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Generate trading signal from component scores
        
        Args:
            symbol: Trading pair
            **kwargs: Component scores:
                - sentiment: Social sentiment score (-1 to 1)
                - news: News sentiment score (-1 to 1)
                - onchain: On-chain score (-1 to 1)
                - whale: Whale accumulation score (-1 to 1)
                - fear_greed: Fear & Greed Index (0-100)
                - price: Current price
                
        Returns:
            Trading signal with confidence and breakdown
        """
        # Extract component scores from kwargs
        sentiment = kwargs.get("sentiment", 0.0)
        news = kwargs.get("news", 0.0)
        onchain = kwargs.get("onchain", 0.0)
        whale = kwargs.get("whale", 0.0)
        fear_greed = kwargs.get("fear_greed", 50)
        price = kwargs.get("price", 0.0)
        
        logger.info(
            "signal_generation_started",
            symbol=symbol,
            sentiment=sentiment,
            onchain=onchain,
            whale=whale
        )
        
        try:
            # Generate signal
            signal = self.generator.generate(
                sentiment=sentiment,
                news=news,
                onchain=onchain,
                whale=whale,
                fear_greed=fear_greed,
                symbol=symbol,
                price=price
            )
            
            result = {
                "success": True,
                "signal": signal.signal.value,
                "score": signal.score,
                "confidence": signal.confidence,
                "thesis": signal.thesis,
                "components": {
                    "sentiment": signal.components.sentiment,
                    "news": signal.components.news,
                    "onchain": signal.components.on_chain,
                    "whale": signal.components.whale
                },
                "symbol": symbol,
                "price": price,
                "timestamp": signal.timestamp.isoformat()
            }
            
            logger.info(
                "signal_generation_complete",
                symbol=symbol,
                signal=signal.signal.value,
                confidence=signal.confidence
            )
            
            return result
            
        except Exception as e:
            logger.error(f"signal_generation_error: {e}")
            return self._get_neutral_signal_result(symbol)
    
    def generate_from_agents(
        self,
        social_result: Dict[str, Any],
        news_result: Dict[str, Any],
        chain_result: Dict[str, Any],
        whale_result: Dict[str, Any],
        symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        Generate signal from agent results
        
        This is a synchronous method for direct use by the coordinator
        """
        # Extract scores from agent results
        sentiment = social_result.get("sentiment_score", 0.0)
        news = news_result.get("sentiment_score", 0.0)
        onchain = chain_result.get("chain_score", 0.0)
        whale = whale_result.get("signal", 0.0)
        fear_greed = chain_result.get("fear_greed_index", 50)
        price = chain_result.get("current_price", 0.0)
        
        # Generate signal
        signal = self.generator.generate(
            sentiment=sentiment,
            news=news,
            onchain=onchain,
            whale=whale,
            fear_greed=fear_greed,
            symbol=symbol,
            price=price
        )
        
        return {
            "success": True,
            "signal": signal.signal.value,
            "score": signal.score,
            "confidence": signal.confidence,
            "thesis": signal.thesis,
            "components": {
                "sentiment": signal.components.sentiment,
                "news": signal.components.news,
                "onchain": signal.components.on_chain,
                "whale": signal.components.whale
            },
            "symbol": symbol,
            "price": price,
            "fear_greed": fear_greed,
            "timestamp": signal.timestamp.isoformat()
        }
    
    def _get_neutral_signal_result(self, symbol: str) -> Dict[str, Any]:
        """Return neutral signal when generation fails"""
        return {
            "success": True,
            "is_fallback": True,
            "signal": "NEUTRAL",
            "score": 0.0,
            "confidence": 30.0,
            "thesis": "Unable to generate signal - insufficient data",
            "components": {
                "sentiment": 0.0,
                "news": 0.0,
                "onchain": 0.0,
                "whale": 0.0
            },
            "symbol": symbol,
            "price": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
