"""
Multi-Agent Orchestrator - Coordinates all agents and generates final signals
"""
import asyncio
from typing import Dict, Any
from datetime import datetime
import ccxt.async_support as ccxt

from app.agents.whale_tracker import WhaleTrackerAgent
from app.agents.orderbook_analyzer import OrderbookAnalyzer
from app.agents.sentiment_analyzer import SentimentAnalyzer
from app.agents.coordinator import LLMCoordinator
from app.services.cache_service import get_cache
from app.services.database_service import get_db
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates all agents to generate comprehensive trading signals
    
    Pipeline:
    1. Gather data from all agents concurrently
    2. Synthesize signals with LLM coordinator
    3. Store in database
    4. Return complete signal package
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize all agents
        self.whale_tracker = WhaleTrackerAgent()
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.coordinator = LLMCoordinator()
        
        # Services
        self.cache = get_cache()
        self.db = get_db()
        
        logger.info('orchestrator_initialized')
    
    async def generate_signal(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Generate comprehensive trading signal for given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Complete trading signal with analysis from all agents
        """
        start_time = datetime.now()
        logger.info('signal_generation_started', symbol=symbol)
        
        try:
            # Step 1: Gather data from all agents concurrently
            logger.info('gathering_agent_data', symbol=symbol)
            
            whale_data, orderbook_data, sentiment_data = await asyncio.gather(
                self._get_whale_data(symbol),
                self._get_orderbook_data(symbol),
                self._get_sentiment_data(symbol),
                return_exceptions=True
            )
            
            # Handle any exceptions from agents
            if isinstance(whale_data, Exception):
                logger.error('whale_agent_failed', error=str(whale_data))
                whale_data = {'whale_score': 0, 'sentiment': 'NEUTRAL', 'signal': 'HOLD', 'confidence': 0}
            
            if isinstance(orderbook_data, Exception):
                logger.error('orderbook_agent_failed', error=str(orderbook_data))
                orderbook_data = {'imbalance': 0, 'market_depth_score': 50, 'spoofing_detected': False}
            
            if isinstance(sentiment_data, Exception):
                logger.error('sentiment_agent_failed', error=str(sentiment_data))
                sentiment_data = {'sentiment_score': 0, 'fear_greed_index': 50, 'dominant_emotion': 'neutral'}
            
            # Step 2: Get current price
            current_price = await self._get_current_price(symbol)
            
            # Step 3: LLM coordination
            logger.info('coordinating_with_llm', symbol=symbol)
            
            signal = await self.coordinator.generate_trading_thesis(
                retail_sentiment=sentiment_data,
                whale_sentiment=whale_data,
                orderbook=orderbook_data,
                current_price=current_price,
                symbol=symbol.split('/')[0]
            )
            
            # Step 4: Add metadata
            signal['symbol'] = symbol
            signal['price'] = current_price
            signal['generated_at'] = datetime.utcnow().isoformat() + 'Z'
            
            # Step 5: Store in database
            await self.db.store_signal(signal)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                'signal_generation_complete',
                symbol=symbol,
                signal=signal['signal'],
                confidence=signal['confidence'],
                execution_time_seconds=round(execution_time, 2)
            )
            
            return signal
            
        except Exception as e:
            logger.error_with_context(
                e,
                {'operation': 'generate_signal', 'symbol': symbol}
            )
            raise
    
    async def _get_whale_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch whale tracking data"""
        return await self.whale_tracker.analyze(symbol=symbol)
    
    async def _get_orderbook_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch orderbook analysis"""
        return await self.orderbook_analyzer.analyze(symbol=symbol)
    
    async def _get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch sentiment analysis"""
        return await self.sentiment_analyzer.analyze(symbol=symbol)
    
    async def _get_current_price(self, symbol: str) -> float:
        """
        Fetch current market price
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price
        """
        exchange = None
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, self.settings.DEFAULT_EXCHANGE)
            exchange = exchange_class({'enableRateLimit': True})
            
            # Fetch ticker
            ticker = await exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            logger.info('price_fetched', symbol=symbol, price=price)
            return price
            
        except Exception as e:
            logger.error('price_fetch_error', symbol=symbol, error=str(e))
            # Return fallback price
            return 97000.0 if 'BTC' in symbol else 3500.0
        finally:
            if exchange:
                await exchange.close()
    
    async def get_historical_signals(
        self,
        symbol: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve historical signals from database
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of signals
            
        Returns:
            Historical signals
        """
        signals = await self.db.get_signals(symbol=symbol, limit=limit)
        
        return {
            'signals': signals,
            'count': len(signals),
            'symbol': symbol
        }


# Global orchestrator instance
_orchestrator_instance: MultiAgentOrchestrator = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiAgentOrchestrator()
    return _orchestrator_instance
