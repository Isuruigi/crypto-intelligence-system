"""
On-Chain Analysis Agent for Crypto Intelligence System
Analyzes exchange orderbook data and on-chain metrics
"""
from typing import Dict, Any
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.data.collectors.chain_collector import OnChainCollector
from src.utils.logger import get_logger
from src.utils.metrics import timed

logger = get_logger(__name__)


class ChainAgent(BaseAgent):
    """
    Analyzes on-chain and exchange data
    
    Data Sources:
    - CCXT (orderbook, price, volume)
    - Fear & Greed Index
    
    Output:
    - Orderbook imbalance
    - Market depth
    - Technical indicators
    - Fear & Greed Index
    """
    
    def __init__(self, exchange: str = None):
        """
        Initialize on-chain agent
        
        Args:
            exchange: Exchange to use (default: from settings)
        """
        super().__init__("chain_analysis")
        self.collector = OnChainCollector(exchange=exchange)
    
    @timed("chain_agent_analyze")
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Analyze on-chain and exchange data for the given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            On-chain analysis results
        """
        logger.info("chain_analysis_started", symbol=symbol)
        
        try:
            # Get orderbook data
            orderbook = await self.collector.get_orderbook_data(symbol)
            
            # Get price data with indicators
            price_df = await self.collector.get_price_data(
                symbol, 
                timeframe="1h",
                limit=24
            )
            
            # Get Fear & Greed Index
            fear_greed = await self.collector.get_fear_greed_index()
            
            # Get market metrics
            metrics = await self.collector.get_market_metrics(symbol)
            
            # Calculate technical signals
            tech_signals = self._analyze_technicals(price_df)
            
            # Convert imbalance to signal score (-1 to 1)
            imbalance_signal = self._imbalance_to_signal(orderbook.imbalance)
            
            result = {
                "success": True,
                "symbol": symbol,
                
                # Orderbook metrics
                "orderbook_imbalance": orderbook.imbalance,
                "imbalance_signal": imbalance_signal,
                "bid_volume": orderbook.bid_volume,
                "ask_volume": orderbook.ask_volume,
                "spread": orderbook.spread,
                "mid_price": orderbook.mid_price,
                "whale_walls": orderbook.whale_walls[:5],
                "market_depth": orderbook.depth_levels,
                
                # Price and technicals
                "current_price": metrics.price,
                "price_change_24h": metrics.price_change_24h,
                "volume_24h": metrics.volume_24h,
                "volatility": metrics.volatility,
                "technical_signals": tech_signals,
                
                # Market sentiment
                "fear_greed_index": fear_greed,
                "fear_greed_label": self._interpret_fear_greed(fear_greed),
                
                # Combined score
                "chain_score": self._calculate_chain_score(
                    orderbook.imbalance,
                    fear_greed,
                    tech_signals
                ),
                
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "chain_analysis_complete",
                symbol=symbol,
                imbalance=orderbook.imbalance,
                fear_greed=fear_greed
            )
            
            return result
            
        except Exception as e:
            logger.error(f"chain_analysis_error: {e}")
            return self._get_neutral_chain_result(symbol)
        finally:
            await self.collector.close()
    
    def _imbalance_to_signal(self, imbalance: float) -> float:
        """
        Convert orderbook imbalance to signal score
        
        Positive imbalance (more bids) = bullish
        Negative imbalance (more asks) = bearish
        """
        # Already in -1 to 1 range, but we can apply non-linear scaling
        # Stronger imbalances get amplified
        if abs(imbalance) > 0.5:
            return imbalance * 1.2  # Amplify strong signals
        return imbalance
    
    def _analyze_technicals(self, price_df) -> Dict[str, Any]:
        """Analyze technical indicators from price data"""
        signals = {
            "rsi_signal": "neutral",
            "macd_signal": "neutral",
            "trend_signal": "neutral",
            "rsi_value": None,
            "macd_value": None
        }
        
        if price_df.empty:
            return signals
        
        try:
            # Get latest values
            latest = price_df.iloc[-1]
            
            # RSI analysis
            rsi = latest.get("rsi")
            if rsi is not None:
                signals["rsi_value"] = round(rsi, 2)
                if rsi < 30:
                    signals["rsi_signal"] = "oversold"  # Bullish
                elif rsi > 70:
                    signals["rsi_signal"] = "overbought"  # Bearish
            
            # MACD analysis
            macd = latest.get("macd")
            macd_signal = latest.get("macd_signal")
            if macd is not None and macd_signal is not None:
                signals["macd_value"] = round(macd, 4)
                if macd > macd_signal:
                    signals["macd_signal"] = "bullish"
                else:
                    signals["macd_signal"] = "bearish"
            
            # Simple trend analysis
            sma_20 = latest.get("sma_20")
            close = latest.get("close")
            if sma_20 is not None and close is not None:
                if close > sma_20 * 1.02:
                    signals["trend_signal"] = "bullish"
                elif close < sma_20 * 0.98:
                    signals["trend_signal"] = "bearish"
                    
        except Exception as e:
            logger.debug(f"technical_analysis_error: {e}")
        
        return signals
    
    def _interpret_fear_greed(self, index: int) -> str:
        """Convert Fear & Greed Index to label"""
        if index < 20:
            return "extreme_fear"
        elif index < 40:
            return "fear"
        elif index < 60:
            return "neutral"
        elif index < 80:
            return "greed"
        return "extreme_greed"
    
    def _calculate_chain_score(
        self,
        imbalance: float,
        fear_greed: int,
        technicals: Dict[str, Any]
    ) -> float:
        """
        Calculate combined on-chain score (-1 to 1)
        
        Components:
        - Orderbook imbalance
        - Fear/Greed (contrarian)
        - Technical indicators
        """
        scores = []
        
        # Orderbook imbalance (40% weight)
        scores.append(imbalance * 0.4)
        
        # Fear/Greed as contrarian indicator (30% weight)
        # When fear is high, it's often a buy signal
        fg_normalized = (50 - fear_greed) / 50  # Inverted: fear = positive
        scores.append(fg_normalized * 0.3)
        
        # Technical indicators (30% weight)
        tech_score = 0.0
        
        rsi_signal = technicals.get("rsi_signal", "neutral")
        if rsi_signal == "oversold":
            tech_score += 0.3
        elif rsi_signal == "overbought":
            tech_score -= 0.3
        
        macd_signal = technicals.get("macd_signal", "neutral")
        if macd_signal == "bullish":
            tech_score += 0.2
        elif macd_signal == "bearish":
            tech_score -= 0.2
        
        scores.append(tech_score * 0.3)
        
        return round(sum(scores), 4)
    
    def _get_neutral_chain_result(self, symbol: str) -> Dict[str, Any]:
        """Return neutral result when no data available"""
        return {
            "success": True,
            "is_fallback": True,
            "symbol": symbol,
            "orderbook_imbalance": 0.0,
            "imbalance_signal": 0.0,
            "bid_volume": 0,
            "ask_volume": 0,
            "spread": 0,
            "mid_price": 0,
            "whale_walls": [],
            "market_depth": {},
            "current_price": 0,
            "price_change_24h": 0,
            "volume_24h": 0,
            "volatility": 0,
            "technical_signals": {
                "rsi_signal": "neutral",
                "macd_signal": "neutral",
                "trend_signal": "neutral"
            },
            "fear_greed_index": 50,
            "fear_greed_label": "neutral",
            "chain_score": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
