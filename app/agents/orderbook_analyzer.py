"""
Orderbook Analyzer Agent - Analyzes order book microstructure
Uses CCXT library for exchange data
"""
import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.services.rate_limiter import get_rate_limiter
from app.services.cache_service import get_cache
from app.utils.circuit_breaker import CircuitBreaker


class OrderbookAnalyzer(BaseAgent):
    """
    Analyzes order book microstructure for market depth and imbalance
    
    Features:
    - Bid/ask imbalance calculation
    - Market depth measurement
    - Spoofing detection
    - Support/resistance level identification
    """
    
    def __init__(self):
        super().__init__("orderbook_analyzer")
        self.settings = get_settings()
        self.rate_limiter = get_rate_limiter('ccxt')
        self.cache = get_cache()
        self.circuit_breaker = CircuitBreaker("exchange_api", failure_threshold=3, timeout=60)
        self.orderbook_history: List[Dict] = []  # For spoofing detection
    
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Analyze order book for the given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Order book analysis
        """
        # Try cache first
        cache_key = f"orderbook:{symbol}"
        cached = await self.cache.get(cache_key)
        if cached:
            self.logger.info('orderbook_from_cache', symbol=symbol)
            return cached
        
        try:
            # Fetch orderbook
            orderbook = await self._fetch_orderbook(symbol)
            
            # Calculate metrics
            analysis = self._analyze_orderbook(orderbook, symbol)
            
            # Store for spoofing detection
            self.orderbook_history.append(orderbook)
            if len(self.orderbook_history) > 10:
                self.orderbook_history.pop(0)
            
            # Detect spoofing
            analysis['spoofing_detected'] = self._detect_spoofing()
            
            # Cache the result
            await self.cache.set(cache_key, analysis, self.settings.CACHE_ORDERBOOK)
            
            self.logger.info(
                'orderbook_analysis_complete',
                symbol=symbol,
                imbalance=analysis['imbalance']
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error_with_context(
                e,
                {'agent': self.name, 'symbol': symbol}
            )
            return self._get_neutral_data()
    
    async def _fetch_orderbook(self, symbol: str, exchange_name: str = None) -> Dict[str, Any]:
        """
        Fetch order book from exchange
        
        Args:
            symbol: Trading pair
            exchange_name: Exchange to use (default: from config)
            
        Returns:
            Order book data
        """
        await self.rate_limiter.acquire()
        
        if exchange_name is None:
            exchange_name = self.settings.DEFAULT_EXCHANGE
        
        exchange = None
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 10000
            })
            
            # Fetch order book
            orderbook = await self.circuit_breaker.call(
                exchange.fetch_order_book,
                symbol,
                limit=self.settings.ORDERBOOK_DEPTH_LEVELS * 2  # Fetch extra to ensure depth
            )
            
            return orderbook
            
        except Exception as e:
            self.logger.error('orderbook_fetch_error', symbol=symbol, error=str(e))
            raise
        finally:
            if exchange:
                await exchange.close()
    
    def _analyze_orderbook(self, orderbook: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Analyze order book data to calculate metrics
        
        Args:
            orderbook: Raw orderbook data from exchange
            symbol: Trading pair
            
        Returns:
            Analysis results
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return self._get_neutral_data()
        
        # Limit to configured depth levels
        depth = self.settings.ORDERBOOK_DEPTH_LEVELS
        bids = bids[:depth]
        asks = asks[:depth]
        
        # Calculate volumes
        bid_volume = sum(price * amount for price, amount in bids)
        ask_volume = sum(price * amount for price, amount in asks)
        
        # Calculate imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate spread
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 0
        
        # Calculate market depth score (0-100)
        # Higher volume and tighter spread = better liquidity
        depth_score = min(100, (total_volume / 1000000) * 50 + (1 / max(spread_pct, 0.01)) * 10)
        
        # Identify support and resistance levels
        # Support = price levels with large bids
        support_levels = self._find_key_levels(bids, is_support=True)
        resistance_levels = self._find_key_levels(asks, is_support=False)
        
        return {
            'imbalance': round(imbalance, 4),
            'bid_volume': round(bid_volume, 2),
            'ask_volume': round(ask_volume, 2),
            'spread': round(spread_pct, 4),
            'market_depth_score': round(depth_score, 2),
            'spoofing_detected': False,  # Updated by _detect_spoofing
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def _find_key_levels(self, orders: List[List[float]], is_support: bool, threshold: float = 1.5) -> List[float]:
        """
        Find key price levels with significant volume
        
        Args:
            orders: List of [price, amount] pairs
            is_support: True for bids (support), False for asks (resistance)
            threshold: Volume threshold multiplier
            
        Returns:
            List of key price levels
        """
        if not orders:
            return []
        
        # Calculate average volume
        avg_volume = sum(amount for _, amount in orders) / len(orders)
        
        # Find levels with volume > threshold * average
        key_levels = [
            price for price, amount in orders
            if amount > threshold * avg_volume
        ]
        
        # Return top 3 levels
        return sorted(key_levels, reverse=is_support)[:3]
    
    def _detect_spoofing(self) -> bool:
        """
        Detect spoofing patterns by comparing recent orderbook snapshots
        
        Spoofing: Large orders that appear and disappear quickly
        
        Returns:
            True if spoofing detected
        """
        if len(self.orderbook_history) < 3:
            return False
        
        # Compare last 3 snapshots
        recent = self.orderbook_history[-3:]
        
        # Look for large orders that disappeared
        for i in range(len(recent) - 1):
            current_bids = set((price, amount) for price, amount in recent[i].get('bids', [])[:5])
            next_bids = set((price, amount) for price, amount in recent[i+1].get('bids', [])[:5])
            
            disappeared = current_bids - next_bids
            
            # If large orders disappeared, potential spoofing
            if any(amount > 10 for _, amount in disappeared):
                self.logger.warning('potential_spoofing_detected', disappeared_orders=len(disappeared))
                return True
        
        return False
    
    def _get_neutral_data(self) -> Dict[str, Any]:
        """Return neutral orderbook data when no data available"""
        return {
            'imbalance': 0.0,
            'bid_volume': 0.0,
            'ask_volume': 0.0,
            'spread': 0.0,
            'market_depth_score': 50.0,
            'spoofing_detected': False,
            'support_levels': [],
            'resistance_levels': [],
            'best_bid': 0.0,
            'best_ask': 0.0,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
