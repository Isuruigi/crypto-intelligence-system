"""
On-Chain Data Collector for Crypto Intelligence System
Collects exchange data, orderbook analysis, and market metrics using CCXT
"""
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import timed, get_metrics

logger = get_logger(__name__)


@dataclass
class OrderbookData:
    """Schema for orderbook data"""
    symbol: str
    timestamp: datetime
    bid_volume: float  # Sum of top bids
    ask_volume: float  # Sum of top asks
    imbalance: float  # (bid - ask) / (bid + ask), range -1 to 1
    spread: float  # Percentage spread
    mid_price: float
    best_bid: float
    best_ask: float
    whale_walls: List[Dict[str, Any]] = field(default_factory=list)
    depth_levels: Dict[str, float] = field(default_factory=dict)


@dataclass
class PriceData:
    """Schema for OHLCV price data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    sma_20: Optional[float] = None
    ema_12: Optional[float] = None


@dataclass
class MarketMetrics:
    """Schema for aggregated market metrics"""
    symbol: str
    timestamp: datetime
    price: float
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float] = None
    fear_greed_index: int = 50
    orderbook_imbalance: float = 0.0
    volatility: float = 0.0


class OnChainCollector:
    """
    On-chain and exchange data collector using CCXT
    
    Data Sources:
    - CCXT for exchange data (Binance, Coinbase, Kraken)
    - CoinGecko API for market data
    - Alternative.me for Fear & Greed Index
    
    Features:
    - Orderbook analysis
    - Price/volume data
    - Technical indicators
    - Whale wall detection
    """
    
    SUPPORTED_EXCHANGES = ["binance", "coinbase", "kraken", "bybit"]
    
    # Whale wall thresholds (in base currency units)
    WHALE_THRESHOLDS = {
        "BTC": 10,    # 10 BTC
        "ETH": 100,   # 100 ETH
        "SOL": 1000,  # 1000 SOL
        "DEFAULT": 10000  # $10k equivalent
    }
    
    def __init__(self, exchange: str = None):
        """
        Initialize On-Chain collector
        
        Args:
            exchange: Exchange to use (default: from settings)
        """
        self.settings = get_settings()
        self.exchange_name = exchange or self.settings.DEFAULT_EXCHANGE
        self._exchange: Optional[Any] = None
        
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = {
            "orderbook": 30,
            "price": 60,
            "fear_greed": 3600,
        }
        
        logger.info(f"chain_collector_initialized", exchange=self.exchange_name)
    
    async def _get_exchange(self) -> Optional[Any]:
        """Get or create exchange client"""
        if not CCXT_AVAILABLE:
            logger.warning("ccxt_not_available")
            return None
        
        if self._exchange is None:
            try:
                exchange_class = getattr(ccxt, self.exchange_name)
                self._exchange = exchange_class({
                    "enableRateLimit": True,
                    "timeout": 30000,
                })
                logger.info(f"exchange_client_created", exchange=self.exchange_name)
            except Exception as e:
                logger.error(f"exchange_client_error: {e}")
                return None
        
        return self._exchange
    
    async def close(self) -> None:
        """Close exchange connection"""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
    
    def _get_cache_key(self, data_type: str, symbol: str) -> str:
        """Generate cache key"""
        return f"{data_type}:{symbol}"
    
    def _is_cache_valid(self, key: str, ttl: int) -> bool:
        """Check if cached data is valid"""
        if key not in self._cache:
            return False
        cached_time = self._cache[key].get("timestamp", 0)
        return (datetime.now().timestamp() - cached_time) < ttl
    
    @timed("chain_get_orderbook")
    async def get_orderbook_data(
        self,
        symbol: str = "BTC/USDT",
        depth: int = 20
    ) -> OrderbookData:
        """
        Get orderbook data with analysis
        
        Args:
            symbol: Trading pair
            depth: Number of levels to fetch
            
        Returns:
            OrderbookData with analysis
        """
        cache_key = self._get_cache_key("orderbook", symbol)
        
        if self._is_cache_valid(cache_key, self._cache_ttl["orderbook"]):
            logger.debug(f"cache_hit: {cache_key}")
            get_metrics().increment("orderbook_cache_hits")
            return self._cache[cache_key]["data"]
        
        get_metrics().increment("orderbook_cache_misses")
        
        exchange = await self._get_exchange()
        
        if exchange:
            try:
                orderbook = await exchange.fetch_order_book(symbol, limit=depth)
                
                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                
                if bids and asks:
                    # Calculate volumes
                    bid_volume = sum(b[1] for b in bids[:10])
                    ask_volume = sum(a[1] for a in asks[:10])
                    
                    # Calculate imbalance
                    total_volume = bid_volume + ask_volume
                    imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
                    
                    # Calculate spread
                    best_bid = bids[0][0]
                    best_ask = asks[0][0]
                    mid_price = (best_bid + best_ask) / 2
                    spread = ((best_ask - best_bid) / mid_price) * 100
                    
                    # Detect whale walls
                    base_currency = symbol.split("/")[0]
                    threshold = self.WHALE_THRESHOLDS.get(
                        base_currency, 
                        self.WHALE_THRESHOLDS["DEFAULT"]
                    )
                    whale_walls = self._detect_whale_walls(bids, asks, threshold)
                    
                    # Calculate depth at different levels
                    depth_levels = self._calculate_depth_levels(bids, asks, mid_price)
                    
                    data = OrderbookData(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        bid_volume=bid_volume,
                        ask_volume=ask_volume,
                        imbalance=round(imbalance, 4),
                        spread=round(spread, 4),
                        mid_price=mid_price,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        whale_walls=whale_walls,
                        depth_levels=depth_levels
                    )
                    
                    # Cache result
                    self._cache[cache_key] = {
                        "data": data,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    logger.info(
                        "orderbook_fetched",
                        symbol=symbol,
                        imbalance=data.imbalance
                    )
                    
                    return data
                    
            except Exception as e:
                logger.error(f"orderbook_fetch_error: {e}")
        
        # Return default data
        return self._get_default_orderbook(symbol)
    
    def _detect_whale_walls(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Detect large orders (whale walls)
        
        Args:
            bids: Bid orders
            asks: Ask orders
            threshold: Minimum size to consider whale
            
        Returns:
            List of whale walls
        """
        walls = []
        
        for price, size in bids:
            if size >= threshold:
                walls.append({
                    "side": "bid",
                    "price": price,
                    "size": size,
                    "type": "support"
                })
        
        for price, size in asks:
            if size >= threshold:
                walls.append({
                    "side": "ask",
                    "price": price,
                    "size": size,
                    "type": "resistance"
                })
        
        return walls[:10]  # Limit to top 10
    
    def _calculate_depth_levels(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float
    ) -> Dict[str, float]:
        """Calculate cumulative depth at percentage levels"""
        levels = {}
        
        for pct in [0.5, 1.0, 2.0, 5.0]:
            lower = mid_price * (1 - pct / 100)
            upper = mid_price * (1 + pct / 100)
            
            bid_depth = sum(b[1] for b in bids if b[0] >= lower)
            ask_depth = sum(a[1] for a in asks if a[0] <= upper)
            
            levels[f"bid_{pct}pct"] = round(bid_depth, 4)
            levels[f"ask_{pct}pct"] = round(ask_depth, 4)
        
        return levels
    
    def calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """
        Calculate orderbook imbalance score
        
        Args:
            orderbook: Raw orderbook data
            
        Returns:
            Imbalance score from -1 (bearish) to 1 (bullish)
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        bid_volume = sum(b[1] for b in bids[:10])
        ask_volume = sum(a[1] for a in asks[:10])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    @timed("chain_get_price_data")
    async def get_price_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get OHLCV price data with technical indicators
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles
            
        Returns:
            DataFrame with price data and indicators
        """
        cache_key = self._get_cache_key(f"price_{timeframe}", symbol)
        
        if self._is_cache_valid(cache_key, self._cache_ttl["price"]):
            return self._cache[cache_key]["data"]
        
        exchange = await self._get_exchange()
        
        if exchange:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if ohlcv:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    
                    # Add technical indicators
                    df = self._add_technical_indicators(df)
                    
                    # Cache result
                    self._cache[cache_key] = {
                        "data": df,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    logger.info("price_data_fetched", symbol=symbol, rows=len(df))
                    return df
                    
            except Exception as e:
                logger.error(f"price_fetch_error: {e}")
        
        # Return empty DataFrame
        return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        if df.empty:
            return df
        
        # RSI
        df["rsi"] = self._calculate_rsi(df["close"])
        
        # MACD
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        
        # Volatility (ATR-like)
        df["volatility"] = (df["high"] - df["low"]).rolling(window=14).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @timed("chain_get_fear_greed")
    async def get_fear_greed_index(self) -> int:
        """
        Fetch Fear & Greed Index from Alternative.me
        
        Returns:
            Index value (0-100)
        """
        cache_key = "fear_greed"
        
        if self._is_cache_valid(cache_key, self._cache_ttl["fear_greed"]):
            return self._cache[cache_key]["data"]
        
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        value = int(data["data"][0]["value"])
                        
                        self._cache[cache_key] = {
                            "data": value,
                            "timestamp": datetime.now().timestamp()
                        }
                        
                        logger.info("fear_greed_fetched", value=value)
                        return value
                        
        except Exception as e:
            logger.error(f"fear_greed_error: {e}")
        
        return 50  # Neutral default
    
    async def get_volume_profile(
        self,
        symbol: str = "BTC/USDT",
        periods: int = 24
    ) -> Dict[float, float]:
        """
        Get volume profile (price -> volume mapping)
        
        Args:
            symbol: Trading pair
            periods: Number of periods to analyze
            
        Returns:
            Dictionary of price_level -> total_volume
        """
        df = await self.get_price_data(symbol, "1h", periods)
        
        if df.empty:
            return {}
        
        profile = {}
        
        for _, row in df.iterrows():
            # Use close price as representative
            price_level = round(row["close"], -2)  # Round to nearest 100
            profile[price_level] = profile.get(price_level, 0) + row["volume"]
        
        return dict(sorted(profile.items()))
    
    async def get_market_metrics(
        self,
        symbol: str = "BTC/USDT"
    ) -> MarketMetrics:
        """
        Get comprehensive market metrics
        
        Args:
            symbol: Trading pair
            
        Returns:
            MarketMetrics object
        """
        exchange = await self._get_exchange()
        
        if exchange:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                
                # Get additional data
                orderbook = await self.get_orderbook_data(symbol)
                fear_greed = await self.get_fear_greed_index()
                price_df = await self.get_price_data(symbol, "1h", 24)
                
                # Calculate volatility
                volatility = 0.0
                if not price_df.empty:
                    returns = price_df["close"].pct_change().dropna()
                    volatility = float(returns.std() * 100) if len(returns) > 0 else 0.0
                
                return MarketMetrics(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    price=ticker.get("last", 0),
                    price_change_24h=ticker.get("percentage", 0) or 0,
                    volume_24h=ticker.get("quoteVolume", 0) or 0,
                    fear_greed_index=fear_greed,
                    orderbook_imbalance=orderbook.imbalance,
                    volatility=round(volatility, 2)
                )
                
            except Exception as e:
                logger.error(f"market_metrics_error: {e}")
        
        # Return default metrics
        return MarketMetrics(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=0,
            price_change_24h=0,
            volume_24h=0
        )
    
    def _get_default_orderbook(self, symbol: str) -> OrderbookData:
        """Return default orderbook data when API unavailable"""
        return OrderbookData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bid_volume=0,
            ask_volume=0,
            imbalance=0,
            spread=0,
            mid_price=0,
            best_bid=0,
            best_ask=0
        )
