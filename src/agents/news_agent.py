"""
News Sentiment Agent for Crypto Intelligence System
Analyzes sentiment from crypto news articles
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.data.collectors.news_collector import NewsCollector
from src.models.sentiment_model import get_sentiment_analyzer
from src.utils.logger import get_logger
from src.utils.metrics import timed

logger = get_logger(__name__)


class NewsSentimentAgent(BaseAgent):
    """
    Analyzes sentiment from crypto news articles
    
    Data Sources:
    - NewsAPI
    - Synthetic news fallback
    
    Output:
    - News sentiment score
    - Key headlines
    - Source breakdown
    """
    
    def __init__(self):
        """Initialize news sentiment agent"""
        super().__init__("news_sentiment")
        self.news_collector = NewsCollector()
        self.sentiment_analyzer = get_sentiment_analyzer()
    
    @timed("news_agent_analyze")
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Analyze news sentiment for the given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            News sentiment analysis results
        """
        # Extract currency name for query
        currency = symbol.split("/")[0].upper()
        currency_names = self._get_currency_names(currency)
        
        logger.info("news_analysis_started", symbol=symbol, currency=currency)
        
        try:
            # Build search query
            query = " OR ".join(currency_names)
            
            # Collect news articles
            articles_df = await self.news_collector.collect_crypto_news(
                hours_back=24,
                query=query
            )
            
            if articles_df.empty:
                logger.warning("no_news_data", currency=currency)
                return self._get_neutral_news_result(currency)
            
            # Prepare texts for sentiment analysis
            texts = []
            headlines = []
            sources = {}
            
            for _, row in articles_df.iterrows():
                title = row.get("title", "")
                description = row.get("description", "")
                source = row.get("source", "Unknown")
                
                combined = f"{title} {description}".strip()
                if combined:
                    texts.append(combined)
                    headlines.append({
                        "title": title[:100],
                        "source": source,
                        "sentiment": None  # Will be filled
                    })
                    sources[source] = sources.get(source, 0) + 1
            
            if not texts:
                return self._get_neutral_news_result(currency)
            
            # Analyze sentiment
            aggregated = self.sentiment_analyzer.get_aggregated_sentiment(texts)
            
            # Analyze individual headlines
            individual_results = self.sentiment_analyzer.analyze_batch(texts[:10])
            for i, result in enumerate(individual_results):
                if i < len(headlines):
                    headlines[i]["sentiment"] = result.sentiment
                    headlines[i]["score"] = result.compound_score
            
            # Sort headlines by sentiment score (most positive/negative first)
            headlines = sorted(
                headlines[:10], 
                key=lambda x: abs(x.get("score", 0)),
                reverse=True
            )
            
            result = {
                "success": True,
                "currency": currency,
                "sentiment_score": aggregated.weighted_compound,
                "overall_sentiment": aggregated.overall_sentiment,
                "confidence": aggregated.avg_confidence,
                "positive_pct": aggregated.positive_pct,
                "negative_pct": aggregated.negative_pct,
                "neutral_pct": aggregated.neutral_pct,
                "article_count": len(texts),
                "key_headlines": headlines[:5],
                "source_breakdown": sources,
                "coin_mentions": self._count_coin_mentions(articles_df),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "news_analysis_complete",
                currency=currency,
                sentiment=aggregated.weighted_compound,
                articles=len(texts)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"news_analysis_error: {e}")
            return self._get_neutral_news_result(currency)
    
    def _get_currency_names(self, currency: str) -> List[str]:
        """Get full names for currency symbol"""
        name_map = {
            "BTC": ["bitcoin", "BTC"],
            "ETH": ["ethereum", "ETH", "ether"],
            "SOL": ["solana", "SOL"],
            "XRP": ["ripple", "XRP"],
            "ADA": ["cardano", "ADA"],
            "DOGE": ["dogecoin", "DOGE"],
            "BNB": ["binance coin", "BNB"],
            "MATIC": ["polygon", "MATIC"],
        }
        return name_map.get(currency, [currency.lower(), currency])
    
    def _count_coin_mentions(self, df) -> Dict[str, int]:
        """Count coin mentions across articles"""
        mentions = {}
        
        if df.empty or "coin_mentions" not in df.columns:
            return mentions
        
        for _, row in df.iterrows():
            coins = row.get("coin_mentions", [])
            if isinstance(coins, list):
                for coin in coins:
                    mentions[coin] = mentions.get(coin, 0) + 1
        
        return dict(sorted(mentions.items(), key=lambda x: -x[1])[:10])
    
    def _get_neutral_news_result(self, currency: str) -> Dict[str, Any]:
        """Return neutral result when no data available"""
        return {
            "success": True,
            "is_fallback": True,
            "currency": currency,
            "sentiment_score": 0.0,
            "overall_sentiment": "neutral",
            "confidence": 0.5,
            "positive_pct": 33.0,
            "negative_pct": 33.0,
            "neutral_pct": 34.0,
            "article_count": 0,
            "key_headlines": [],
            "source_breakdown": {},
            "coin_mentions": {},
            "timestamp": datetime.utcnow().isoformat()
        }
