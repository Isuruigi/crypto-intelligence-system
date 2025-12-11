"""
Social Sentiment Agent for Crypto Intelligence System
Analyzes sentiment from Reddit and other social platforms
"""
from typing import Dict, Any, List
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.data.collectors.reddit_collector import RedditSentimentCollector
from src.models.sentiment_model import get_sentiment_analyzer
from src.utils.logger import get_logger
from src.utils.metrics import timed

logger = get_logger(__name__)


class SocialSentimentAgent(BaseAgent):
    """
    Analyzes social media sentiment for crypto assets
    
    Data Sources:
    - Reddit (via PRAW)
    - Synthetic data fallback
    
    Output:
    - Aggregated sentiment score
    - Trending topics
    - Post volume metrics
    """
    
    # Subreddit mapping for different coins
    COIN_SUBREDDITS = {
        "BTC": ["Bitcoin", "cryptocurrency", "CryptoMarkets"],
        "ETH": ["ethereum", "ethfinance", "cryptocurrency"],
        "SOL": ["solana", "cryptocurrency"],
        "XRP": ["XRP", "Ripple", "cryptocurrency"],
        "ADA": ["cardano", "cryptocurrency"],
        "DOGE": ["dogecoin", "cryptocurrency"],
        "DEFAULT": ["cryptocurrency", "CryptoMarkets", "altcoin"]
    }
    
    def __init__(self):
        """Initialize social sentiment agent"""
        super().__init__("social_sentiment")
        self.reddit_collector = RedditSentimentCollector()
        self.sentiment_analyzer = get_sentiment_analyzer()
    
    @timed("social_agent_analyze")
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Analyze social sentiment for the given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Social sentiment analysis results
        """
        # Extract base currency
        currency = symbol.split("/")[0].upper()
        
        logger.info("social_analysis_started", symbol=symbol, currency=currency)
        
        try:
            # Get relevant subreddits
            subreddits = self.COIN_SUBREDDITS.get(
                currency, 
                self.COIN_SUBREDDITS["DEFAULT"]
            )
            
            # Collect posts from subreddits
            posts_df = self.reddit_collector.collect_multi_subreddit(
                subreddits=subreddits,
                limit_per_sub=30,
                time_filter="day"
            )
            
            if posts_df.empty:
                logger.warning("no_social_data", currency=currency)
                return self._get_neutral_social_result(currency)
            
            # Prepare texts for sentiment analysis
            texts = []
            for _, row in posts_df.iterrows():
                title = row.get("title", "")
                text = row.get("text", "")
                combined = f"{title} {text}".strip()
                if combined:
                    texts.append(combined)
            
            if not texts:
                return self._get_neutral_social_result(currency)
            
            # Analyze sentiment
            aggregated = self.sentiment_analyzer.get_aggregated_sentiment(texts)
            
            # Get trending topics
            trending = self.reddit_collector.get_trending_topics(subreddits)
            
            # Calculate engagement score
            avg_score = posts_df["score"].mean() if "score" in posts_df else 0
            avg_comments = posts_df["num_comments"].mean() if "num_comments" in posts_df else 0
            engagement = self._calculate_engagement_score(avg_score, avg_comments)
            
            result = {
                "success": True,
                "currency": currency,
                "sentiment_score": aggregated.weighted_compound,
                "overall_sentiment": aggregated.overall_sentiment,
                "confidence": aggregated.avg_confidence,
                "positive_pct": aggregated.positive_pct,
                "negative_pct": aggregated.negative_pct,
                "neutral_pct": aggregated.neutral_pct,
                "post_count": len(texts),
                "trending_topics": trending,
                "engagement_score": engagement,
                "avg_post_score": round(avg_score, 2),
                "avg_comments": round(avg_comments, 2),
                "subreddits_analyzed": subreddits,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "social_analysis_complete",
                currency=currency,
                sentiment=aggregated.weighted_compound,
                posts=len(texts)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"social_analysis_error: {e}")
            return self._get_neutral_social_result(currency)
    
    def _calculate_engagement_score(
        self, 
        avg_score: float, 
        avg_comments: float
    ) -> float:
        """
        Calculate engagement score (0-100)
        
        Based on average post score and comment count
        """
        # Normalize score (typical range 0-1000)
        score_component = min(avg_score / 500, 1.0) * 50
        
        # Normalize comments (typical range 0-100)
        comment_component = min(avg_comments / 50, 1.0) * 50
        
        return round(score_component + comment_component, 2)
    
    def _get_neutral_social_result(self, currency: str) -> Dict[str, Any]:
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
            "post_count": 0,
            "trending_topics": {},
            "engagement_score": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
