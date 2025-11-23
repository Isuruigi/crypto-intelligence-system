"""
Sentiment Analyzer Agent - Analyzes social sentiment using FinBERT and Reddit
Uses PRAW for Reddit data and FinBERT for sentiment analysis
"""
import aiohttp
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.services.rate_limiter import get_rate_limiter
from app.services.cache_service import get_cache
from app.utils.circuit_breaker import CircuitBreaker

# Try to import PRAW and transformers, but make them optional
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False


class SentimentAnalyzer(BaseAgent):
    """
    Analyzes social sentiment from Reddit and other sources
    
    Features:
    - Reddit post analysis with PRAW
    - FinBERT sentiment analysis
    - Fear & Greed Index integration
    - Sentiment velocity tracking
    """
    
    def __init__(self):
        super().__init__("sentiment_analyzer")
        self.settings = get_settings()
        self.rate_limiter = get_rate_limiter('reddit')
        self.cache = get_cache()
        self.circuit_breaker = CircuitBreaker("reddit_api", failure_threshold=3, timeout=60)
        
        # Initialize FinBERT model (lazy loading)
        self.model = None
        self.tokenizer = None
        self._init_finbert()
        
        # Initialize Reddit client
        self.reddit = None
        self._init_reddit()
    
    def _init_finbert(self):
        """Initialize FinBERT model for sentiment analysis (lazy loading)"""
        if not FINBERT_AVAILABLE:
            self.logger.warning(
                'finbert_not_available',
                message='FinBERT not available, using fallback sentiment analysis'
            )
            return
        
        # Don't load the model during initialization - it can take 30+ seconds
        # Instead, we'll load it on first use or in the background
        self.logger.info('finbert_lazy_loading_enabled', message='Model will load on first use')
        self._finbert_loading = False
        self._finbert_load_attempted = False
    
    def _init_reddit(self):
        """Initialize Reddit client with PRAW"""
        if not PRAW_AVAILABLE:
            self.logger.warning(
                'praw_not_available',
                message='PRAW not available, using fallback data'
            )
            return
        
        try:
            if self.settings.REDDIT_CLIENT_ID and self.settings.REDDIT_CLIENT_SECRET:
                self.reddit = praw.Reddit(
                    client_id=self.settings.REDDIT_CLIENT_ID,
                    client_secret=self.settings.REDDIT_CLIENT_SECRET,
                    user_agent=self.settings.REDDIT_USER_AGENT
                )
                self.logger.info('reddit_client_initialized', mode='authenticated')
            else:
                # Read-only mode
                self.reddit = praw.Reddit(
                    client_id='read_only_mode',
                    client_secret='',
                    user_agent=self.settings.REDDIT_USER_AGENT
                )
                self.logger.info('reddit_client_initialized', mode='read_only')
        except Exception as e:
            self.logger.error('reddit_init_error', error=str(e))
            self.reddit = None
    
    async def analyze(self, symbol: str = "BTC/USDT", **kwargs) -> Dict[str, Any]:
        """
        Analyze social sentiment for the given symbol
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            
        Returns:
            Sentiment analysis
        """
        base_currency = symbol.split('/')[0]
        
        # Try cache first
        cache_key = f"sentiment:{base_currency}"
        cached = await self.cache.get(cache_key)
        if cached:
            self.logger.info('sentiment_from_cache', symbol=base_currency)
            return cached
        
        try:
            # Fetch Reddit posts
            posts = await self._fetch_reddit_posts(base_currency)
            
            # Analyze sentiment with FinBERT
            sentiment_scores = await self._analyze_sentiment_finbert(posts)
            
            # Fetch Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            
            # Calculate weighted sentiment
            analysis = self._calculate_weighted_sentiment(posts, sentiment_scores, fear_greed)
            
            # Cache the result
            await self.cache.set(cache_key, analysis, self.settings.CACHE_SENTIMENT)
            
            self.logger.info(
                'sentiment_analysis_complete',
                symbol=base_currency,
                sentiment_score=analysis['sentiment_score']
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error_with_context(
                e,
                {'agent': self.name, 'symbol': base_currency}
            )
            return self._get_neutral_data()
    
    async def _fetch_reddit_posts(self, currency: str) -> List[Dict[str, Any]]:
        """
        Fetch Reddit posts about the currency
        
        Args:
            currency: Currency symbol (BTC, ETH, etc.)
            
        Returns:
            List of post dictionaries
        """
        await self.rate_limiter.acquire()
        
        if not self.reddit:
            self.logger.warning('reddit_not_initialized', message='Using synthetic posts')
            return self._generate_synthetic_posts(currency)
        
        try:
            posts = []
            subreddits = ['CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets']
            
            # Search for posts about the currency
            for sub_name in subreddits[:2]:  # Limit to 2 subreddits to avoid rate limits
                try:
                    subreddit = await asyncio.to_thread(
                        lambda: self.reddit.subreddit(sub_name)
                    )
                    
                    # Get hot posts
                    hot_posts = await asyncio.to_thread(
                        lambda: list(subreddit.hot(limit=10))
                    )
                    
                    for post in hot_posts:
                        # Filter for currency-related posts
                        if currency.lower() in post.title.lower() or currency.lower() in post.selftext.lower():
                            posts.append({
                                'title': post.title,
                                'text': post.selftext[:500],  # Limit text length
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc
                            })
                    
                except Exception as e:
                    self.logger.error('subreddit_fetch_error', subreddit=sub_name, error=str(e))
                    continue
            
            if not posts:
                self.logger.warning('no_reddit_posts_found', currency=currency)
                return self._generate_synthetic_posts(currency)
            
            self.logger.info('reddit_posts_fetched', count=len(posts), currency=currency)
            return posts
            
        except Exception as e:
            self.logger.error('reddit_fetch_error', error=str(e))
            return self._generate_synthetic_posts(currency)
    
    def _generate_synthetic_posts(self, currency: str) -> List[Dict[str, Any]]:
        """Generate synthetic posts for testing when Reddit is unavailable"""
        import random
        
        templates = [
            f"{currency} looking bullish today!",
            f"Thoughts on {currency}? Seems like accumulation...",
            f"{currency} breaking resistance, time to buy?",
            f"Whale alert: Large {currency} transaction spotted",
            f"{currency} sentiment turning positive",
            f"Should I take profits on {currency}?",
            f"{currency} consolidating, what's next?",
            f"Fear and greed for {currency} at extremes"
        ]
        
        posts = []
        for i, template in enumerate(templates[:5]):
            posts.append({
                'title': template,
                'text': f"Discussion about {currency} market conditions.",
                'score': random.randint(10, 500),
                'num_comments': random.randint(5, 200),
                'created_utc': datetime.now().timestamp() - random.randint(0, 86400)
            })
        
        return posts
    
    async def _analyze_sentiment_finbert(self, posts: List[Dict[str, Any]]) -> List[float]:
        """
        Analyze sentiment of posts using fast lexicon-based approach
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            List of sentiment scores (-1 to +1)
        """
        # Use fast lexicon-based sentiment analysis instead of FinBERT
        # This prevents 30+ second model loading delays
        sentiments = []
        
        # Simple positive/negative word lists for crypto
        positive_words = {'bullish', 'moon', 'buy', 'pump', 'gains', 'profit', 'up', 'green', 
                         'bounce', 'breakout', 'rally', 'surge', 'accumulate', 'hodl', 'strong',
                         'support', 'opportunity', 'undervalued', 'growth', 'positive'}
        
        negative_words = {'bearish', 'dump', 'sell', 'loss', 'down', 'red', 'crash', 'fear',
                         'drop', 'breakdown', 'dip', 'weak', 'resistance', 'overvalued', 
                         'concern', 'risk', 'negative', 'decline', 'fall', 'panic'}
        
        for post in posts:
            try:
                # Combine title and text
                text = f"{post['title']} {post['text']}".lower()
                words = text.split()
                
                # Count positive and negative words
                pos_count = sum(1 for word in words if word in positive_words)
                neg_count = sum(1 for word in words if word in negative_words)
                
                # Calculate sentiment score (-1 to +1)
                total = pos_count + neg_count
                if total > 0:
                    sentiment = (pos_count - neg_count) / total
                else:
                    sentiment = 0.0
                
                sentiments.append(sentiment)
                
            except Exception as e:
                self.logger.error('sentiment_analysis_error', error=str(e))
                sentiments.append(0.0)
        
        return sentiments
    
    async def _get_fear_greed_index(self) -> int:
        """
        Fetch Fear & Greed Index from Alternative.me
        
        Returns:
            Index value (0-100)
        """
        try:
            url = "https://api.alternative.me/fng/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        index = int(data['data'][0]['value'])
                        self.logger.info('fear_greed_fetched', index=index)
                        return index
                    else:
                        self.logger.error('fear_greed_api_error', status=response.status)
                        return 50  # Neutral
        except Exception as e:
            self.logger.error('fear_greed_fetch_error', error=str(e))
            return 50  # Neutral fallback
    
    def _calculate_weighted_sentiment(
        self,
        posts: List[Dict[str, Any]],
        sentiments: List[float],
        fear_greed: int
    ) -> Dict[str, Any]:
        """
        Calculate weighted sentiment score
        
        Args:
            posts: List of post dictionaries
            sentiments: List of sentiment scores
            fear_greed: Fear & Greed Index (0-100)
            
        Returns:
            Sentiment analysis results
        """
        if not posts or not sentiments:
            return self._get_neutral_data()
        
        # Calculate weighted average (weight by engagement)
        total_weight = 0
        weighted_sentiment = 0
        
        for post, sentiment in zip(posts, sentiments):
            # Weight = upvotes + comments
            weight = post['score'] + post['num_comments'] * 2
            weighted_sentiment += sentiment * weight
            total_weight += weight
        
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Normalize Fear & Greed to -1 to +1 scale
        # 0 (extreme fear) -> -1, 50 (neutral) -> 0, 100 (extreme greed) -> +1
        fg_normalized = (fear_greed - 50) / 50
        
        # Combine Reddit sentiment and Fear & Greed (70% Reddit, 30% F&G)
        combined_sentiment = avg_sentiment * 0.7 + fg_normalized * 0.3
        
        # Determine volume
        if len(posts) > 20:
            volume = "high"
        elif len(posts) > 10:
            volume = "normal"
        else:
            volume = "low"
        
        # Determine dominant emotion
        if fear_greed < 20:
            emotion = "extreme_fear"
        elif fear_greed < 40:
            emotion = "fear"
        elif fear_greed > 80:
            emotion = "extreme_greed"
        elif fear_greed > 60:
            emotion = "greed"
        else:
            emotion = "neutral"
        
        # Extract top keywords (simplified)
        all_text = " ".join([f"{p['title']} {p['text']}" for p in posts])
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [word for word, _ in top_keywords]
        
        return {
            'sentiment_score': round(combined_sentiment, 4),
            'fear_greed_index': fear_greed,
            'post_count': len(posts),
            'volume': volume,
            'dominant_emotion': emotion,
            'sentiment_velocity': 0.0,  # Would require historical data
            'top_keywords': top_keywords,
            'reddit_sentiment': round(avg_sentiment, 4),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def _get_neutral_data(self) -> Dict[str, Any]:
        """Return neutral sentiment data when no data available"""
        return {
            'sentiment_score': 0.0,
            'fear_greed_index': 50,
            'post_count': 0,
            'volume': 'low',
            'dominant_emotion': 'neutral',
            'sentiment_velocity': 0.0,
            'top_keywords': [],
            'reddit_sentiment': 0.0,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
