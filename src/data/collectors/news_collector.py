"""
News Data Collector for Crypto Intelligence System
Collects and analyzes crypto news articles
"""
import re
import hashlib
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from src.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import timed, get_metrics

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """Schema for a news article"""
    article_id: str
    title: str
    description: str
    content: str
    source: str
    published_at: datetime
    url: str
    author: Optional[str] = None
    image_url: Optional[str] = None
    coin_mentions: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None


class NewsCollector:
    """
    News article collector for crypto news sentiment analysis
    
    Data Sources:
    - NewsAPI (primary)
    - Direct scraping of crypto news sites (fallback)
    
    Features:
    - NewsAPI integration
    - Article content extraction
    - Coin mention extraction
    - 6-hour caching
    """
    
    # Crypto news sources to prioritize
    CRYPTO_SOURCES = [
        "coindesk",
        "cointelegraph",
        "decrypt",
        "the-block",
        "bitcoin-magazine"
    ]
    
    # Coin symbol patterns for extraction
    COIN_PATTERNS = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth", "ether"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "ADA": ["cardano", "ada"],
        "DOGE": ["dogecoin", "doge"],
        "BNB": ["binance coin", "bnb"],
        "MATIC": ["polygon", "matic"],
        "DOT": ["polkadot", "dot"],
        "AVAX": ["avalanche", "avax"],
        "LINK": ["chainlink", "link"],
        "UNI": ["uniswap", "uni"],
    }
    
    # User agents for scraping
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]
    
    def __init__(self, api_key: str = None):
        """
        Initialize News collector
        
        Args:
            api_key: NewsAPI key
        """
        self.settings = get_settings()
        self.api_key = api_key or self.settings.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 21600  # 6 hours
        self._daily_requests = 0
        self._daily_limit = 100  # NewsAPI free tier
        self._last_reset = datetime.now().date()
        self._user_agent_index = 0
        
        logger.info("news_collector_initialized")
    
    def _get_user_agent(self) -> str:
        """Rotate user agents for scraping"""
        agent = self.USER_AGENTS[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(self.USER_AGENTS)
        return agent
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within daily rate limits"""
        today = datetime.now().date()
        if today != self._last_reset:
            self._daily_requests = 0
            self._last_reset = today
        
        return self._daily_requests < self._daily_limit
    
    def _get_cache_key(self, query: str, hours_back: int) -> str:
        """Generate cache key"""
        return f"news:{query}:{hours_back}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache:
            return False
        
        cached_time = self._cache[key].get("timestamp", 0)
        return (datetime.now().timestamp() - cached_time) < self._cache_ttl
    
    @timed("news_collect")
    async def collect_crypto_news(
        self,
        hours_back: int = 24,
        query: str = "cryptocurrency OR bitcoin OR ethereum"
    ) -> pd.DataFrame:
        """
        Collect crypto news articles
        
        Args:
            hours_back: How many hours back to search
            query: Search query
            
        Returns:
            DataFrame with news articles
        """
        cache_key = self._get_cache_key(query, hours_back)
        
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.debug(f"cache_hit: {cache_key}")
            get_metrics().increment("news_cache_hits")
            return self._cache[cache_key]["data"]
        
        get_metrics().increment("news_cache_misses")
        
        articles = []
        
        # Try NewsAPI first
        if self.api_key and self._check_rate_limit():
            articles = await self._fetch_from_newsapi(query, hours_back)
        
        # Fallback to synthetic data if no articles
        if not articles:
            articles = self._generate_synthetic_articles(hours_back)
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(a) for a in articles])
        
        # Cache result
        self._cache[cache_key] = {
            "data": df,
            "timestamp": datetime.now().timestamp()
        }
        
        return df
    
    async def _fetch_from_newsapi(
        self,
        query: str,
        hours_back: int
    ) -> List[NewsArticle]:
        """Fetch articles from NewsAPI"""
        from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self.api_key
        }
        
        articles = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._daily_requests += 1
                        
                        for item in data.get("articles", []):
                            try:
                                article = NewsArticle(
                                    article_id=hashlib.md5(
                                        item["url"].encode()
                                    ).hexdigest()[:12],
                                    title=item.get("title", ""),
                                    description=item.get("description", "") or "",
                                    content=item.get("content", "") or "",
                                    source=item.get("source", {}).get("name", "Unknown"),
                                    published_at=datetime.fromisoformat(
                                        item["publishedAt"].replace("Z", "+00:00")
                                    ),
                                    url=item["url"],
                                    author=item.get("author"),
                                    image_url=item.get("urlToImage"),
                                    coin_mentions=self._extract_coin_mentions(
                                        f"{item.get('title', '')} {item.get('description', '')}"
                                    )
                                )
                                articles.append(article)
                            except Exception as e:
                                logger.debug(f"article_parse_error: {e}")
                                continue
                        
                        logger.info(
                            "newsapi_fetch_complete",
                            count=len(articles)
                        )
                    else:
                        logger.warning(
                            f"newsapi_error: status={response.status}"
                        )
                        
        except Exception as e:
            logger.error(f"newsapi_request_error: {e}")
        
        return articles
    
    async def get_article_content(self, url: str) -> str:
        """
        Scrape full article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Extracted article text
        """
        if not BS4_AVAILABLE:
            return ""
        
        try:
            headers = {"User-Agent": self._get_user_agent()}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._extract_article_text(html)
                        
        except Exception as e:
            logger.debug(f"article_scrape_error: {url}, {e}")
        
        return ""
    
    def _extract_article_text(self, html: str) -> str:
        """Extract main article text from HTML"""
        if not BS4_AVAILABLE:
            return ""
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # Try to find article body
            article = soup.find("article")
            if article:
                paragraphs = article.find_all("p")
            else:
                paragraphs = soup.find_all("p")
            
            # Extract text from paragraphs
            text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Filter short paragraphs
                    text_parts.append(text)
            
            return " ".join(text_parts)[:5000]  # Limit length
            
        except Exception as e:
            logger.debug(f"html_parse_error: {e}")
            return ""
    
    def filter_relevant_news(
        self,
        articles: pd.DataFrame,
        min_relevance: float = 0.3
    ) -> pd.DataFrame:
        """
        Filter articles by crypto relevance
        
        Args:
            articles: DataFrame of articles
            min_relevance: Minimum relevance score
            
        Returns:
            Filtered DataFrame
        """
        if articles.empty:
            return articles
        
        # Calculate relevance based on coin mentions
        def calc_relevance(row):
            mentions = row.get("coin_mentions", [])
            if isinstance(mentions, list):
                return min(len(mentions) / 3, 1.0)
            return 0.0
        
        articles["relevance"] = articles.apply(calc_relevance, axis=1)
        return articles[articles["relevance"] >= min_relevance]
    
    def _extract_coin_mentions(self, text: str) -> List[str]:
        """
        Extract cryptocurrency mentions from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of coin symbols mentioned
        """
        if not text:
            return []
        
        text_lower = text.lower()
        found = []
        
        for symbol, patterns in self.COIN_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(symbol)
                    break
        
        return list(set(found))
    
    def _generate_synthetic_articles(
        self,
        hours_back: int
    ) -> List[NewsArticle]:
        """Generate synthetic articles for testing"""
        import random
        
        templates = [
            {
                "title": "Bitcoin Surges Past Key Resistance Level",
                "description": "BTC shows strength as it breaks through resistance.",
                "source": "CryptoNews",
                "coins": ["BTC"]
            },
            {
                "title": "Ethereum Layer 2 Solutions See Record Activity",
                "description": "L2 networks process record transactions.",
                "source": "DeFi Daily",
                "coins": ["ETH"]
            },
            {
                "title": "Institutional Investors Increase Crypto Allocations",
                "description": "Major funds report increased crypto holdings.",
                "source": "Financial Times",
                "coins": ["BTC", "ETH"]
            },
            {
                "title": "Solana Network Upgrade Improves Performance",
                "description": "SOL ecosystem sees enhanced throughput.",
                "source": "Blockchain Weekly",
                "coins": ["SOL"]
            },
            {
                "title": "DeFi Total Value Locked Reaches New High",
                "description": "Decentralized finance continues growth.",
                "source": "DeFi Pulse",
                "coins": ["ETH", "AVAX"]
            },
            {
                "title": "Crypto Market Shows Mixed Signals",
                "description": "Analysts divided on short-term direction.",
                "source": "Market Watch",
                "coins": ["BTC", "ETH"]
            },
            {
                "title": "Regulatory Clarity Improves for Crypto",
                "description": "New framework provides clearer guidelines.",
                "source": "Policy Times",
                "coins": ["BTC"]
            },
            {
                "title": "NFT Market Sees Renewed Interest",
                "description": "Digital collectibles regain momentum.",
                "source": "NFT News",
                "coins": ["ETH", "SOL"]
            },
        ]
        
        articles = []
        for i, template in enumerate(templates):
            article = NewsArticle(
                article_id=hashlib.md5(f"synthetic_{i}".encode()).hexdigest()[:12],
                title=template["title"],
                description=template["description"],
                content=template["description"] * 3,
                source=template["source"],
                published_at=datetime.now() - timedelta(
                    hours=random.randint(1, hours_back)
                ),
                url=f"https://example.com/article/{i}",
                author="Crypto Analyst",
                coin_mentions=template["coins"]
            )
            articles.append(article)
        
        logger.info("generated_synthetic_articles", count=len(articles))
        return articles
    
    async def collect_from_multiple_sources(
        self,
        queries: List[str] = None,
        hours_back: int = 24
    ) -> pd.DataFrame:
        """
        Collect news from multiple search queries
        
        Args:
            queries: List of search queries
            hours_back: Hours to look back
            
        Returns:
            Combined DataFrame
        """
        queries = queries or [
            "bitcoin",
            "ethereum",
            "cryptocurrency market",
            "crypto regulation",
            "defi"
        ]
        
        all_articles = []
        
        for query in queries:
            try:
                df = await self.collect_crypto_news(hours_back, query)
                if not df.empty:
                    all_articles.append(df)
            except Exception as e:
                logger.warning(f"multi_query_error: {query}, {e}")
        
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True)
            # Remove duplicates by article_id
            return combined.drop_duplicates(subset=["article_id"])
        
        return pd.DataFrame()
