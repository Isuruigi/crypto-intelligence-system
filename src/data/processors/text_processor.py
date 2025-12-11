"""
Text Processing Module for Crypto Intelligence System
Handles text cleaning, normalization, and preprocessing
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import unicodedata

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedText:
    """Container for processed text data"""
    original: str
    cleaned: str
    tokens: List[str]
    crypto_mentions: List[str]
    sentiment_keywords: Dict[str, int]


class TextProcessor:
    """
    Production-ready text processor for crypto content
    
    Features:
    - Text cleaning and normalization
    - Crypto-specific preprocessing
    - Keyword extraction
    - Emoji handling
    """
    
    # Crypto-related keywords
    CRYPTO_KEYWORDS = {
        'bullish': ['bullish', 'bull', 'moon', 'pump', 'rocket', 'ath', 'breakout', 'rally'],
        'bearish': ['bearish', 'bear', 'dump', 'crash', 'dip', 'correction', 'fall', 'plunge'],
        'neutral': ['hodl', 'hold', 'accumulate', 'consolidation', 'sideways']
    }
    
    # Common crypto symbols
    CRYPTO_SYMBOLS = {
        'BTC': ['bitcoin', 'btc', '$btc'],
        'ETH': ['ethereum', 'eth', '$eth', 'ether'],
        'SOL': ['solana', 'sol', '$sol'],
        'ADA': ['cardano', 'ada', '$ada'],
        'XRP': ['ripple', 'xrp', '$xrp'],
        'DOGE': ['dogecoin', 'doge', '$doge'],
        'BNB': ['binance', 'bnb', '$bnb']
    }
    
    def __init__(self, remove_urls: bool = True, lowercase: bool = True):
        """
        Initialize text processor
        
        Args:
            remove_urls: Whether to remove URLs from text
            lowercase: Whether to convert text to lowercase
        """
        self.remove_urls = remove_urls
        self.lowercase = lowercase
        logger.info("TextProcessor initialized")
    
    def process(self, text: str) -> ProcessedText:
        """
        Process a single text string
        
        Args:
            text: Raw text to process
            
        Returns:
            ProcessedText with cleaned text and metadata
        """
        if not text or not isinstance(text, str):
            return ProcessedText(
                original="",
                cleaned="",
                tokens=[],
                crypto_mentions=[],
                sentiment_keywords={}
            )
        
        original = text
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        crypto_mentions = self._extract_crypto_mentions(cleaned)
        sentiment_keywords = self._extract_sentiment_keywords(cleaned)
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            tokens=tokens,
            crypto_mentions=crypto_mentions,
            sentiment_keywords=sentiment_keywords
        )
    
    def process_batch(self, texts: List[str]) -> List[ProcessedText]:
        """
        Process multiple texts
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of ProcessedText objects
        """
        return [self.process(text) for text in texts]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'/r/\w+', '', text)
        text = re.sub(r'/u/\w+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?$%\'-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.split()
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract mentioned cryptocurrencies"""
        mentions = []
        text_lower = text.lower()
        
        for symbol, keywords in self.CRYPTO_SYMBOLS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if symbol not in mentions:
                        mentions.append(symbol)
                    break
        
        return mentions
    
    def _extract_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """Extract sentiment-related keywords"""
        counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        text_lower = text.lower()
        
        for sentiment, keywords in self.CRYPTO_KEYWORDS.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + keyword + r'\b', text_lower))
                counts[sentiment] += count
        
        return counts
    
    def get_overall_sentiment_hint(self, text: str) -> str:
        """
        Get a quick sentiment hint based on keywords
        
        Args:
            text: Text to analyze
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        counts = self._extract_sentiment_keywords(text.lower())
        
        if counts['bullish'] > counts['bearish']:
            return 'bullish'
        elif counts['bearish'] > counts['bullish']:
            return 'bearish'
        return 'neutral'


# Global instance
_processor: Optional[TextProcessor] = None

def get_text_processor() -> TextProcessor:
    """Get global text processor instance"""
    global _processor
    if _processor is None:
        _processor = TextProcessor()
    return _processor


if __name__ == "__main__":
    # Test the processor
    processor = TextProcessor()
    
    test_texts = [
        "Bitcoin is going to the moon! 🚀 BTC will reach $100k soon!",
        "Market crash incoming. ETH might dump hard. Bearish on crypto.",
        "Just HODLing my SOL and waiting for the next bull run.",
    ]
    
    for text in test_texts:
        result = processor.process(text)
        print(f"\nOriginal: {result.original}")
        print(f"Cleaned: {result.cleaned}")
        print(f"Crypto mentions: {result.crypto_mentions}")
        print(f"Sentiment keywords: {result.sentiment_keywords}")
        print(f"Hint: {processor.get_overall_sentiment_hint(text)}")
