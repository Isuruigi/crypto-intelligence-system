"""
Tests for AI agents
"""
import pytest
from app.agents.whale_tracker import WhaleTrackerAgent
from app.agents.orderbook_analyzer import OrderbookAnalyzer
from app.agents.sentiment_analyzer import SentimentAnalyzer
from app.agents.coordinator import LLMCoordinator


@pytest.mark.asyncio
async def test_whale_tracker():
    """Test whale tracker agent"""
    agent = WhaleTrackerAgent()
    data = await agent.analyze(symbol="BTC/USDT")
    
    # Validate output structure
    assert 'whale_score' in data
    assert 'whale_count' in data
    assert 'sentiment' in data
    assert 'signal' in data
    
    # Validate ranges
    assert -100 <= data['whale_score'] <= 100
    assert data['whale_count'] >= 0
    assert data['signal'] in ['BUY', 'SELL', 'HOLD']


@pytest.mark.asyncio
async def test_orderbook_analyzer():
    """Test orderbook analyzer agent"""
    agent = OrderbookAnalyzer()
    data = await agent.analyze(symbol="BTC/USDT")
    
    # Validate output structure
    assert 'imbalance' in data
    assert 'market_depth_score' in data
    assert 'spoofing_detected' in data
    
    # Validate ranges
    assert -1 <= data['imbalance'] <= 1
    assert 0 <= data['market_depth_score'] <= 100
    assert isinstance(data['spoofing_detected'], bool)


@pytest.mark.asyncio
async def test_sentiment_analyzer():
    """Test sentiment analyzer agent"""
    agent = SentimentAnalyzer()
    data = await agent.analyze(symbol="BTC/USDT")
    
    # Validate output structure
    assert 'sentiment_score' in data
    assert 'fear_greed_index' in data
    assert 'dominant_emotion' in data
    
    # Validate ranges
    assert -1 <= data['sentiment_score'] <= 1
    assert 0 <= data['fear_greed_index'] <= 100
    assert data['dominant_emotion'] in ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']


@pytest.mark.asyncio
async def test_llm_coordinator():
    """Test LLM coordinator agent"""
    coordinator = LLMCoordinator()
    
    # Mock data
    sentiment = {
        'sentiment_score': 0.5,
        'fear_greed_index': 70,
        'post_count': 50,
        'volume': 'high',
        'dominant_emotion': 'greed',
        'sentiment_velocity': 0.0,
        'top_keywords': ['bullish', 'moon', 'buy']
    }
    
    whale = {
        'whale_score': -80,
        'whale_count': 15,
        'total_volume': 1500,
        'exchange_pressure': 1500,
        'signal': 'SELL'
    }
    
    orderbook = {
        'imbalance': -0.3,
        'market_depth_score': 75,
        'spoofing_detected': False,
        'best_bid': 96000,
        'best_ask': 96050
    }
    
    # Generate signal
    signal = await coordinator.generate_trading_thesis(
        retail_sentiment=sentiment,
        whale_sentiment=whale,
        orderbook=orderbook,
        current_price=96025.0,
        symbol="BTC"
    )
    
    # Validate output
    assert 'signal' in signal
    assert 'confidence' in signal
    assert 'thesis' in signal
    assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal['confidence'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
