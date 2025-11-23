"""
LLM Coordinator Agent - Synthesizes signals using Groq API (llama-3.1-70b-versatile)
Coordinates outputs from all agents and generates final trading signals
"""
import json
import asyncio
from typing import Dict, Any
from datetime import datetime
from groq import AsyncGroq
from app.agents.base_agent import BaseAgent
from app.config import get_settings
from app.services.rate_limiter import get_rate_limiter
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCoordinator(BaseAgent):
    """
    LLM-based coordinator using Groq API
    
    Synthesizes signals from whale tracker, orderbook analyzer, and sentiment analyzer
    """
    
    def __init__(self):
        super().__init__("llm_coordinator")
        self.settings = get_settings()
        self.rate_limiter = get_rate_limiter('groq')
        
        # Initialize Groq client
        self.client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze method required by BaseAgent
        Delegates to generate_trading_thesis
        """
        return await self.generate_trading_thesis(**kwargs)
    
    async def generate_trading_thesis(
        self,
        retail_sentiment: Dict[str, Any],
        whale_sentiment: Dict[str, Any],
        orderbook: Dict[str, Any],
        current_price: float,
        symbol: str = "BTC"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading thesis using LLM
        
        Args:
            retail_sentiment: Sentiment analysis from social media
            whale_sentiment: Whale tracking analysis
            orderbook: Order book microstructure analysis
            current_price: Current market price
            symbol: Trading symbol
            
        Returns:
            Trading signal with thesis
        """
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(
                retail_sentiment,
                whale_sentiment,
                orderbook,
                current_price,
                symbol
            )
            
            # Call Groq API with rate limiting
            await self.rate_limiter.acquire()
            
            self.logger.info('calling_groq_api', model=self.settings.GROQ_MODEL)
            
            response = await self.client.chat.completions.create(
                model=self.settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantitative trading analyst with deep knowledge of market microstructure, behavioral finance, and multi-modal data synthesis. You provide actionable trading insights based on data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = self._parse_llm_response(
                response.choices[0].message.content,
                retail_sentiment,
                whale_sentiment,
                orderbook
            )
            
            self.logger.info(
                'llm_coordination_complete',
                signal=result['signal'],
                confidence=result['confidence']
            )
            
            return result
            
        except Exception as e:
            self.logger.error_with_context(
                e,
                {'agent': self.name, 'symbol': symbol}
            )
            # Return neutral signal on error
            return self._get_fallback_signal(retail_sentiment, whale_sentiment, orderbook)
    
    def _build_analysis_prompt(
        self,
        sentiment: Dict[str, Any],
        whale: Dict[str, Any],
        orderbook: Dict[str, Any],
        price: float,
        symbol: str
    ) -> str:
        """Build comprehensive analysis prompt for LLM"""
        
        # Calculate divergence
        divergence = abs(whale['whale_score'] - (sentiment['sentiment_score'] * 100))
        
        # Determine volatility regime (simplified)
        spread = orderbook.get('spread', 0)
        regime = "high_volatility" if spread > 0.5 else "low_volatility"
        
        prompt = f"""Analyze this multi-modal cryptocurrency market data and provide a trading recommendation.

**CURRENT MARKET DATA**
Symbol: {symbol}
Price: ${price:,.2f}
Volatility Regime: {regime}
Spread: {spread:.4f}%

**RETAIL SENTIMENT (Social Media & Fear/Greed)**
- Sentiment Score: {sentiment['sentiment_score']:.2f} (-1 = extreme fear, +1 = extreme greed)
- Fear/Greed Index: {sentiment['fear_greed_index']}/100
- Post Volume: {sentiment['post_count']} posts ({sentiment['volume']} volume)
- Dominant Emotion: {sentiment['dominant_emotion']}
- Top Keywords: {', '.join(sentiment.get('top_keywords', [])[:3])}

**WHALE ACTIVITY (Smart Money)**
- Accumulation Score: {whale['whale_score']:.1f} (-100 = strong distribution, +100 = strong accumulation)
- Whale Transactions: {whale['whale_count']} large transactions
- Total Volume: {whale['total_volume']:.2f} {symbol}
- Exchange Pressure: {whale['exchange_pressure']:.2f} {symbol} (negative = accumulation)
- Whale Signal: {whale['signal']}

**ORDER BOOK MICROSTRUCTURE**
- Bid/Ask Imbalance: {orderbook['imbalance']:.4f} (-1 = sell pressure, +1 = buy pressure)
- Market Depth Score: {orderbook['market_depth_score']:.1f}/100
- Spoofing Detected: {orderbook['spoofing_detected']}
- Best Bid: ${orderbook.get('best_bid', 0):,.2f}
- Best Ask: ${orderbook.get('best_ask', 0):,.2f}

**DIVERGENCE ANALYSIS**
- Whale vs Retail Divergence: {divergence:.1f} points
- Pattern: {"STRONG DIVERGENCE - Contrarian Signal" if divergence > 50 else "Moderate Alignment"}

**ANALYSIS FRAMEWORK**
1. Whale/retail divergence is the STRONGEST signal (>50 points = high confidence contrarian trade)
2. When whales accumulate + retail fears = BULLISH
3. When whales distribute + retail greedy = BEARISH
4. Spoofing invalidates orderbook signals
5. Fear < 20 or Greed > 80 often marks reversals
6. Confirm with orderbook imbalance

Provide your analysis in JSON format:
{{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0-100,
  "timeframe": "2-6 hours" | "6-24 hours" | "24-48 hours",
  "thesis": "2-3 sentence explanation of the trade setup",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_factors": ["risk1", "risk2"],
  "historical_pattern": "Description of similar past scenarios",
  "position_sizing": "Conservative: X% | Moderate: Y% | Aggressive: Z%"
}}

Be decisive but honest about uncertainty. Higher divergence = higher confidence."""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        sentiment: Dict,
        whale: Dict,
        orderbook: Dict
    ) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Parse JSON response
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'timeframe', 'thesis', 'key_factors', 'risk_factors']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure signal is valid
            if data['signal'] not in ['BUY', 'SELL', 'HOLD']:
                data['signal'] = 'HOLD'
            
            # Clamp confidence
            data['confidence'] = max(0, min(100, int(data['confidence'])))
            
            # Add metadata
            data['raw_analysis'] = response
            data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            data['underlying_data'] = {
                'retail_sentiment': sentiment,
                'whale_sentiment': whale,
                'orderbook': orderbook
            }
            
            return data
            
        except Exception as e:
            self.logger.error('llm_parse_error', error=str(e), response=response[:200])
            return self._get_fallback_signal(sentiment, whale, orderbook)
    
    def _get_fallback_signal(
        self,
        sentiment: Dict,
        whale: Dict,
        orderbook: Dict
    ) -> Dict[str, Any]:
        """
        Generate fallback signal using rule-based logic
        Used when LLM fails
        """
        # Simple rule-based logic
        whale_score = whale['whale_score']
        sentiment_score = sentiment['sentiment_score'] * 100
        imbalance = orderbook['imbalance']
        
        # Calculate divergence
        divergence = whale_score - sentiment_score
        
        # Decision logic
        if divergence > 50:
            # Whales accumulating, retail fearful -> BUY
            signal = "BUY"
            confidence = min(80, int(abs(divergence)))
            thesis = f"Strong whale accumulation (+{whale_score:.0f}) while retail shows fear ({sentiment_score:.0f}). Classic contrarian setup."
        elif divergence < -50:
            # Whales distributing, retail greedy -> SELL
            signal = "SELL"
            confidence = min(80, int(abs(divergence)))
            thesis = f"Whales distributing ({whale_score:.0f}) while retail shows greed ({sentiment_score:.0f}). Distribution phase."
        elif whale_score > 40 and imbalance > 0.3:
            # Whale accumulation + orderbook support -> BUY
            signal = "BUY"
            confidence = 65
            thesis = "Whale accumulation confirmed by strong bid support in orderbook."
        elif whale_score < -40 and imbalance < -0.3:
            # Whale distribution + orderbook resistance -> SELL
            signal = "SELL"
            confidence = 65
            thesis = "Whale distribution confirmed by heavy ask pressure in orderbook."
        else:
            # No clear signal
            signal = "HOLD"
            confidence = 30
            thesis = "Mixed signals across data sources. Waiting for clearer setup."
        
        return {
            'signal': signal,
            'confidence': confidence,
            'timeframe': '6-24 hours',
            'thesis': thesis,
            'key_factors': [
                f"Whale score: {whale_score:.0f}",
                f"Sentiment score: {sentiment_score:.0f}",
                f"Divergence: {divergence:.0f} points"
            ],
            'risk_factors': [
                "Fallback signal due to LLM unavailability",
                "Limited factor analysis"
            ],
            'historical_pattern': "Rule-based analysis",
            'position_sizing': "Conservative: 2% | Moderate: 4% | Aggressive: 6%",
            'raw_analysis': "Fallback rule-based signal",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'underlying_data': {
                'retail_sentiment': sentiment,
                'whale_sentiment': whale,
                'orderbook': orderbook
            }
        }
