"""
LangChain Agent Coordinator for Crypto Intelligence System
Orchestrates all agents and synthesizes results using Groq LLM
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.agents.base_agent import BaseAgent
from src.agents.social_agent import SocialSentimentAgent
from src.agents.news_agent import NewsSentimentAgent
from src.agents.chain_agent import ChainAgent
from src.agents.whale_agent import WhaleAgent
from src.agents.signal_agent import SignalAgent
from src.agents.risk_agent import RiskAgent
from src.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import timed, Timer, get_metrics

logger = get_logger(__name__)


class AgentCoordinator:
    """
    LangChain-based multi-agent coordinator
    
    Orchestrates 6 specialized agents:
    1. SocialSentimentAgent - Reddit sentiment
    2. NewsSentimentAgent - News sentiment
    3. ChainAgent - On-chain/orderbook data
    4. WhaleAgent - Whale tracking
    5. SignalAgent - Signal generation
    6. RiskAgent - Risk assessment
    
    Uses Groq's Mixtral-8x7b for final synthesis
    """
    
    SYSTEM_PROMPT = """You are an expert crypto market analyst coordinating multiple AI agents to generate trading signals.

Your task is to synthesize data from multiple sources and provide a clear, actionable trading recommendation.

You will receive data from:
1. Social Sentiment - Reddit and social media sentiment
2. News Sentiment - Crypto news article sentiment
3. On-Chain Data - Exchange orderbook, technical indicators
4. Whale Tracking - Large wallet movements

Based on divergences between retail sentiment and whale behavior, generate your analysis.

Key principles:
- Whale accumulation during retail fear often precedes upward moves
- Whale distribution during retail greed often precedes corrections
- Strong orderbook imbalances indicate near-term price pressure
- Align your confidence with the signal agreement across sources

Respond in JSON format with these fields:
- signal: BUY, SELL, or HOLD
- confidence: 0-100
- thesis: 2-3 sentence explanation
- key_factors: list of top 3 factors driving the signal
- risk_notes: any concerns or caveats
- timeframe: expected timeframe (e.g., "2-8 hours", "1-3 days")
"""
    
    def __init__(self):
        """Initialize agent coordinator"""
        self.settings = get_settings()
        
        # Initialize LLM
        self._llm: Optional[Any] = None
        if LANGCHAIN_AVAILABLE and self.settings.GROQ_API_KEY:
            try:
                self._llm = ChatGroq(
                    model=self.settings.LLM_MODEL,
                    api_key=self.settings.GROQ_API_KEY,
                    temperature=0.3,
                    max_tokens=1024
                )
                logger.info("groq_llm_initialized", model=self.settings.LLM_MODEL)
            except Exception as e:
                logger.warning(f"groq_init_error: {e}")
        
        # Initialize agents
        self.social_agent = SocialSentimentAgent()
        self.news_agent = NewsSentimentAgent()
        self.chain_agent = ChainAgent()
        self.whale_agent = WhaleAgent()
        self.signal_agent = SignalAgent()
        self.risk_agent = RiskAgent()
        
        # Agent execution tracking
        self._last_results: Dict[str, Any] = {}
        
        logger.info("agent_coordinator_initialized")
    
    @timed("coordinator_run_analysis")
    async def run_analysis(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Run complete multi-agent analysis
        
        Args:
            symbol: Trading pair to analyze
            
        Returns:
            Complete analysis with signal, risk, and reasoning
        """
        start_time = datetime.now()
        
        logger.info("analysis_started", symbol=symbol)
        
        try:
            # Step 1: Parallel data collection from all agents
            with Timer("parallel_agent_execution"):
                results = await self._parallel_agent_execution(symbol)
            
            social_result = results.get("social", {})
            news_result = results.get("news", {})
            chain_result = results.get("chain", {})
            whale_result = results.get("whale", {})
            
            # Step 2: Generate signal from collected data
            signal_result = self.signal_agent.generate_from_agents(
                social_result=social_result,
                news_result=news_result,
                chain_result=chain_result,
                whale_result=whale_result,
                symbol=symbol
            )
            
            # Step 3: Assess risk
            risk_result = self.risk_agent.assess_from_signal(
                signal_result=signal_result,
                chain_result=chain_result
            )
            
            # Step 4: LLM synthesis (optional)
            llm_reasoning = ""
            llm_signal = None
            
            if self._llm:
                try:
                    llm_response = await self._synthesize_with_llm(
                        social_result,
                        news_result,
                        chain_result,
                        whale_result,
                        signal_result,
                        symbol
                    )
                    llm_reasoning = llm_response.get("thesis", "")
                    llm_signal = llm_response
                except Exception as e:
                    logger.warning(f"llm_synthesis_error: {e}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build final result
            result = {
                "success": True,
                "symbol": symbol,
                
                # Primary signal
                "signal": signal_result.get("signal", "NEUTRAL"),
                "score": signal_result.get("score", 0.0),
                "confidence": signal_result.get("confidence", 50.0),
                "thesis": llm_reasoning or signal_result.get("thesis", ""),
                
                # Price data
                "price": chain_result.get("current_price", 0.0),
                "price_change_24h": chain_result.get("price_change_24h", 0.0),
                
                # Component breakdown
                "components": signal_result.get("components", {}),
                
                # Risk assessment
                "risk": {
                    "level": risk_result.get("risk_level", "medium"),
                    "score": risk_result.get("risk_score", 50.0),
                    "position_size": risk_result.get("recommended_position_size", 0.02),
                    "stop_loss_pct": risk_result.get("stop_loss_pct", 5.0),
                    "take_profit_pct": risk_result.get("take_profit_pct", 10.0),
                    "warnings": risk_result.get("warnings", [])
                },
                
                # Market context
                "market_context": {
                    "fear_greed": chain_result.get("fear_greed_index", 50),
                    "fear_greed_label": chain_result.get("fear_greed_label", "neutral"),
                    "volatility": chain_result.get("volatility", 0.0),
                    "orderbook_imbalance": chain_result.get("orderbook_imbalance", 0.0)
                },
                
                # Agent results summary
                "agent_summary": {
                    "social": {
                        "sentiment": social_result.get("sentiment_score", 0.0),
                        "posts_analyzed": social_result.get("post_count", 0)
                    },
                    "news": {
                        "sentiment": news_result.get("sentiment_score", 0.0),
                        "articles_analyzed": news_result.get("article_count", 0)
                    },
                    "whale": {
                        "accumulation_score": whale_result.get("accumulation_score", 0.0),
                        "sentiment": whale_result.get("whale_sentiment", "neutral")
                    },
                    "chain": {
                        "imbalance": chain_result.get("orderbook_imbalance", 0.0),
                        "technical_signals": chain_result.get("technical_signals", {})
                    }
                },
                
                # LLM analysis (if available)
                "llm_analysis": llm_signal,
                
                # Metadata
                "execution_time_ms": round(execution_time, 2),
                "timestamp": datetime.utcnow().isoformat(),
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            
            # Store for history
            self._last_results[symbol] = result
            
            logger.info(
                "analysis_complete",
                symbol=symbol,
                signal=result["signal"],
                confidence=result["confidence"],
                execution_time_ms=execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"analysis_error: {e}")
            return self._get_error_result(symbol, str(e))
    
    async def _parallel_agent_execution(
        self,
        symbol: str
    ) -> Dict[str, Dict[str, Any]]:
        """Execute all data-gathering agents in parallel"""
        
        tasks = [
            self.social_agent.safe_analyze(symbol),
            self.news_agent.safe_analyze(symbol),
            self.chain_agent.safe_analyze(symbol),
            self.whale_agent.safe_analyze(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_names = ["social", "news", "chain", "whale"]
        processed = {}
        
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                logger.warning(f"agent_{name}_exception: {result}")
                processed[name] = {"success": False, "error": str(result)}
            else:
                processed[name] = result
        
        return processed
    
    async def _synthesize_with_llm(
        self,
        social: Dict,
        news: Dict,
        chain: Dict,
        whale: Dict,
        signal: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """Use LLM to synthesize all agent outputs"""
        
        # Build context prompt
        context = f"""
Symbol: {symbol}

SOCIAL SENTIMENT:
- Score: {social.get('sentiment_score', 0):.2f}
- Label: {social.get('overall_sentiment', 'neutral')}
- Posts analyzed: {social.get('post_count', 0)}
- Trending topics: {list(social.get('trending_topics', {}).keys())[:5]}

NEWS SENTIMENT:
- Score: {news.get('sentiment_score', 0):.2f}  
- Label: {news.get('overall_sentiment', 'neutral')}
- Articles analyzed: {news.get('article_count', 0)}

ON-CHAIN DATA:
- Price: ${chain.get('current_price', 0):,.2f}
- 24h Change: {chain.get('price_change_24h', 0):.2f}%
- Orderbook Imbalance: {chain.get('orderbook_imbalance', 0):.3f}
- Fear & Greed Index: {chain.get('fear_greed_index', 50)}
- Technical Signals: {chain.get('technical_signals', {})}

WHALE ACTIVITY:
- Accumulation Score: {whale.get('accumulation_score', 0):.1f}
- Whale Sentiment: {whale.get('whale_sentiment', 'neutral')}
- Transaction Count: {whale.get('transaction_count', 0)}

GENERATED SIGNAL:
- Signal: {signal.get('signal', 'NEUTRAL')}
- Confidence: {signal.get('confidence', 50):.1f}%
- Components: Sentiment={signal.get('components', {}).get('sentiment', 0):.2f}, OnChain={signal.get('components', {}).get('onchain', 0):.2f}, Whale={signal.get('components', {}).get('whale', 0):.2f}

Please analyze this data and provide your trading recommendation in JSON format.
"""
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=context)
            ]
            
            response = await self._llm.ainvoke(messages)
            content = response.content
            
            # Parse JSON from response
            # Try to extract JSON if response contains other text
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)
            
            return {"thesis": content}
            
        except Exception as e:
            logger.warning(f"llm_parse_error: {e}")
            return {"thesis": "", "error": str(e)}
    
    async def get_trading_signal(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Get just the trading signal (simplified output)
        
        Args:
            symbol: Trading pair
            
        Returns:
            Simplified signal dict
        """
        result = await self.run_analysis(symbol)
        
        return {
            "symbol": symbol,
            "signal": result.get("signal"),
            "confidence": result.get("confidence"),
            "price": result.get("price"),
            "thesis": result.get("thesis"),
            "risk_level": result.get("risk", {}).get("level"),
            "timestamp": result.get("timestamp")
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        return {
            "social": self.social_agent.stats,
            "news": self.news_agent.stats,
            "chain": self.chain_agent.stats,
            "whale": self.whale_agent.stats,
            "signal": self.signal_agent.stats,
            "risk": self.risk_agent.stats
        }
    
    def _get_error_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            "success": False,
            "symbol": symbol,
            "error": error,
            "signal": "NEUTRAL",
            "confidence": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global coordinator instance
_coordinator_instance: Optional[AgentCoordinator] = None


def get_coordinator() -> AgentCoordinator:
    """Get global coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = AgentCoordinator()
    return _coordinator_instance
