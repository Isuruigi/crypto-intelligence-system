"""
FastAPI Application for Crypto Intelligence System
Production-ready API with WebSocket support
"""
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.agents.coordinator import get_coordinator
from src.data.storage.cache import init_cache
from src.utils.logger import get_logger
from src.utils.metrics import get_metrics

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("application_starting")
    
    # Initialize cache
    await init_cache()
    
    # Initialize coordinator (warms up agents)
    get_coordinator()
    
    logger.info("application_started", port=settings.API_PORT)
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Crypto Intelligence API",
    description="Multi-agent AI system for cryptocurrency market analysis and trading signals",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health and Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/v1/status")
async def system_status():
    """Get system status and agent statistics"""
    coordinator = get_coordinator()
    metrics = get_metrics()
    
    return {
        "status": "operational",
        "agents": coordinator.get_agent_stats(),
        "metrics": metrics.get_all_metrics(),
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# Signal Generation Endpoints
# =============================================================================

@app.get("/api/v1/signal/{symbol}")
async def get_signal(
    symbol: str = "BTC/USDT",
    include_details: bool = True
):
    """
    Get trading signal for a symbol
    
    Args:
        symbol: Trading pair (e.g., BTC/USDT, ETH/USDT)
        include_details: Include full analysis details
        
    Returns:
        Trading signal with confidence and reasoning
    """
    # Normalize symbol format
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        
        if include_details:
            result = await coordinator.run_analysis(symbol)
        else:
            result = await coordinator.get_trading_signal(symbol)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"signal_generation_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Signal generation failed: {str(e)}"
        )


@app.post("/api/v1/signal")
async def generate_signal(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    include_llm: bool = True
):
    """
    Generate trading signal (POST version)
    
    Args:
        symbol: Trading pair
        timeframe: Analysis timeframe
        include_llm: Include LLM reasoning
        
    Returns:
        Complete trading analysis
    """
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        result = await coordinator.run_analysis(symbol)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"signal_generation_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Signal generation failed: {str(e)}"
        )


@app.post("/api/v1/analyze")
async def full_analysis(
    symbol: str = "BTC/USDT",
    depth: str = "full"
):
    """
    Perform full multi-agent analysis
    
    Args:
        symbol: Trading pair
        depth: Analysis depth (full or quick)
        
    Returns:
        Complete analysis from all agents
    """
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        result = await coordinator.run_analysis(symbol)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"analysis_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# =============================================================================
# Sentiment Endpoints
# =============================================================================

@app.get("/api/v1/sentiment/{symbol}")
async def get_sentiment(
    symbol: str = "BTC/USDT",
    timeframe: str = "24h"
):
    """
    Get sentiment analysis for a symbol
    
    Args:
        symbol: Trading pair
        timeframe: Analysis timeframe
        
    Returns:
        Sentiment analysis from social and news sources
    """
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        
        # Run social and news agents
        import asyncio
        social_result, news_result = await asyncio.gather(
            coordinator.social_agent.safe_analyze(symbol),
            coordinator.news_agent.safe_analyze(symbol)
        )
        
        return {
            "symbol": symbol,
            "social": social_result,
            "news": news_result,
            "combined_score": (
                social_result.get("sentiment_score", 0) * 0.6 +
                news_result.get("sentiment_score", 0) * 0.4
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"sentiment_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


# =============================================================================
# Whale Activity Endpoints
# =============================================================================

@app.get("/api/v1/whale-activity/{symbol}")
async def get_whale_activity(
    symbol: str = "BTC/USDT",
    hours_back: int = 24
):
    """
    Get whale activity analysis
    
    Args:
        symbol: Trading pair
        hours_back: Hours to look back
        
    Returns:
        Whale movement analysis
    """
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        result = await coordinator.whale_agent.safe_analyze(symbol)
        
        return {
            "symbol": symbol,
            **result,
            "hours_analyzed": hours_back
        }
        
    except Exception as e:
        logger.error(f"whale_activity_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Whale activity analysis failed: {str(e)}"
        )


# =============================================================================
# Market Data Endpoints
# =============================================================================

@app.get("/api/v1/market/{symbol}")
async def get_market_data(symbol: str = "BTC/USDT"):
    """
    Get market data and on-chain metrics
    
    Args:
        symbol: Trading pair
        
    Returns:
        Market data including orderbook and technicals
    """
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    try:
        coordinator = get_coordinator()
        result = await coordinator.chain_agent.safe_analyze(symbol)
        
        return {
            "symbol": symbol,
            **result
        }
        
    except Exception as e:
        logger.error(f"market_data_error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Market data fetch failed: {str(e)}"
        )


# =============================================================================
# WebSocket Streaming
# =============================================================================

@app.websocket("/ws/stream/{symbol}")
async def websocket_stream(websocket: WebSocket, symbol: str = "BTC/USDT"):
    """
    WebSocket endpoint for real-time signal streaming
    
    Streams updates every 60 seconds
    """
    await websocket.accept()
    
    symbol = symbol.upper().replace("-", "/")
    if "/" not in symbol:
        symbol = f"{symbol}/USDT"
    
    logger.info("websocket_connected", symbol=symbol)
    
    try:
        coordinator = get_coordinator()
        
        while True:
            try:
                # Generate signal
                result = await coordinator.run_analysis(symbol)
                
                # Send to client
                await websocket.send_json({
                    "type": "signal",
                    "data": result,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Wait 60 seconds
                import asyncio
                await asyncio.sleep(60)
                
            except WebSocketDisconnect:
                logger.info("websocket_disconnected", symbol=symbol)
                break
            except Exception as e:
                logger.error(f"websocket_stream_error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                await asyncio.sleep(10)  # Wait before retry
                
    except Exception as e:
        logger.error(f"websocket_error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    General trading signals WebSocket
    
    Streams signals for multiple symbols
    """
    await websocket.accept()
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    logger.info("websocket_signals_connected")
    
    try:
        coordinator = get_coordinator()
        
        while True:
            try:
                import asyncio
                
                for symbol in symbols:
                    result = await coordinator.get_trading_signal(symbol)
                    
                    await websocket.send_json({
                        "type": "signal",
                        "symbol": symbol,
                        "data": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Wait 60 seconds between rounds
                await asyncio.sleep(60)
                
            except WebSocketDisconnect:
                logger.info("websocket_signals_disconnected")
                break
            except Exception as e:
                logger.error(f"websocket_signals_error: {e}")
                await asyncio.sleep(10)
                
    except Exception as e:
        logger.error(f"websocket_error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"unhandled_exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
