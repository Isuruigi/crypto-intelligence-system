"""
FastAPI REST endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from datetime import datetime

from app.models.requests import SignalRequest, BacktestRequest
from app.models.responses import SignalResponse, HistoricalSignalsResponse, HealthResponse
from app.agents.orchestrator import get_orchestrator, MultiAgentOrchestrator
from app.services.database_service import get_db, DatabaseService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/signal", response_model=SignalResponse, summary="Generate trading signal")
async def generate_signal(
    request: SignalRequest,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    Generate a comprehensive trading signal for the given symbol
    
    - **symbol**: Trading pair (e.g., BTC/USDT)
    - **timeframe**: Analysis timeframe (1h, 4h, 24h)
    
    Returns:
    - Trading signal (BUY/SELL/HOLD)
    - Confidence level (0-100)
    - Trading thesis and analysis
    - Underlying data from all agents
    """
    try:
        logger.info('api_generate_signal_request', symbol=request.symbol, timeframe=request.timeframe)
        
        signal =await orchestrator.generate_signal(symbol=request.symbol)
        
        return signal
        
    except Exception as e:
        logger.error('api_generate_signal_error', error=str(e), symbol=request.symbol)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate signal: {str(e)}"
        )


@router.get("/signals/history", response_model=HistoricalSignalsResponse, summary="Get historical signals")
async def get_signal_history(
    symbol: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: DatabaseService = Depends(get_db)
):
    """
    Retrieve historical trading signals
    
    - **symbol**: Filter by symbol (optional)
    - **limit**: Maximum number of signals to return (default: 50)
    - **offset**: Offset for pagination (default: 0)
    
    Returns:
    - List of historical signals with metadata
    """
    try:
        logger.info('api_history_request', symbol=symbol, limit=limit)
        
        signals = await db.get_signals(symbol=symbol, limit=limit, offset=offset)
        
        return {
            'signals': signals,
            'count': len(signals),
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error('api_history_error', error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve signals: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Health check endpoint
    
    Returns:
    - Service status
    - Version
    - Timestamp
    """
    return {
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }


@router.get("/", summary="API root")
async def root():
    """
    API root endpoint
    
    Returns basic information about the API
    """
    return {
        'name': 'Crypto Market Intelligence API',
        'version': '1.0.0',
        'description': 'Multi-modal crypto trading signal generator',
        'endpoints': {
            'generate_signal': '/signal',
            'historical_signals': '/signals/history',
            'health': '/health',
            'docs': '/docs'
        }
    }
