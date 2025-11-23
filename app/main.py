"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.api.websocket import websocket_endpoint
from app.services.database_service import get_db
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info('application_starting', environment=settings.ENVIRONMENT)
    
    # Initialize database
    db = get_db()
    await db.connect()
    
    logger.info('application_started')
    
    yield
    
    # Shutdown
    logger.info('application_shutting_down')
    
    # Close database connection
    await db.close()
    
    logger.info('application_shutdown_complete')


# Create FastAPI app
app = FastAPI(
    title="Crypto Market Intelligence API",
    description="""
    Multi-modal crypto trading signal generator using:
    - **Whale Tracking**: On-chain analysis of smart money movements
    - **Order Book Analysis**: Market microstructure and depth
    - **Sentiment Analysis**: Social media sentiment with FinBERT
    - **LLM Coordination**: Groq API for signal synthesis
    
    Provides confidence-calibrated trading signals via REST and WebSocket.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="", tags=["signals"])

# Add WebSocket route
app.add_websocket_route("/ws/signals", websocket_endpoint)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
