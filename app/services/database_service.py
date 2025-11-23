"""
PostgreSQL database service for signal storage and history
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import asyncpg, but make it optional
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning(
        'asyncpg_not_available',
        message='PostgreSQL not available, using in-memory storage'
    )


class DatabaseService:
    """
    PostgreSQL database service for persistent storage
    
    Stores trading signals, historical data, and backtest results
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.pool = None
        self.memory_store: List[Dict[str, Any]] = []  # Fallback storage
    
    async def connect(self):
        """Initialize database connection pool"""
        if not ASYNCPG_AVAILABLE:
            logger.info('database_service_initialized', backend='memory')
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.settings.DATABASE_URL,
                min_size=2,
                max_size=10
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info('database_service_initialized', backend='postgresql')
            
        except Exception as e:
            logger.error(
                'database_connection_failed',
                error=str(e),
                message='Falling back to in-memory storage'
            )
            self.pool = None
    
    async def _create_tables(self):
        """Create database tables"""
        if not self.pool:
            return
        
        create_signals_table = """
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            signal VARCHAR(10) NOT NULL,
            confidence INTEGER NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            timeframe VARCHAR(20),
            thesis TEXT,
            key_factors JSONB,
            risk_factors JSONB,
            underlying_data JSONB,
            generated_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_signals_table)
            logger.info('database_tables_created')
    
    async def store_signal(self, signal: Dict[str, Any]) -> int:
        """
        Store a trading signal
        
        Args:
            signal: Signal data dictionary
            
        Returns:
            Signal ID
        """
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    signal_id = await conn.fetchval(
                        """
                        INSERT INTO signals (
                            symbol, signal, confidence, price, timeframe,
                            thesis, key_factors, risk_factors, underlying_data,
                            generated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                        """,
                        signal.get('symbol'),
                        signal.get('signal'),
                        signal.get('confidence'),
                        float(signal.get('price', 0)),
                        signal.get('timeframe'),
                        signal.get('thesis'),
                        json.dumps(signal.get('key_factors', [])),
                        json.dumps(signal.get('risk_factors', [])),
                        json.dumps(signal.get('underlying_data', {})),
                        datetime.fromisoformat(signal.get('generated_at', datetime.now().isoformat()))
                    )
                    logger.info('signal_stored', signal_id=signal_id, symbol=signal.get('symbol'))
                    return signal_id
            else:
                # In-memory fallback
                signal_id = len(self.memory_store) + 1
                self.memory_store.append({**signal, 'id': signal_id})
                logger.info('signal_stored', signal_id=signal_id, symbol=signal.get('symbol'), backend='memory')
                return signal_id
                
        except Exception as e:
            logger.error('signal_store_error', error=str(e), signal=signal.get('symbol'))
            return -1
    
    async def get_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical signals
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of signals to return
            offset: Offset for pagination
            
        Returns:
            List of signal dictionaries
        """
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    if symbol:
                        rows = await conn.fetch(
                            """
                            SELECT * FROM signals
                            WHERE symbol = $1
                            ORDER BY generated_at DESC
                            LIMIT $2 OFFSET $3
                            """,
                            symbol, limit, offset
                        )
                    else:
                        rows = await conn.fetch(
                            """
                            SELECT * FROM signals
                            ORDER BY generated_at DESC
                            LIMIT $1 OFFSET $2
                            """,
                            limit, offset
                        )
                    
                    signals = [dict(row) for row in rows]
                    logger.info('signals_retrieved', count=len(signals), symbol=symbol)
                    return signals
            else:
                # In-memory fallback
                if symbol:
                    filtered = [s for s in self.memory_store if s.get('symbol') == symbol]
                else:
                    filtered = self.memory_store
                
                # Sort by generated_at in reverse
                sorted_signals = sorted(
                    filtered,
                    key=lambda x: x.get('generated_at', ''),
                    reverse=True
                )
                
                result = sorted_signals[offset:offset + limit]
                logger.info('signals_retrieved', count=len(result), symbol=symbol, backend='memory')
                return result
                
        except Exception as e:
            logger.error('signals_retrieve_error', error=str(e))
            return []
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info('database_service_closed')


# Global database instance
_db_instance: Optional[DatabaseService] = None


def get_db() -> DatabaseService:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseService()
    return _db_instance
