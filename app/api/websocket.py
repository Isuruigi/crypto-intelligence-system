"""
WebSocket endpoint for real-time signals
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import List
import asyncio
import json

from app.agents.orchestrator import get_orchestrator, MultiAgentOrchestrator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info('websocket_connected', total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info('websocket_disconnected', total_connections=len(self.active_connections))
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error('websocket_send_error', error=str(e))
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        dead_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error('websocket_broadcast_error', error=str(e))
                dead_connections.append(connection)
        
        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    orchestrator: MultiAgentOrchestrator = Depends(get_orchestrator)
):
    """
    WebSocket endpoint for real-time trading signals
    
    Sends updated signals every 60 seconds
    
    Usage:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/signals');
    ws.onmessage = (event) => {
        const signal = JSON.parse(event.data);
        console.log(signal);
    };
    ```
    """
    await manager.connect(websocket)
    
    try:
        # Send initial signal
        logger.info('websocket_sending_initial_signal')
        initial_signal = await orchestrator.generate_signal("BTC/USDT")
        await manager.send_personal_message(initial_signal, websocket)
        
        # Keep connection alive and send updates
        while True:
            # Wait 60 seconds
            await asyncio.sleep(60)
            
            # Generate and send new signal
            logger.info('websocket_sending_update')
            signal = await orchestrator.generate_signal("BTC/USDT")
            await manager.send_personal_message(signal, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info('websocket_client_disconnected')
    except Exception as e:
        logger.error('websocket_error', error=str(e))
        manager.disconnect(websocket)
