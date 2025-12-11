"""
Tests for FastAPI endpoints
"""
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'timestamp' in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test root endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data


@pytest.mark.asyncio
async def test_generate_signal_endpoint():
    """Test signal generation endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/signal",
            json={"symbol": "BTC/USDT", "timeframe": "4h"},
            timeout=60.0
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert 'signal' in data
        assert 'confidence' in data
        assert 'thesis' in data
        assert 'key_factors' in data
        assert 'risk_factors' in data
        assert 'symbol' in data
        assert 'price' in data
        
        # Validate data types
        assert data['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= data['confidence'] <= 100
        assert isinstance(data['thesis'], str)
        assert isinstance(data['key_factors'], list)


@pytest.mark.asyncio
async def test_history_endpoint():
    """Test historical signals endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/signals/history",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'signals' in data
        assert 'count' in data
        assert isinstance(data['signals'], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
