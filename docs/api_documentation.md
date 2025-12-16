# 📡 API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Generate Trading Signal

**POST** `/signal`

Generate a trading signal for a specific trading pair.

#### Request
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "4h"
}
```

#### Response
```json
{
  "symbol": "BTC/USDT",
  "signal": "BUY",
  "confidence": 78,
  "price": 96025.50,
  "timeframe": "4-8 hours",
  "thesis": "Whale accumulation detected while retail sentiment shows fear...",
  "key_factors": [
    "Whale accumulation score: +65",
    "Fear & Greed Index: 28 (Fear)",
    "Order book imbalance: +0.15 (buy pressure)"
  ],
  "risk_factors": [
    "High market volatility",
    "Potential resistance at $98,000"
  ],
  "position_sizing": "Moderate (2-3% of portfolio)",
  "generated_at": "2024-12-11T12:00:00Z"
}
```

---

### 2. Get Signal History

**GET** `/signals/history`

Retrieve historical signals.

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| symbol | string | - | Trading pair filter |
| limit | int | 10 | Max results |

#### Response
```json
{
  "signals": [
    {
      "symbol": "BTC/USDT",
      "signal": "BUY",
      "confidence": 78,
      "generated_at": "2024-12-11T12:00:00Z"
    }
  ]
}
```

---

### 3. Health Check

**GET** `/health`

Check system health status.

#### Response
```json
{
  "status": "healthy",
  "components": {
    "api": true,
    "agents": true,
    "cache": true,
    "database": true
  },
  "timestamp": "2024-12-11T12:00:00Z"
}
```

---

### 4. WebSocket Streaming

**WS** `ws://localhost:8000/ws/signals`

Real-time signal streaming.

#### JavaScript Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals');

ws.onmessage = (event) => {
    const signal = JSON.parse(event.data);
    console.log('New signal:', signal.signal, signal.confidence);
};
```

---

## Python Client Example

```python
import requests

# Generate signal
response = requests.post(
    "http://localhost:8000/signal",
    json={"symbol": "BTC/USDT", "timeframe": "4h"}
)
signal = response.json()

print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Thesis: {signal['thesis']}")
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| /signal | 30 req/min |
| /signals/history | 60 req/min |
| WebSocket | 1 connection/client |

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid request parameters |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service temporarily unavailable |
