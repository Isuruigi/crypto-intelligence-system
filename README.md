# 🤖 Multi-Modal Crypto Market Intelligence System

> **Production-ready crypto trading signal generator combining on-chain whale tracking, order book microstructure analysis, social sentiment, and LLM-based coordination using Groq API.**

## 🌟 Features

- **🐋 Whale Tracking**: Monitors large on-chain transactions using Blockchain.com API
- **📊 Order Book Analysis**: Market microstructure analysis with CCXT, spoofing detection
- **😱 Sentiment Analysis**: Reddit sentiment with FinBERT + Fear & Greed Index
- **🤖 LLM Coordination**: Groq API (llama-3.1-70b-versatile) for multi-modal signal synthesis
- **⚡ Real-time Streaming**: WebSocket support for live signal updates
- **📈 Backtesting Framework**: Complete performance validation with Sharpe ratio, profit factor
- **🎨 Interactive Dashboard**: Beautiful Streamlit UI with gauge charts and multi-tab analysis
- **🐳 Docker Ready**: Full containerization with PostgreSQL and Redis
- **✅ Production Features**: Circuit breakers, rate limiting, caching, structured logging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│              (Real-time Visualization)                       │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend                             │
│         (REST API + WebSocket Streaming)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Multi-Agent Orchestrator                        │
└─┬────────────┬────────────┬────────────┬────────────────────┘
  │            │            │            │
  │            │            │            │
┌─▼────────┐ ┌─▼──────────┐ ┌─▼────────┐ ┌─▼──────────────────┐
│  Whale   │ │ Orderbook  │ │Sentiment │ │  LLM Coordinator   │
│ Tracker  │ │ Analyzer   │ │Analyzer  │ │  (Groq API)        │
└──────────┘ └────────────┘ └──────────┘ └────────────────────┘
     │             │             │                │
     │             │             │                │
┌────▼─────┐  ┌───▼─────┐  ┌────▼────┐      ┌───▼─────┐
│Blockchain│  │  CCXT   │  │ Reddit  │      │  Groq   │
│   .com   │  │Binance  │  │ PRAW    │      │   API   │
│          │  │Coinbase │  │FinBERT  │      │llama-3.1│
└──────────┘  └─────────┘  └─────────┘      └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (installed ✅)
- Python 3.10+ (for local development)
- Groq API key (provided ✅)

### Option 1: Docker (Recommended)

1. **Clone and navigate to project**:
   ```bash
   cd "d:\Projects\Sentiment analyzer\crypto-intelligence-system"
   ```

2. **Create `.env` file** (already created with your Groq API key):
   ```bash
   # Copy example and edit if needed
   cp .env.example .env
   ```

3. **Start all services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the system**:
   - **Dashboard**: http://localhost:8501
   - **API Docs**: http://localhost:8000/docs
   - **WebSocket**: ws://localhost:8000/ws/signals

### Option 2: Local Development

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup verification**:
   ```bash
   python setup_test.py
   ```

4. **Start FastAPI backend**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. **Start Streamlit dashboard** (new terminal):
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```

## 📡 API Usage

### Generate Trading Signal

```bash
curl -X POST "http://localhost:8000/signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC/USDT", "timeframe": "4h"}'
```

### Get Historical Signals

```bash
curl "http://localhost:8000/signals/history?symbol=BTC/USDT&limit=10"
```

### WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals');

ws.onmessage = (event) => {
    const signal = JSON.parse(event.data);
    console.log('New signal:', signal.signal, 'Confidence:', signal.confidence);
};
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/signal",
    json={"symbol": "BTC/USDT", "timeframe": "4h"}
)

signal = response.json()
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']}%")
print(f"Thesis: {signal['thesis']}")
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v --cov=app
```

Run specific test modules:
```bash
pytest tests/test_agents.py -v
pytest tests/test_api.py -v
pytest tests/test_backtesting.py -v
```

## 📊 System Components

### 1. Whale Tracker Agent
- **Data Source**: Blockchain.com API (free, no key required)
- **Metrics**: Accumulation score (-100 to +100), transaction count, volume
- **Algorithm**: Analyzes input/output patterns to detect accumulation vs distribution
- **Caching**: 60-second TTL

### 2. Orderbook Analyzer
- **Data Source**: CCXT (Binance, Coinbase)
- **Metrics**: Bid/ask imbalance, market depth, spread
- **Features**: Spoofing detection, support/resistance levels
- **Caching**: 30-second TTL

### 3. Sentiment Analyzer
- **Data Sources**: Reddit (PRAW), Alternative.me Fear & Greed Index
- **Model**: FinBERT (ProsusAI/finbert) for financial sentiment
- **Metrics**: Sentiment score (-1 to +1), Fear/Greed Index (0-100)
- **Caching**: 300-second TTL

### 4. LLM Coordinator
- **Model**: Groq API - llama-3.1-70b-versatile
- **Strategy**: Divergence-based contrarian signals
- **Key Logic**: 
  - Whale accumulation + retail fear = BULLISH
  - Whale distribution + retail greed = BEARISH
  - Divergence >50 points = High confidence
- **Output**: BUY/SELL/HOLD with confidence (0-100)

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# Groq API (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile

# Database (Optional - uses in-memory fallback)
DATABASE_URL=postgresql://postgres:password@localhost:5432/signals
REDIS_URL=redis://localhost:6379

# Rate Limits
REDDIT_RATE_LIMIT=60
CCXT_RATE_LIMIT=1200
BLOCKCHAIN_RATE_LIMIT=20

# Cache TTL
CACHE_WHALE_DATA=60
CACHE_ORDERBOOK=30
CACHE_SENTIMENT=300
```

## 🎯 Signal Generation Logic

The system uses a multi-modal divergence strategy:

1. **Data Collection** (Concurrent):
   - Whale transactions from blockchain
   - Order book depth from exchanges
   - Social sentiment from Reddit + Fear/Greed

2. **Divergence Detection**:
   - Calculate gap between whale behavior and retail sentiment
   - High divergence (>50 points) = Strong contrarian signal

3. **LLM Synthesis**:
   - Groq API analyzes all data with structured prompt
   - Generates trading thesis with reasoning
   - Provides confidence score and risk factors

4. **Output**:
   - Signal: BUY/SELL/HOLD
   - Confidence: 0-100
   - Timeframe: 2-48 hours
   - Position sizing: Conservative/Moderate/Aggressive

## 📈 Performance Metrics

The backtesting framework calculates:

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max drawdown

## 🛠️ Development

### Project Structure

```
crypto-intelligence-system/
├── app/
│   ├── agents/          # 4 AI agents + orchestrator
│   ├── api/             # REST + WebSocket endpoints
│   ├── models/          # Pydantic schemas
│   ├── services/        # Cache, rate limiter, database
│   ├── utils/           # Logger, circuit breaker
│   ├── config.py        # Configuration management
│   └── main.py          # FastAPI app
├── dashboard/           # Streamlit UI
├── backtesting/         # Backtesting engine
├── tests/               # Unit tests
├── docker-compose.yml   # Docker orchestration
└── requirements.txt     # Python dependencies
```

### Adding New Features

1. **New Agent**: Extend `BaseAgent` in `app/agents/`
2. **New Endpoint**: Add route in `app/api/routes.py`
3. **New Metric**: Update `backtesting/performance_metrics.py`

## ⚠️ Important Notes

- **Not Financial Advice**: This is a signal generator for educational purposes
- **API Rate Limits**: 
  - Groq: 30 requests/minute
  - CCXT: 1200 requests/minute
  - Blockchain.com: 20 requests/minute
- **Production Use**: 
  - Enable authentication
  - Use HTTPS
  - Set proper CORS origins
  - Monitor API costs

## 🐛 Troubleshooting

### FinBERT model not downloading
```bash
# Manually download
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModel.from_pretrained('ProsusAI/finbert')"
```

### Redis connection failed
System automatically falls back to in-memory cache. For full functionality, ensure Redis is running:
```bash
docker-compose up redis
```

### Database connection failed
System uses in-memory storage as fallback. For persistence:
```bash
docker-compose up db
```

### Groq API errors
Check your API key in `.env` file and verify rate limits (30 req/min).

## 📚 Additional Resources

- **Groq Documentation**: https://console.groq.com/docs
- **CCXT Manual**: https://docs.ccxt.com/
- **FinBERT**: https://huggingface.co/ProsusAI/finbert
- **PRAW Reddit**: https://praw.readthedocs.io/

## 📄 License

MIT License - See LICENSE file for details

## 🙌 Credits

Built with:
- Groq API (llama-3.1-70b-versatile)
- FastAPI
- Streamlit
- FinBERT
- CCXT
- Docker

---

**Made with ❤️ for crypto traders | ⚠️ Not financial advice | Trade responsibly**
