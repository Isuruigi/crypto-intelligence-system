"""
Quick setup verification script
Tests that all components can be imported and basic functionality works
"""
import sys
import asyncio


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core modules
        from app.config import get_settings
        from app.utils.logger import get_logger
        from app.utils.circuit_breaker import CircuitBreaker
        
        # Services
        from app.services.cache_service import get_cache
        from app.services.rate_limiter import get_rate_limiter
        from app.services.database_service import get_db
        
        # Models
        from app.models.requests import SignalRequest
        from app.models.responses import SignalResponse
        
        # Agents
        from app.agents.whale_tracker import WhaleTrackerAgent
        from app.agents.orderbook_analyzer import OrderbookAnalyzer
        from app.agents.sentiment_analyzer import SentimentAnalyzer
        from app.agents.coordinator import LLMCoordinator
        from app.agents.orchestrator import get_orchestrator
        
        # API
        from app.main import app
        from app.api.routes import router
        
        # Backtesting
        from backtesting.backtest_engine import BacktestEngine
        from backtesting.performance_metrics import calculate_sharpe_ratio
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from app.config import get_settings
        settings = get_settings()
        
        print(f"  - Groq API Key: {'✅ Set' if settings.GROQ_API_KEY else '❌ Not set'}")
        print(f"  - Groq Model: {settings.GROQ_MODEL}")
        print(f"  - Database URL: {settings.DATABASE_URL}")
        print(f"  - Redis URL: {settings.REDIS_URL}")
        print(f"  - Environment: {settings.ENVIRONMENT}")
        
        if not settings.GROQ_API_KEY:
            print("\n⚠️  WARNING: GROQ_API_KEY not set. LLM coordination will fail.")
            print("   Set it in your .env file or environment variables.")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def test_agents():
    """Test agent initialization"""
    print("\nTesting agent initialization...")
    
    try:
        from app.agents.whale_tracker import WhaleTrackerAgent
        from app.agents.orderbook_analyzer import OrderbookAnalyzer
        from app.agents.sentiment_analyzer import SentimentAnalyzer
        from app.agents.coordinator import LLMCoordinator
        
        whale = WhaleTrackerAgent()
        print("  ✅ Whale Tracker initialized")
        
        orderbook = OrderbookAnalyzer()
        print("  ✅ Orderbook Analyzer initialized")
        
        sentiment = SentimentAnalyzer()
        print("  ✅ Sentiment Analyzer initialized")
        
        coordinator = LLMCoordinator()
        print("  ✅ LLM Coordinator initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False


async def test_database_connection():
    """Test database connection"""
    print("\nTesting database connection...")
    
    try:
        from app.services.database_service import get_db
        
        db = get_db()
        await db.connect()
        
        print("  ✅ Database connection successful")
        
        await db.close()
        return True
        
    except Exception as e:
        print(f"  ⚠️  Database connection failed (this is OK if DB not running): {e}")
        print("     The system will use in-memory fallback")
        return True  # Don't fail the test


def test_dependencies():
    """Check if optional dependencies are available"""
    print("\nChecking optional dependencies...")
    
    deps = {
        'praw': 'Reddit API (PRAW)',
        'transformers': 'FinBERT (Transformers)',
        'torch': 'PyTorch',
        'ccxt': 'Exchange API (CCXT)',
        'groq': 'Groq API',
        'redis': 'Redis',
        'asyncpg': 'PostgreSQL (asyncpg)'
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} not available")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🤖 Crypto Intelligence System - Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(test_imports())
    results.append(test_config())
    results.append(await test_agents())
    results.append(await test_database_connection())
    test_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All core tests passed! System is ready to run.")
        print("\nNext steps:")
        print("  1. Ensure .env file has GROQ_API_KEY set")
        print("  2. Start with: python -m uvicorn app.main:app --reload")
        print("  3. Or use Docker: docker-compose up --build")
        print("  4. Dashboard: streamlit run dashboard/streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
