"""
Streamlit Dashboard for Crypto Market Intelligence
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Crypto Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d2ff;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
        font-size: 32px;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
        font-size: 32px;
    }
    .signal-hold {
        color: #ffaa00;
        font-weight: bold;
        font-size: 32px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Multi-Modal Crypto Intelligence</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
show_details = st.sidebar.checkbox("Show detailed analysis", value=True)

# Add refresh button
if st.sidebar.button("🔄 Refresh Now") or auto_refresh:
    st.rerun()

# Fetch signal
try:
    with st.spinner("Generating signal..."):
        response = requests.post(
            f"{API_URL}/signal",
            json={"symbol": symbol, "timeframe": "4h"},
            timeout=120  # Increased timeout for first run (FinBERT model download)
        )
        
        if response.status_code == 200:
            signal = response.json()
            
            # Main signal display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_class = f"signal-{signal['signal'].lower()}"
                st.markdown(f"### Trading Signal")
                st.markdown(f'<p class="{signal_class}">{signal["signal"]}</p>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{signal['confidence']}%")
                
            with col3:
                st.metric("Price", f"${signal['price']:,.2f}")
            
            with col4:
                st.metric("Timeframe", signal['timeframe'])
            
            st.markdown("---")
            
            # Trading thesis
            st.markdown("### 📊 Trading Thesis")
            st.info(signal['thesis'])
            
            # Two columns for factors
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### ✅ Key Factors")
                for factor in signal['key_factors']:
                    st.markdown(f"- {factor}")
            
            with col_right:
                st.markdown("### ⚠️ Risk Factors")
                for risk in signal['risk_factors']:
                    st.markdown(f"- {risk}")
            
            st.markdown("---")
            
            # Position sizing
            st.markdown("### 💰 Position Sizing Recommendation")
            st.success(signal['position_sizing'])
            
            # Historical pattern
            if signal.get('historical_pattern'):
                st.markdown("### 📈 Historical Context")
                st.write(signal['historical_pattern'])
            
            st.markdown("---")
            
            # Detailed analysis (if enabled)
            if show_details and 'underlying_data' in signal:
                st.markdown("### 🔍 Detailed Multi-Modal Analysis")
                
                tab1, tab2, tab3 = st.tabs(["🐋 Whale Activity", "📊 Order Book", "😱 Sentiment"])
                
                with tab1:
                    whale_data = signal['underlying_data'].get('whale_sentiment', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Whale Score", f"{whale_data.get('whale_score', 0):.1f}")
                    with col2:
                        st.metric("Transactions", whale_data.get('whale_count', 0))
                    with col3:
                        st.metric("Volume", f"{whale_data.get('total_volume', 0):.2f} BTC")
                    
                    st.write(f"**Sentiment:** {whale_data.get('sentiment', 'NEUTRAL')}")
                    st.write(f"**Exchange Pressure:** {whale_data.get('exchange_pressure', 0):.2f} BTC")
                
                with tab2:
                    orderbook_data = signal['underlying_data'].get('orderbook', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        imbalance = orderbook_data.get('imbalance', 0)
                        st.metric("Bid/Ask Imbalance", f"{imbalance:.4f}")
                    with col2:
                        st.metric("Market Depth", f"{orderbook_data.get('market_depth_score', 0):.1f}/100")
                    with col3:
                        spoofing = "⚠️ YES" if orderbook_data.get('spoofing_detected') else "✅ NO"
                        st.metric("Spoofing", spoofing)
                    
                    # Gauge chart for imbalance
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = imbalance,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Order Book Imbalance"},
                        gauge = {
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "lightblue"},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "#ffcccc"},
                                {'range': [-0.3, 0.3], 'color': "#ffffcc"},
                                {'range': [0.3, 1], 'color': "#ccffcc"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    sentiment_data = signal['underlying_data'].get('retail_sentiment', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment Score", f"{sentiment_data.get('sentiment_score', 0):.2f}")
                    with col2:
                        st.metric("Fear/Greed", f"{sentiment_data.get('fear_greed_index', 50)}/100")
                    with col3:
                        st.metric("Posts Analyzed", sentiment_data.get('post_count', 0))
                    
                    # Fear & Greed gauge
                    fg_index = sentiment_data.get('fear_greed_index', 50)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = fg_index,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fear & Greed Index"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "#ff4444"},
                                {'range': [25, 45], 'color': "#ffaa44"},
                                {'range': [45, 55], 'color': "#ffff44"},
                                {'range': [55, 75], 'color': "#88ff44"},
                                {'range': [75, 100], 'color': "#44ff44"}
                            ],
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write(f"**Dominant Emotion:** {sentiment_data.get('dominant_emotion', 'neutral').upper()}")
                    st.write(f"**Volume:** {sentiment_data.get('volume', 'low').upper()}")
                    
                    if sentiment_data.get('top_keywords'):
                        st.write(f"**Top Keywords:** {', '.join(sentiment_data['top_keywords'][:5])}")
            
            # Metadata
            st.markdown("---")
            st.caption(f"Generated at: {signal['generated_at']} | Symbol: {signal['symbol']}")
            
        else:
            st.error(f"Failed to fetch signal: {response.status_code}")
            st.write(response.text)

except requests.exceptions.ConnectionError:
    st.error("⚠️ Cannot connect to API. Make sure the FastAPI server is running at " + API_URL)
    st.info("Start the server with: `uvicorn app.main:app --reload`")
except requests.exceptions.Timeout:
    st.error("⚠️ Request timed out. Signal generation is taking longer than expected.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

# Auto-refresh logic
if auto_refresh:
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Multi-Modal Crypto Intelligence System v1.0.0</p>
        <p>🐋 Whale Tracking | 📊 Order Book Analysis | 😱 Sentiment Analysis | 🤖 LLM Coordination</p>
    </div>
""", unsafe_allow_html=True)
