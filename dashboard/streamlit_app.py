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
    .signal-buy, .signal-strong_buy {
        color: #00ff00;
        font-weight: bold;
        font-size: 32px;
    }
    .signal-sell, .signal-strong_sell {
        color: #ff0000;
        font-weight: bold;
        font-size: 32px;
    }
    .signal-neutral, .signal-hold {
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
            f"{API_URL}/api/v1/signal",
            params={"symbol": symbol, "timeframe": "4h"},
            timeout=120  # Increased timeout for first run (FinBERT model download)
        )
        
        if response.status_code == 200:
            signal = response.json()
            
            # Main signal display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_type = signal.get('signal', 'NEUTRAL')
                signal_class = f"signal-{signal_type.lower()}"
                st.markdown(f"### Trading Signal")
                st.markdown(f'<p class="{signal_class}">{signal_type}</p>', unsafe_allow_html=True)
            
            with col2:
                confidence = signal.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1f}%")
                
            with col3:
                price = signal.get('price', 0)
                price_change = signal.get('price_change_24h', 0)
                st.metric("Price", f"${price:,.2f}", f"{price_change:+.2f}%")
            
            with col4:
                risk_level = signal.get('risk', {}).get('level', 'medium')
                st.metric("Risk Level", risk_level.upper())
            
            st.markdown("---")
            
            # Trading thesis
            thesis = signal.get('thesis', 'No thesis available')
            if thesis:
                st.markdown("### 📊 Trading Thesis")
                st.info(thesis)
            
            st.markdown("---")
            
            # Risk Management
            risk = signal.get('risk', {})
            st.markdown("### 💰 Risk Management")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                position_size = risk.get('position_size', 0.02)
                st.metric("Position Size", f"{position_size*100:.1f}%")
            with col_r2:
                stop_loss = risk.get('stop_loss_pct', 5)
                st.metric("Stop Loss", f"{stop_loss:.1f}%")
            with col_r3:
                take_profit = risk.get('take_profit_pct', 10)
                st.metric("Take Profit", f"{take_profit:.1f}%")
            
            # Warnings
            warnings = risk.get('warnings', [])
            if warnings:
                st.markdown("### ⚠️ Risk Warnings")
                for w in warnings:
                    st.warning(w)
            
            st.markdown("---")
            
            # Component Scores
            components = signal.get('components', {})
            if components:
                st.markdown("### 📈 Component Scores")
                
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    social = components.get('social', 0)
                    st.metric("Social", f"{(social+1)/2*100:.1f}%")
                with col_c2:
                    news = components.get('news', 0)
                    st.metric("News", f"{(news+1)/2*100:.1f}%")
                with col_c3:
                    chain = components.get('chain', 0)
                    st.metric("On-Chain", f"{(chain+1)/2*100:.1f}%")
                with col_c4:
                    whale = components.get('whale', 0)
                    st.metric("Whale", f"{(whale+1)/2*100:.1f}%")
                
                # Component bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Social', 'News', 'On-Chain', 'Whale'],
                        y=[(social+1)/2*100, (news+1)/2*100, (chain+1)/2*100, (whale+1)/2*100],
                        marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
                    )
                ])
                fig.update_layout(
                    title="Component Performance",
                    yaxis_title="Score (%)",
                    yaxis_range=[0, 100],
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed analysis (if enabled)
            if show_details:
                st.markdown("### 🔍 Detailed Multi-Modal Analysis")
                
                tab1, tab2, tab3 = st.tabs(["🐋 Whale Activity", "📊 Market Data", "😱 Sentiment"])
                
                with tab1:
                    agent_summary = signal.get('agent_summary', {})
                    whale_data = agent_summary.get('whale', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accumulation Score", f"{whale_data.get('accumulation_score', 0):.2f}")
                    with col2:
                        st.metric("Whale Sentiment", whale_data.get('sentiment', 'neutral').upper())
                
                with tab2:
                    market_context = signal.get('market_context', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fg = market_context.get('fear_greed', 50)
                        st.metric("Fear & Greed", f"{fg}/100")
                    with col2:
                        volatility = market_context.get('volatility', 0)
                        st.metric("Volatility", f"{volatility:.2f}%")
                    with col3:
                        imbalance = market_context.get('orderbook_imbalance', 0)
                        st.metric("Orderbook Imbalance", f"{imbalance:.4f}")
                    
                    # Fear & Greed gauge
                    fg_index = market_context.get('fear_greed', 50)
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
                
                with tab3:
                    agent_summary = signal.get('agent_summary', {})
                    social_data = agent_summary.get('social', {})
                    news_data = agent_summary.get('news', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Social Sentiment**")
                        st.metric("Score", f"{social_data.get('sentiment', 0):.2f}")
                        st.write(f"Posts Analyzed: {social_data.get('posts_analyzed', 0)}")
                    with col2:
                        st.markdown("**News Sentiment**")
                        st.metric("Score", f"{news_data.get('sentiment', 0):.2f}")
                        st.write(f"Articles Analyzed: {news_data.get('articles_analyzed', 0)}")
            
            # Metadata
            st.markdown("---")
            timestamp = signal.get('generated_at', signal.get('timestamp', 'N/A'))
            execution_time = signal.get('execution_time_ms', 0)
            st.caption(f"Generated at: {timestamp} | Execution Time: {execution_time:.0f}ms | Symbol: {signal.get('symbol', symbol)}")
            
        else:
            st.error(f"Failed to fetch signal: {response.status_code}")
            st.write(response.text)

except requests.exceptions.ConnectionError:
    st.error("⚠️ Cannot connect to API. Make sure the FastAPI server is running at " + API_URL)
    st.info("Start the server with: `python run_api.py`")
except requests.exceptions.Timeout:
    st.error("⚠️ Request timed out. Signal generation is taking longer than expected.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

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
