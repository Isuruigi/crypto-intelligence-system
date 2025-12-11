"""
Whale Activity View Component for Streamlit Dashboard
Displays whale tracking and on-chain analysis
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List


class WhaleView:
    """Component for displaying whale activity analysis"""
    
    def __init__(self):
        self.colors = {
            'accumulation': '#00ff00',
            'distribution': '#ff0000',
            'neutral': '#888888'
        }
    
    def render(self, whale_data: Dict[str, Any]):
        """
        Render whale activity view
        
        Args:
            whale_data: Dictionary containing whale metrics
        """
        st.markdown("### 🐋 Whale Activity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            whale_score = whale_data.get('whale_score', 0)
            st.metric(
                "Whale Score",
                f"{whale_score:.1f}",
                delta="Accumulating" if whale_score > 20 else "Distributing" if whale_score < -20 else "Neutral"
            )
        
        with col2:
            st.metric(
                "Large Transactions",
                whale_data.get('whale_count', 0)
            )
        
        with col3:
            volume = whale_data.get('total_volume', 0)
            st.metric(
                "Volume",
                f"{volume:.2f} BTC"
            )
        
        # Whale Score Gauge
        self._render_whale_gauge(whale_data.get('whale_score', 0))
        
        # Exchange Flow
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Exchange Pressure:**")
            pressure = whale_data.get('exchange_pressure', 0)
            if pressure > 0:
                st.error(f"🔴 {pressure:.2f} BTC flowing to exchanges")
            elif pressure < 0:
                st.success(f"🟢 {abs(pressure):.2f} BTC leaving exchanges")
            else:
                st.info("⚪ Neutral flow")
        
        with col2:
            st.markdown("**Whale Signal:**")
            signal = whale_data.get('signal', 'HOLD')
            if signal == 'BUY':
                st.success(f"🟢 {signal}")
            elif signal == 'SELL':
                st.error(f"🔴 {signal}")
            else:
                st.warning(f"🟡 {signal}")
    
    def _render_whale_gauge(self, score: float):
        """Render whale accumulation/distribution gauge"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Whale Accumulation Score"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [-100, -50], 'color': "#ff4444"},
                    {'range': [-50, -20], 'color': "#ffaa44"},
                    {'range': [-20, 20], 'color': "#ffff44"},
                    {'range': [20, 50], 'color': "#88ff44"},
                    {'range': [50, 100], 'color': "#44ff44"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_table(self, transactions: List[Dict[str, Any]]):
        """Render recent whale transactions table"""
        if not transactions:
            st.info("No recent whale transactions detected")
            return
        
        st.markdown("**Recent Large Transactions:**")
        
        for tx in transactions[:5]:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"📍 {tx.get('from', 'Unknown')[:20]}...")
            with col2:
                st.text(f"{tx.get('amount', 0):.2f} BTC")
            with col3:
                flow = tx.get('flow', 'unknown')
                if flow == 'in':
                    st.text("➡️ Exchange")
                elif flow == 'out':
                    st.text("⬅️ Cold Storage")
                else:
                    st.text("↔️ Transfer")
    
    def render_mini(self, whale_data: Dict[str, Any]):
        """Render compact whale view"""
        score = whale_data.get('whale_score', 0)
        
        if score > 20:
            label = "🟢 Accumulating"
        elif score < -20:
            label = "🔴 Distributing"
        else:
            label = "🟡 Neutral"
        
        st.markdown(f"**Whale Activity:** {label} ({score:.1f})")
        st.markdown(f"**Large Txs:** {whale_data.get('whale_count', 0)}")


def create_whale_view() -> WhaleView:
    """Factory function for whale view"""
    return WhaleView()
