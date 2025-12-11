"""
Signals View Component for Streamlit Dashboard
Displays trading signals and history
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from datetime import datetime


class SignalsView:
    """Component for displaying trading signals"""
    
    def __init__(self):
        self.signal_colors = {
            'BUY': '#00ff00',
            'SELL': '#ff0000',
            'HOLD': '#ffaa00'
        }
    
    def render(self, signal_data: Dict[str, Any]):
        """
        Render trading signal view
        
        Args:
            signal_data: Dictionary containing signal data
        """
        st.markdown("### 📊 Trading Signal")
        
        # Main signal display
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            signal_color = self.signal_colors.get(signal, '#888888')
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; 
                     background-color: {signal_color}20; border-radius: 10px;
                     border: 2px solid {signal_color};'>
                    <h1 style='color: {signal_color}; margin: 0;'>{signal}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence}%")
        
        with col3:
            st.metric("Price", f"${signal_data.get('price', 0):,.2f}")
        
        with col4:
            st.metric("Timeframe", signal_data.get('timeframe', '4h'))
        
        # Trading thesis
        st.markdown("---")
        st.markdown("**📝 Trading Thesis:**")
        st.info(signal_data.get('thesis', 'No thesis available'))
        
        # Key factors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Key Factors:**")
            for factor in signal_data.get('key_factors', []):
                st.markdown(f"- {factor}")
        
        with col2:
            st.markdown("**⚠️ Risk Factors:**")
            for risk in signal_data.get('risk_factors', []):
                st.markdown(f"- {risk}")
        
        # Position sizing
        st.markdown("---")
        st.success(f"**Position Sizing:** {signal_data.get('position_sizing', 'Conservative')}")
    
    def render_confidence_gauge(self, confidence: float, signal: str):
        """Render confidence gauge"""
        color = self.signal_colors.get(signal, 'blue')
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Signal Confidence ({signal})"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "#ffcccc"},
                    {'range': [30, 60], 'color': "#ffffcc"},
                    {'range': [60, 100], 'color': "#ccffcc"}
                ],
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_history(self, signals: List[Dict[str, Any]]):
        """Render signal history table"""
        if not signals:
            st.info("No signal history available")
            return
        
        st.markdown("### 📜 Signal History")
        
        for signal in signals[:10]:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            
            with col1:
                sig = signal.get('signal', 'HOLD')
                color = self.signal_colors.get(sig, '#888')
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>{sig}</span>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.text(f"{signal.get('confidence', 0)}%")
            
            with col3:
                timestamp = signal.get('generated_at', '')
                if isinstance(timestamp, datetime):
                    st.text(timestamp.strftime("%H:%M"))
                else:
                    st.text(str(timestamp)[:10])
            
            with col4:
                st.text(signal.get('symbol', 'BTC/USDT'))
        
        st.markdown("---")
    
    def render_performance_summary(self, metrics: Dict[str, Any]):
        """Render performance summary from backtesting"""
        st.markdown("### 📈 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col3:
            st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.1f}%")
        
        with col4:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
    
    def render_mini(self, signal_data: Dict[str, Any]):
        """Render compact signal view"""
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        
        color = self.signal_colors.get(signal, '#888')
        st.markdown(f"""
            <span style='color: {color}; font-weight: bold; font-size: 24px;'>
                {signal}
            </span>
            <span style='color: #888;'> ({confidence}% confidence)</span>
        """, unsafe_allow_html=True)


def create_signals_view() -> SignalsView:
    """Factory function for signals view"""
    return SignalsView()
