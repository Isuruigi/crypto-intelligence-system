"""
Sentiment View Component for Streamlit Dashboard
Displays sentiment analysis visualizations
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional


class SentimentView:
    """Component for displaying sentiment analysis results"""
    
    def __init__(self):
        self.colors = {
            'positive': '#00ff00',
            'negative': '#ff0000',
            'neutral': '#ffaa00',
            'fear': '#ff4444',
            'greed': '#44ff44'
        }
    
    def render(self, sentiment_data: Dict[str, Any]):
        """
        Render sentiment analysis view
        
        Args:
            sentiment_data: Dictionary containing sentiment metrics
        """
        st.markdown("### 😱 Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = sentiment_data.get('sentiment_score', 0)
            st.metric(
                "Sentiment Score",
                f"{score:.2f}",
                delta=f"{sentiment_data.get('sentiment_velocity', 0):.2f}/hr"
            )
        
        with col2:
            fg_index = sentiment_data.get('fear_greed_index', 50)
            st.metric("Fear/Greed Index", f"{fg_index}/100")
        
        with col3:
            st.metric(
                "Posts Analyzed",
                sentiment_data.get('post_count', 0)
            )
        
        # Fear & Greed Gauge
        self._render_fear_greed_gauge(sentiment_data.get('fear_greed_index', 50))
        
        # Sentiment Distribution
        if 'distribution' in sentiment_data:
            self._render_distribution(sentiment_data['distribution'])
        
        # Top Keywords
        if sentiment_data.get('top_keywords'):
            st.markdown("**Top Keywords:**")
            keywords = sentiment_data['top_keywords'][:5]
            st.write(", ".join(keywords))
    
    def _render_fear_greed_gauge(self, value: int):
        """Render Fear & Greed gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fear & Greed Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': self.colors['fear']},
                    {'range': [25, 45], 'color': "#ffaa44"},
                    {'range': [45, 55], 'color': "#ffff44"},
                    {'range': [55, 75], 'color': "#88ff44"},
                    {'range': [75, 100], 'color': self.colors['greed']}
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_distribution(self, distribution: Dict[str, int]):
        """Render sentiment distribution pie chart"""
        labels = list(distribution.keys())
        values = list(distribution.values())
        
        colors = [
            self.colors.get(label.lower(), '#888888')
            for label in labels
        ]
        
        fig = px.pie(
            names=labels,
            values=values,
            title="Sentiment Distribution",
            color_discrete_sequence=colors
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_mini(self, sentiment_data: Dict[str, Any]):
        """Render compact sentiment view"""
        score = sentiment_data.get('sentiment_score', 0)
        fg = sentiment_data.get('fear_greed_index', 50)
        
        if score > 0.2:
            sentiment_label = "🟢 Bullish"
        elif score < -0.2:
            sentiment_label = "🔴 Bearish"
        else:
            sentiment_label = "🟡 Neutral"
        
        st.markdown(f"**Sentiment:** {sentiment_label} ({score:.2f})")
        st.markdown(f"**Fear/Greed:** {fg}/100")


def create_sentiment_view() -> SentimentView:
    """Factory function for sentiment view"""
    return SentimentView()
