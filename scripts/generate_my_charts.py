"""
Chart Generator for Crypto Intelligence System
Generates poster-ready charts at 300 DPI from collected data

Usage:
    python -m scripts.generate_my_charts
"""
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not installed. Run: pip install plotly kaleido")

# File paths
DATA_DIR = project_root / "data"
CHARTS_DIR = DATA_DIR / "my_charts"
SIGNALS_CSV = DATA_DIR / "real_results" / "my_signals.csv"
COMPONENT_SCORES_CSV = DATA_DIR / "poster_component_scores.csv"
SIGNAL_DISTRIBUTION_CSV = DATA_DIR / "my_signal_distribution.csv"
PERFORMANCE_BY_SYMBOL_CSV = DATA_DIR / "performance_by_symbol.csv"


def ensure_charts_dir():
    """Create charts directory if it doesn't exist"""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filepath: Path) -> List[Dict]:
    """Load data from CSV file"""
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def create_component_scores_chart():
    """Create component scores bar chart"""
    print("Creating component scores chart...")
    
    data = load_csv(COMPONENT_SCORES_CSV)
    if not data:
        print("   Run 'python -m scripts.analyze_my_results' first.")
        return False
    
    components = [row['Component'] for row in data]
    scores = [float(row['Score']) for row in data]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']  # green, blue, purple, orange
    
    fig = go.Figure(data=[
        go.Bar(
            x=components,
            y=scores,
            text=[f"{s:.1f}%" for s in scores],
            textposition='outside',
            marker_color=colors[:len(components)]
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Component Performance - My System",
            font=dict(size=24),
            x=0.5
        ),
        xaxis_title="Component",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(size=14),
        showlegend=False,
        width=800,
        height=600
    )
    
    output_path = CHARTS_DIR / "component_scores.png"
    fig.write_image(str(output_path), scale=3)  # 300 DPI
    print(f"   [OK] Saved: {output_path}")
    return True


def create_signal_distribution_chart():
    """Create signal distribution pie chart"""
    print("Creating signal distribution chart...")
    
    data = load_csv(SIGNAL_DISTRIBUTION_CSV)
    if not data:
        print("   Run 'python -m scripts.analyze_my_results' first.")
        return False
    
    labels = [row['Signal_Type'] for row in data]
    counts = [int(row['Count']) for row in data]
    
    # Color mapping
    color_map = {
        'STRONG_BUY': '#27ae60',
        'BUY': '#2ecc71',
        'NEUTRAL': '#95a5a6',
        'SELL': '#e74c3c',
        'STRONG_SELL': '#c0392b'
    }
    colors = [color_map.get(label, '#3498db') for label in labels]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=counts,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=colors),
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Signal Distribution - My 100 Signals",
            font=dict(size=24),
            x=0.5
        ),
        template="plotly_white",
        font=dict(size=14),
        showlegend=True,
        width=800,
        height=600
    )
    
    output_path = CHARTS_DIR / "signal_distribution.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_confidence_by_symbol_chart():
    """Create confidence by symbol bar chart"""
    print("Creating confidence by symbol chart...")
    
    data = load_csv(PERFORMANCE_BY_SYMBOL_CSV)
    if not data:
        print("   Run 'python -m scripts.analyze_my_results' first.")
        return False
    
    symbols = [row['symbol'] for row in data]
    confidences = [float(row['avg_confidence']) for row in data]
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=confidences,
            text=[f"{c:.1f}%" for c in confidences],
            textposition='outside',
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Confidence by Symbol",
            font=dict(size=24),
            x=0.5
        ),
        xaxis_title="Symbol",
        yaxis_title="Average Confidence (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(size=14),
        width=800,
        height=600
    )
    
    output_path = CHARTS_DIR / "confidence_by_symbol.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_confidence_timeline_chart():
    """Create confidence timeline chart"""
    print("Creating confidence timeline chart...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        print("   Run 'python -m scripts.collect_my_results' first.")
        return False
    
    # Filter successful signals
    successful = [d for d in data if d.get('success', 'True').lower() == 'true']
    
    # Group by symbol
    symbols = list(set(d['symbol'] for d in successful))
    color_map = {
        'BTC': '#f39c12',
        'ETH': '#3498db',
        'SOL': '#9b59b6',
        'ADA': '#1abc9c',
        'XRP': '#e74c3c'
    }
    
    fig = go.Figure()
    
    for symbol in symbols:
        sym_data = [d for d in successful if d['symbol'] == symbol]
        timestamps = list(range(len(sym_data)))  # Use index for x-axis
        confidences = [float(d['confidence']) for d in sym_data]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='lines+markers',
            name=symbol,
            line=dict(color=color_map.get(symbol, '#666')),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=dict(
            text="Confidence Trend - My Collection",
            font=dict(size=24),
            x=0.5
        ),
        xaxis_title="Signal Index",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1000,
        height=600
    )
    
    output_path = CHARTS_DIR / "confidence_timeline.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_system_radar_chart():
    """Create system profile radar chart"""
    print("Creating system radar chart...")
    
    data = load_csv(COMPONENT_SCORES_CSV)
    if not data:
        print("   Run 'python -m scripts.analyze_my_results' first.")
        return False
    
    components = [row['Component'] for row in data]
    scores = [float(row['Score']) for row in data]
    
    # Close the radar chart
    components.append(components[0])
    scores.append(scores[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=components,
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=dict(
            text="My System Profile",
            font=dict(size=24),
            x=0.5
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        template="plotly_white",
        font=dict(size=14),
        width=700,
        height=600
    )
    
    output_path = CHARTS_DIR / "system_radar.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("     CRYPTO INTELLIGENCE - CHART GENERATOR")
    print("=" * 60 + "\n")
    
    if not PLOTLY_AVAILABLE:
        print("[ERROR] Plotly is required. Install with:")
        print("   pip install plotly kaleido")
        sys.exit(1)
    
    # Create charts directory
    ensure_charts_dir()
    print(f"Charts will be saved to: {CHARTS_DIR}\n")
    
    # Generate all charts
    results = []
    results.append(("Component Scores", create_component_scores_chart()))
    results.append(("Signal Distribution", create_signal_distribution_chart()))
    results.append(("Confidence by Symbol", create_confidence_by_symbol_chart()))
    results.append(("Confidence Timeline", create_confidence_timeline_chart()))
    results.append(("System Radar", create_system_radar_chart()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("     CHART GENERATION SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\n  Generated {successful}/{len(results)} charts at 300 DPI")
    print(f"  Location: {CHARTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
