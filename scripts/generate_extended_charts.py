"""
Extended Chart Generator for Crypto Intelligence System Poster
Generates additional poster-ready charts at 300 DPI

Usage:
    python -m scripts.generate_extended_charts
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List
import math

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


def load_csv(filepath: Path) -> List[Dict]:
    """Load data from CSV file"""
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def create_system_architecture_chart():
    """Create system architecture flow diagram"""
    print("Creating system architecture chart...")
    
    # Create a Sankey diagram showing data flow
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["Reddit", "Twitter", "News APIs", "CoinGecko", "Binance", 
                     "Social Agent", "News Agent", "Chain Agent", "Whale Agent",
                     "Signal Generator", "Risk Manager", "Final Signal"],
            color = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c",
                    "#2ecc71", "#3498db", "#9b59b6", "#e67e22",
                    "#27ae60", "#e74c3c", "#2c3e50"]
        ),
        link = dict(
            source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # indices
            target = [5, 5, 6, 7, 8, 9, 9, 9, 9, 10, 11],
            value = [30, 30, 40, 50, 30, 25, 25, 25, 25, 100, 100],
            color = ["rgba(231,76,60,0.3)", "rgba(52,152,219,0.3)", 
                    "rgba(155,89,182,0.3)", "rgba(243,156,18,0.3)", 
                    "rgba(26,188,156,0.3)", "rgba(46,204,113,0.3)",
                    "rgba(52,152,219,0.3)", "rgba(155,89,182,0.3)",
                    "rgba(230,126,34,0.3)", "rgba(39,174,96,0.3)",
                    "rgba(44,62,80,0.3)"]
        )
    )])
    
    fig.update_layout(
        title=dict(text="Multi-Agent System Architecture", font=dict(size=24), x=0.5),
        font=dict(size=12),
        width=1000,
        height=600
    )
    
    output_path = CHARTS_DIR / "system_architecture.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_correlation_heatmap():
    """Create correlation heatmap between components"""
    print("Creating correlation heatmap...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        return False
    
    # Extract scores
    social = [float(d.get('social_score', 0)) for d in data]
    news = [float(d.get('news_score', 0)) for d in data]
    chain = [float(d.get('chain_score', 0)) for d in data]
    whale = [float(d.get('whale_score', 0)) for d in data]
    confidence = [float(d.get('confidence', 0)) for d in data]
    
    # Calculate correlations manually
    def correlation(x, y):
        n = len(x)
        mean_x, mean_y = sum(x)/n, sum(y)/n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = (sum((xi - mean_x)**2 for xi in x) * sum((yi - mean_y)**2 for yi in y)) ** 0.5
        return num / den if den != 0 else 0
    
    components = [social, news, chain, whale, confidence]
    labels = ['Social', 'News', 'Chain', 'Whale', 'Confidence']
    
    # Build correlation matrix
    corr_matrix = []
    for i, ci in enumerate(components):
        row = []
        for j, cj in enumerate(components):
            row.append(round(correlation(ci, cj), 2))
        corr_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=dict(text="Component Correlation Matrix", font=dict(size=24), x=0.5),
        width=700,
        height=600
    )
    
    output_path = CHARTS_DIR / "correlation_heatmap.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_signal_confidence_distribution():
    """Create histogram of confidence scores"""
    print("Creating confidence distribution chart...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        return False
    
    confidences = [float(d.get('confidence', 0)) for d in data]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color='#3498db',
        opacity=0.8
    )])
    
    # Add mean line
    mean_conf = sum(confidences) / len(confidences)
    fig.add_vline(x=mean_conf, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_conf:.1f}%")
    
    fig.update_layout(
        title=dict(text="Confidence Score Distribution", font=dict(size=24), x=0.5),
        xaxis_title="Confidence (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        width=800,
        height=500
    )
    
    output_path = CHARTS_DIR / "confidence_distribution.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_multi_component_comparison():
    """Create grouped bar chart comparing components across symbols"""
    print("Creating multi-component comparison chart...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        return False
    
    # Group by symbol
    symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
    by_symbol = {s: {'social': [], 'news': [], 'chain': [], 'whale': []} for s in symbols}
    
    for d in data:
        sym = d.get('symbol', 'BTC')
        by_symbol[sym]['social'].append(float(d.get('social_score', 0)))
        by_symbol[sym]['news'].append(float(d.get('news_score', 0)))
        by_symbol[sym]['chain'].append(float(d.get('chain_score', 0)))
        by_symbol[sym]['whale'].append(float(d.get('whale_score', 0)))
    
    # Calculate averages and convert to 0-100
    def avg_to_pct(lst):
        return (sum(lst)/len(lst) + 1) / 2 * 100 if lst else 50
    
    fig = go.Figure()
    
    components = ['social', 'news', 'chain', 'whale']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
    names = ['Social', 'News', 'Chain', 'Whale']
    
    for comp, color, name in zip(components, colors, names):
        fig.add_trace(go.Bar(
            name=name,
            x=symbols,
            y=[avg_to_pct(by_symbol[s][comp]) for s in symbols],
            marker_color=color
        ))
    
    fig.update_layout(
        title=dict(text="Component Scores by Symbol", font=dict(size=24), x=0.5),
        xaxis_title="Symbol",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        barmode='group',
        template="plotly_white",
        width=900,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    output_path = CHARTS_DIR / "component_by_symbol.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_risk_distribution():
    """Create pie chart of risk levels"""
    print("Creating risk distribution chart...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        return False
    
    # Count risk levels
    risk_counts = {}
    for d in data:
        risk = d.get('risk_level', 'medium').upper()
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    if not risk_counts:
        risk_counts = {'LOW': 20, 'MEDIUM': 60, 'HIGH': 20}
    
    colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c', 'VERY_HIGH': '#c0392b'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        marker_colors=[colors.get(k, '#95a5a6') for k in risk_counts.keys()],
        textinfo='label+percent',
        textposition='outside',
        hole=0.4
    )])
    
    fig.update_layout(
        title=dict(text="Risk Level Distribution", font=dict(size=24), x=0.5),
        width=700,
        height=500,
        annotations=[dict(text='Risk', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    output_path = CHARTS_DIR / "risk_distribution.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_execution_time_chart():
    """Create execution time analysis chart"""
    print("Creating execution time chart...")
    
    data = load_csv(SIGNALS_CSV)
    if not data:
        return False
    
    times = [float(d.get('execution_time', 10)) for d in data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=times,
        name="Execution Time",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='#3498db'
    ))
    
    avg_time = sum(times) / len(times)
    
    fig.update_layout(
        title=dict(text=f"Analysis Execution Time (Avg: {avg_time:.1f}s)", font=dict(size=24), x=0.5),
        yaxis_title="Time (seconds)",
        template="plotly_white",
        width=600,
        height=500
    )
    
    output_path = CHARTS_DIR / "execution_time.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def create_combined_dashboard():
    """Create a combined dashboard with multiple metrics"""
    print("Creating combined dashboard...")
    
    data = load_csv(SIGNALS_CSV)
    component_data = load_csv(COMPONENT_SCORES_CSV)
    
    if not data:
        return False
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Component Scores", "Signal Distribution", 
                       "Confidence by Symbol", "Risk Distribution"),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # 1. Component scores bar chart
    if component_data:
        comps = [r['Component'] for r in component_data]
        scores = [float(r['Score']) for r in component_data]
        fig.add_trace(go.Bar(x=comps, y=scores, marker_color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22']),
                     row=1, col=1)
    
    # 2. Signal distribution pie
    signal_counts = {}
    for d in data:
        sig = d.get('signal_type', 'NEUTRAL')
        signal_counts[sig] = signal_counts.get(sig, 0) + 1
    
    fig.add_trace(go.Pie(labels=list(signal_counts.keys()), values=list(signal_counts.values()),
                        marker_colors=['#2ecc71', '#95a5a6']),
                 row=1, col=2)
    
    # 3. Confidence by symbol
    symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
    sym_conf = {s: [] for s in symbols}
    for d in data:
        sym_conf[d.get('symbol', 'BTC')].append(float(d.get('confidence', 0)))
    
    avg_confs = [sum(sym_conf[s])/len(sym_conf[s]) if sym_conf[s] else 0 for s in symbols]
    fig.add_trace(go.Bar(x=symbols, y=avg_confs, marker_color='#3498db'),
                 row=2, col=1)
    
    # 4. Risk distribution
    risk_counts = {}
    for d in data:
        risk = d.get('risk_level', 'MEDIUM').upper()
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    if not risk_counts:
        risk_counts = {'MEDIUM': 100}
    
    fig.add_trace(go.Pie(labels=list(risk_counts.keys()), values=list(risk_counts.values()),
                        marker_colors=['#2ecc71', '#f39c12', '#e74c3c']),
                 row=2, col=2)
    
    fig.update_layout(
        title=dict(text="Crypto Intelligence System - Results Dashboard", font=dict(size=24), x=0.5),
        height=800,
        width=1200,
        showlegend=False
    )
    
    output_path = CHARTS_DIR / "combined_dashboard.png"
    fig.write_image(str(output_path), scale=3)
    print(f"   [OK] Saved: {output_path}")
    return True


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("     EXTENDED CHART GENERATOR")
    print("=" * 60 + "\n")
    
    if not PLOTLY_AVAILABLE:
        print("[ERROR] Plotly is required. Install with:")
        print("   pip install plotly kaleido")
        sys.exit(1)
    
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Charts will be saved to: {CHARTS_DIR}\n")
    
    results = []
    results.append(("System Architecture", create_system_architecture_chart()))
    results.append(("Correlation Heatmap", create_correlation_heatmap()))
    results.append(("Confidence Distribution", create_signal_confidence_distribution()))
    results.append(("Component by Symbol", create_multi_component_comparison()))
    results.append(("Risk Distribution", create_risk_distribution()))
    results.append(("Execution Time", create_execution_time_chart()))
    results.append(("Combined Dashboard", create_combined_dashboard()))
    
    print("\n" + "=" * 60)
    print("     EXTENDED CHARTS SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\n  Generated {successful}/{len(results)} additional charts")
    print(f"  Location: {CHARTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
