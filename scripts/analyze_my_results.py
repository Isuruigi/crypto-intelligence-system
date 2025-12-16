"""
Data Analyzer for Crypto Intelligence Signal Collection
Analyzes collected signals and generates summary metrics

Usage:
    python -m scripts.analyze_my_results
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# File paths
DATA_DIR = project_root / "data"
SIGNALS_CSV = DATA_DIR / "real_results" / "my_signals.csv"
COMPONENT_SCORES_CSV = DATA_DIR / "poster_component_scores.csv"
SIGNAL_DISTRIBUTION_CSV = DATA_DIR / "my_signal_distribution.csv"
PERFORMANCE_BY_SYMBOL_CSV = DATA_DIR / "performance_by_symbol.csv"
SUMMARY_STATS_CSV = DATA_DIR / "my_summary_stats.csv"


def load_signals() -> List[Dict[str, Any]]:
    """Load signals from CSV file"""
    if not SIGNALS_CSV.exists():
        print(f"[ERROR] {SIGNALS_CSV} not found!")
        print("   Run 'python -m scripts.collect_my_results' first.")
        sys.exit(1)
    
    signals = []
    with open(SIGNALS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['confidence'] = float(row.get('confidence', 0))
            row['social_score'] = float(row.get('social_score', 0))
            row['news_score'] = float(row.get('news_score', 0))
            row['chain_score'] = float(row.get('chain_score', 0))
            row['whale_score'] = float(row.get('whale_score', 0))
            row['combined_score'] = float(row.get('combined_score', 0))
            row['position_size'] = float(row.get('position_size', 0))
            row['success'] = row.get('success', 'True').lower() == 'true'
            signals.append(row)
    
    print(f"[OK] Loaded {len(signals)} signals from {SIGNALS_CSV}")
    return signals


def calculate_component_scores(signals: List[Dict]) -> Dict[str, float]:
    """Calculate average component scores (0-100 scale)"""
    if not signals:
        return {}
    
    # Filter successful signals
    successful = [s for s in signals if s['success']]
    n = len(successful)
    
    if n == 0:
        return {}
    
    # Average raw scores (-1 to 1 range)
    avg_social = sum(s['social_score'] for s in successful) / n
    avg_news = sum(s['news_score'] for s in successful) / n
    avg_chain = sum(s['chain_score'] for s in successful) / n
    avg_whale = sum(s['whale_score'] for s in successful) / n
    
    # Convert to 0-100 scale
    scores = {
        'Social': round((avg_social + 1) / 2 * 100, 2),
        'News': round((avg_news + 1) / 2 * 100, 2),
        'Chain': round((avg_chain + 1) / 2 * 100, 2),
        'Whale': round((avg_whale + 1) / 2 * 100, 2)
    }
    
    # Save to CSV
    with open(COMPONENT_SCORES_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Component', 'Score'])
        for comp, score in scores.items():
            writer.writerow([comp, score])
    
    print(f"[OK] Component scores saved to {COMPONENT_SCORES_CSV}")
    return scores


def calculate_signal_distribution(signals: List[Dict]) -> Dict[str, float]:
    """Calculate signal type distribution"""
    successful = [s for s in signals if s['success']]
    n = len(successful)
    
    if n == 0:
        return {}
    
    # Count signal types
    signal_counts = Counter(s['signal_type'] for s in successful)
    
    # Calculate percentages
    distribution = {}
    for signal_type, count in signal_counts.items():
        distribution[signal_type] = round(count / n * 100, 2)
    
    # Save to CSV
    with open(SIGNAL_DISTRIBUTION_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Signal_Type', 'Count', 'Percentage'])
        for signal_type, count in sorted(signal_counts.items()):
            pct = distribution[signal_type]
            writer.writerow([signal_type, count, f"{pct}%"])
    
    print(f"[OK] Signal distribution saved to {SIGNAL_DISTRIBUTION_CSV}")
    return distribution


def calculate_performance_by_symbol(signals: List[Dict]) -> List[Dict]:
    """Calculate performance metrics for each symbol"""
    successful = [s for s in signals if s['success']]
    
    # Group by symbol
    by_symbol = {}
    for s in successful:
        symbol = s['symbol']
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(s)
    
    # Calculate metrics
    performance = []
    for symbol, sym_signals in by_symbol.items():
        n = len(sym_signals)
        avg_conf = sum(s['confidence'] for s in sym_signals) / n
        signal_counts = Counter(s['signal_type'] for s in sym_signals)
        most_common = signal_counts.most_common(1)[0][0] if signal_counts else 'N/A'
        
        performance.append({
            'symbol': symbol,
            'count': n,
            'avg_confidence': round(avg_conf, 2),
            'most_common_signal': most_common
        })
    
    # Sort by confidence
    performance.sort(key=lambda x: x['avg_confidence'], reverse=True)
    
    # Save to CSV
    with open(PERFORMANCE_BY_SYMBOL_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['symbol', 'count', 'avg_confidence', 'most_common_signal'])
        writer.writeheader()
        writer.writerows(performance)
    
    print(f"[OK] Performance by symbol saved to {PERFORMANCE_BY_SYMBOL_CSV}")
    return performance


def calculate_summary_stats(signals: List[Dict]) -> Dict[str, Any]:
    """Calculate overall summary statistics"""
    successful = [s for s in signals if s['success']]
    n = len(successful)
    
    if n == 0:
        return {}
    
    # Get timestamps
    timestamps = [s['timestamp'] for s in successful]
    start_time = min(timestamps)
    end_time = max(timestamps)
    
    # Calculate confidence stats
    confidences = [s['confidence'] for s in successful]
    avg_conf = sum(confidences) / n
    max_conf = max(confidences)
    max_conf_signal = next(s for s in successful if s['confidence'] == max_conf)
    
    summary = {
        'Total Signals': len(signals),
        'Successful Signals': n,
        'Failed Signals': len(signals) - n,
        'Collection Start': start_time,
        'Collection End': end_time,
        'Average Confidence': round(avg_conf, 2),
        'Highest Confidence': round(max_conf, 2),
        'Highest Confidence Symbol': max_conf_signal['symbol'],
        'Highest Confidence Signal': max_conf_signal['signal_type']
    }
    
    # Save to CSV
    with open(SUMMARY_STATS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in summary.items():
            writer.writerow([metric, value])
    
    print(f"[OK] Summary stats saved to {SUMMARY_STATS_CSV}")
    return summary


def print_beautiful_summary(
    component_scores: Dict,
    signal_distribution: Dict,
    performance: List[Dict],
    summary: Dict
):
    """Print a beautiful summary table"""
    print("\n")
    print("=" * 60)
    print("         MY UNIQUE RESULTS - CRYPTO INTELLIGENCE")
    print("=" * 60)
    
    print("\n[COMPONENT SCORES] (0-100%)")
    print("-" * 40)
    for comp, score in component_scores.items():
        bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
        print(f"  {comp:12s} {bar} {score:.1f}%")
    
    print("\n[SIGNAL DISTRIBUTION]")
    print("-" * 40)
    for signal, pct in sorted(signal_distribution.items(), key=lambda x: -x[1]):
        bar_len = int(pct / 5)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {signal:15s} {bar} {pct:.1f}%")
    
    print("\n[PERFORMANCE BY SYMBOL]")
    print("-" * 40)
    print(f"  {'Symbol':<10} {'Count':<8} {'Avg Conf':<12} {'Top Signal'}")
    print("  " + "-" * 36)
    for p in performance:
        print(f"  {p['symbol']:<10} {p['count']:<8} {p['avg_confidence']:.1f}%{'':<6} {p['most_common_signal']}")
    
    print("\n[SUMMARY STATISTICS]")
    print("-" * 40)
    print(f"  Total Signals:        {summary.get('Total Signals', 'N/A')}")
    print(f"  Successful:           {summary.get('Successful Signals', 'N/A')}")
    print(f"  Collection Start:     {summary.get('Collection Start', 'N/A')[:19]}")
    print(f"  Collection End:       {summary.get('Collection End', 'N/A')[:19]}")
    print(f"  Average Confidence:   {summary.get('Average Confidence', 'N/A')}%")
    print(f"  Highest Confidence:   {summary.get('Highest Confidence', 'N/A')}%")
    print(f"  Best Signal:          {summary.get('Highest Confidence Symbol', '')} - {summary.get('Highest Confidence Signal', '')}")
    
    print("\n" + "=" * 60)
    print("  Files saved to:")
    print(f"    - {COMPONENT_SCORES_CSV}")
    print(f"    - {SIGNAL_DISTRIBUTION_CSV}")
    print(f"    - {PERFORMANCE_BY_SYMBOL_CSV}")
    print(f"    - {SUMMARY_STATS_CSV}")
    print("=" * 60 + "\n")


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("     CRYPTO INTELLIGENCE - DATA ANALYZER")
    print("=" * 60 + "\n")
    
    # Load signals
    signals = load_signals()
    
    print("\nAnalyzing data...\n")
    
    # Calculate all metrics
    component_scores = calculate_component_scores(signals)
    signal_distribution = calculate_signal_distribution(signals)
    performance = calculate_performance_by_symbol(signals)
    summary = calculate_summary_stats(signals)
    
    # Print beautiful summary
    print_beautiful_summary(component_scores, signal_distribution, performance, summary)


if __name__ == "__main__":
    main()
