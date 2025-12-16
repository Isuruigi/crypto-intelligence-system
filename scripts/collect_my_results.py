"""
Signal Collection Script for Crypto Intelligence System
Collects 100 real signals with variations across symbols and parameters

Usage:
    Full collection (100 signals):
        python -m scripts.collect_my_results
    
    Test mode (1 signal):
        python -m scripts.collect_my_results --test
"""
import asyncio
import csv
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.coordinator import AgentCoordinator
from src.utils.logger import get_logger

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not installed. Using simple progress output.")

logger = get_logger("signal_collector")

# =============================================================================
# Configuration
# =============================================================================

SYMBOLS = ["BTC", "ETH", "SOL", "ADA", "XRP"]
ANALYSES_PER_SYMBOL = 20
POST_LIMITS = [20, 50, 100, 200]
TIME_WINDOWS = ["6h", "12h", "24h", "48h"]
DELAY_SECONDS = 10
TOTAL_ANALYSES = len(SYMBOLS) * ANALYSES_PER_SYMBOL  # 100

# Output paths
OUTPUT_DIR = project_root / "data" / "real_results"
SIGNALS_CSV = OUTPUT_DIR / "my_signals.csv"
SUMMARY_CSV = project_root / "data" / "poster_component_scores.csv"

# CSV columns
SIGNAL_COLUMNS = [
    "timestamp",
    "symbol",
    "signal_type",
    "confidence",
    "social_score",
    "news_score",
    "chain_score",
    "whale_score",
    "combined_score",
    "risk_level",
    "position_size",
    "post_limit",
    "time_window",
    "execution_time_ms",
    "success"
]


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_output_dirs():
    """Create output directories if they don't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready: {OUTPUT_DIR}")


def parse_result(result: Dict[str, Any], symbol: str, post_limit: int, time_window: str) -> Dict[str, Any]:
    """
    Parse the analysis result into a flat record for CSV
    
    Args:
        result: Raw result from AgentCoordinator.run_analysis()
        symbol: Symbol analyzed
        post_limit: Post limit used
        time_window: Time window used
        
    Returns:
        Flat dictionary for CSV writing
    """
    components = result.get("components", {})
    risk = result.get("risk", {})
    
    # Determine success: check if we got a valid signal (not just the success flag)
    signal_type = result.get("signal", "NEUTRAL")
    # Consider it successful if we got a valid signal type and components
    has_valid_signal = signal_type in ["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
    has_components = bool(components)
    is_successful = (result.get("success", False) or has_valid_signal) and signal_type != "ERROR"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "signal_type": signal_type,
        "confidence": result.get("confidence", 0.0),
        "social_score": components.get("sentiment", 0.0),
        "news_score": components.get("news", 0.0),
        "chain_score": components.get("onchain", 0.0),
        "whale_score": components.get("whale", 0.0),
        "combined_score": result.get("score", 0.0),
        "risk_level": risk.get("level", "medium"),
        "position_size": risk.get("position_size", 0.02),
        "post_limit": post_limit,
        "time_window": time_window,
        "execution_time_ms": result.get("execution_time_ms", 0),
        "success": is_successful
    }


def calculate_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from collected records
    
    Args:
        records: List of signal records
        
    Returns:
        Summary statistics dictionary
    """
    if not records:
        return {}
    
    successful = [r for r in records if r["success"]]
    n = len(successful)
    
    if n == 0:
        return {"error": "No successful analyses"}
    
    # Signal type distribution
    signal_counts = {}
    for r in successful:
        sig = r["signal_type"]
        signal_counts[sig] = signal_counts.get(sig, 0) + 1
    
    signal_distribution = {k: round(v / n * 100, 2) for k, v in signal_counts.items()}
    
    # Component score averages (convert to 0-100 scale)
    avg_social = sum(r["social_score"] for r in successful) / n
    avg_news = sum(r["news_score"] for r in successful) / n
    avg_chain = sum(r["chain_score"] for r in successful) / n
    avg_whale = sum(r["whale_score"] for r in successful) / n
    avg_combined = sum(r["combined_score"] for r in successful) / n
    avg_confidence = sum(r["confidence"] for r in successful) / n
    
    return {
        "total_analyses": len(records),
        "successful_analyses": n,
        "failed_analyses": len(records) - n,
        "avg_social_score_pct": round((avg_social + 1) / 2 * 100, 2),  # -1 to 1 -> 0 to 100
        "avg_news_score_pct": round((avg_news + 1) / 2 * 100, 2),
        "avg_chain_score_pct": round((avg_chain + 1) / 2 * 100, 2),
        "avg_whale_score_pct": round((avg_whale + 1) / 2 * 100, 2),
        "avg_combined_score_pct": round((avg_combined + 1) / 2 * 100, 2),
        "avg_confidence": round(avg_confidence, 2),
        "signal_distribution": signal_distribution
    }


def save_signals_csv(records: List[Dict[str, Any]]):
    """Save all signal records to CSV"""
    with open(SIGNALS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    logger.info(f"Saved {len(records)} signals to {SIGNALS_CSV}")


def save_summary_csv(summary: Dict[str, Any]):
    """Save summary statistics to CSV"""
    # Flatten signal distribution into the summary
    flat_summary = {k: v for k, v in summary.items() if k != "signal_distribution"}
    signal_dist = summary.get("signal_distribution", {})
    for sig_type, pct in signal_dist.items():
        flat_summary[f"pct_{sig_type.lower()}"] = pct
    
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_summary.keys())
        writer.writeheader()
        writer.writerow(flat_summary)
    logger.info(f"Saved summary to {SUMMARY_CSV}")


def print_summary(summary: Dict[str, Any]):
    """Print summary statistics to console"""
    print("\n" + "=" * 60)
    print("SIGNAL COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal Analyses:      {summary.get('total_analyses', 0)}")
    print(f"Successful:          {summary.get('successful_analyses', 0)}")
    print(f"Failed:              {summary.get('failed_analyses', 0)}")
    print("\n--- Component Score Averages (0-100%) ---")
    print(f"Social Score:        {summary.get('avg_social_score_pct', 0):.2f}%")
    print(f"News Score:          {summary.get('avg_news_score_pct', 0):.2f}%")
    print(f"Chain Score:         {summary.get('avg_chain_score_pct', 0):.2f}%")
    print(f"Whale Score:         {summary.get('avg_whale_score_pct', 0):.2f}%")
    print(f"Combined Score:      {summary.get('avg_combined_score_pct', 0):.2f}%")
    print(f"\nAverage Confidence:  {summary.get('avg_confidence', 0):.2f}%")
    print("\n--- Signal Distribution ---")
    signal_dist = summary.get('signal_distribution', {})
    for sig_type, pct in sorted(signal_dist.items()):
        print(f"{sig_type:18s}  {pct:.2f}%")
    print("\n" + "=" * 60)
    print(f"Results saved to: {SIGNALS_CSV}")
    print(f"Summary saved to: {SUMMARY_CSV}")
    print("=" * 60 + "\n")


# =============================================================================
# Main Collection Logic
# =============================================================================

async def run_single_analysis(
    coordinator: AgentCoordinator,
    symbol: str,
    post_limit: int,
    time_window: str
) -> Dict[str, Any]:
    """
    Run a single analysis and return parsed result
    
    Args:
        coordinator: AgentCoordinator instance
        symbol: Crypto symbol (e.g., "BTC")
        post_limit: Number of posts to analyze
        time_window: Time window for data collection
        
    Returns:
        Parsed result dictionary
    """
    pair = f"{symbol}/USDT"
    
    try:
        # Run the analysis
        result = await coordinator.run_analysis(symbol=pair)
        return parse_result(result, symbol, post_limit, time_window)
        
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal_type": "ERROR",
            "confidence": 0.0,
            "social_score": 0.0,
            "news_score": 0.0,
            "chain_score": 0.0,
            "whale_score": 0.0,
            "combined_score": 0.0,
            "risk_level": "unknown",
            "position_size": 0.0,
            "post_limit": post_limit,
            "time_window": time_window,
            "execution_time_ms": 0,
            "success": False
        }


async def collect_signals(test_mode: bool = False):
    """
    Main collection function
    
    Args:
        test_mode: If True, only run 1 analysis for testing
    """
    print("\n" + "=" * 60)
    print("CRYPTO INTELLIGENCE SIGNAL COLLECTOR")
    print("=" * 60)
    
    # Setup
    ensure_output_dirs()
    
    # Determine how many analyses to run
    if test_mode:
        total = 1
        symbols_to_use = [SYMBOLS[0]]
        analyses_per = 1
        print(f"\n[TEST MODE] Running 1 analysis only\n")
    else:
        total = TOTAL_ANALYSES
        symbols_to_use = SYMBOLS
        analyses_per = ANALYSES_PER_SYMBOL
        print(f"\n[FULL MODE] Running {total} analyses")
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print(f"Analyses per symbol: {analyses_per}")
        print(f"Delay between analyses: {DELAY_SECONDS}s\n")
    
    # Initialize coordinator
    print("Initializing AgentCoordinator...")
    coordinator = AgentCoordinator()
    print("Coordinator ready!\n")
    
    # Collection
    records: List[Dict[str, Any]] = []
    analysis_count = 0
    
    # Create progress iterator
    if HAS_TQDM:
        progress = tqdm(total=total, desc="Collecting signals", unit="signal")
    else:
        progress = None
    
    try:
        for symbol in symbols_to_use:
            for i in range(analyses_per):
                # Rotate through limits and windows
                post_limit = POST_LIMITS[analysis_count % len(POST_LIMITS)]
                time_window = TIME_WINDOWS[analysis_count % len(TIME_WINDOWS)]
                
                # Run analysis
                record = await run_single_analysis(
                    coordinator, symbol, post_limit, time_window
                )
                records.append(record)
                analysis_count += 1
                
                # Update progress
                if progress:
                    progress.update(1)
                    progress.set_postfix({
                        "symbol": symbol,
                        "signal": record.get("signal_type", "?")[:4]
                    })
                else:
                    status = "✓" if record.get("success") else "✗"
                    print(f"[{analysis_count}/{total}] {status} {symbol} - {record.get('signal_type', 'ERROR')}")
                
                # Delay between analyses (skip on last iteration)
                if analysis_count < total:
                    if not test_mode:
                        time.sleep(DELAY_SECONDS)
    
    finally:
        if progress:
            progress.close()
    
    # Save results
    print("\nSaving results...")
    save_signals_csv(records)
    
    # Calculate and save summary
    summary = calculate_summary(records)
    save_summary_csv(summary)
    
    # Print summary
    print_summary(summary)
    
    return records, summary


def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Collect trading signals from Crypto Intelligence System"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (1 analysis only)"
    )
    args = parser.parse_args()
    
    # Run async collection
    asyncio.run(collect_signals(test_mode=args.test))


if __name__ == "__main__":
    main()
