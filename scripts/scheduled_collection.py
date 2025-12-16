"""
Scheduled Signal Collection for Crypto Intelligence System
Collects signals over an extended time period (12 hours) for authentic poster data

Usage:
    python -m scripts.scheduled_collection              # Full 12-hour collection
    python -m scripts.scheduled_collection --hours 6    # Custom duration
    python -m scripts.scheduled_collection --quick      # Quick test (1 hour)
"""
import asyncio
import csv
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.coordinator import AgentCoordinator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Output paths
DATA_DIR = project_root / "data"
RESULTS_DIR = DATA_DIR / "real_results"
OUTPUT_CSV = RESULTS_DIR / "scheduled_signals.csv"


def ensure_dirs():
    """Create output directories"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_csv_fieldnames():
    """CSV column headers"""
    return [
        'timestamp', 'collection_hour', 'symbol', 'signal_type', 'confidence',
        'social_score', 'news_score', 'chain_score', 'whale_score',
        'combined_score', 'risk_level', 'position_size', 'price',
        'fear_greed', 'execution_time', 'success'
    ]


def parse_result(result: dict, symbol: str, collection_hour: int) -> dict:
    """Parse coordinator result into CSV row"""
    components = result.get('components', {})
    risk = result.get('risk', {})
    market = result.get('market_context', {})
    
    signal_type = result.get('signal', 'NEUTRAL')
    success = signal_type in ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
    
    return {
        'timestamp': datetime.now().isoformat(),
        'collection_hour': collection_hour,
        'symbol': symbol,
        'signal_type': signal_type,
        'confidence': result.get('confidence', 0),
        'social_score': components.get('social', 0),
        'news_score': components.get('news', 0),
        'chain_score': components.get('chain', 0),
        'whale_score': components.get('whale', 0),
        'combined_score': result.get('score', 0),
        'risk_level': risk.get('level', 'medium'),
        'position_size': risk.get('position_size', 0.02),
        'price': result.get('price', 0),
        'fear_greed': market.get('fear_greed', 50),
        'execution_time': result.get('execution_time_ms', 0) / 1000,  # Convert to seconds
        'success': success
    }


def append_to_csv(row: dict):
    """Append a single row to CSV file"""
    file_exists = OUTPUT_CSV.exists()
    
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=get_csv_fieldnames())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_existing_count() -> int:
    """Count existing signals in CSV"""
    if not OUTPUT_CSV.exists():
        return 0
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract header


async def run_collection_round(coordinator, symbols: list, collection_hour: int, round_num: int, total_rounds: int):
    """Run one collection round for all symbols"""
    print(f"\n{'='*60}")
    print(f"  ROUND {round_num}/{total_rounds} - Hour {collection_hour}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    results_this_round = 0
    
    for symbol in symbols:
        try:
            print(f"  Analyzing {symbol}...", end=" ", flush=True)
            
            start = time.time()
            result = await coordinator.run_analysis(symbol)
            elapsed = time.time() - start
            
            row = parse_result(result, symbol, collection_hour)
            append_to_csv(row)
            
            signal = row['signal_type']
            conf = row['confidence']
            print(f"[OK] {signal} ({conf:.1f}%) - {elapsed:.1f}s")
            
            results_this_round += 1
            
            # Small delay between symbols
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"[ERROR] {e}")
            logger.error(f"collection_error: {e}")
            # Record failed attempt
            fail_row = {k: '' for k in get_csv_fieldnames()}
            fail_row.update({
                'timestamp': datetime.now().isoformat(),
                'collection_hour': collection_hour,
                'symbol': symbol,
                'signal_type': 'ERROR',
                'success': False
            })
            append_to_csv(fail_row)
    
    total = get_existing_count()
    print(f"\n  Round complete: {results_this_round}/{len(symbols)} signals")
    print(f"  Total collected: {total} signals")
    
    return results_this_round


async def main_collection(hours: float, interval_minutes: int = 30):
    """
    Main collection loop
    
    Args:
        hours: Total collection duration in hours
        interval_minutes: Time between collection rounds
    """
    symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
    
    total_rounds = int((hours * 60) / interval_minutes)
    signals_per_round = len(symbols)
    expected_total = total_rounds * signals_per_round
    
    print("\n" + "=" * 60)
    print("     SCHEDULED SIGNAL COLLECTION")
    print("=" * 60)
    print(f"\n  Duration:      {hours} hours")
    print(f"  Interval:      Every {interval_minutes} minutes")
    print(f"  Symbols:       {', '.join(symbols)}")
    print(f"  Total rounds:  {total_rounds}")
    print(f"  Expected:      {expected_total} signals")
    print(f"  Output:        {OUTPUT_CSV}")
    print(f"\n  Start time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time:      {(datetime.now() + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check for existing data
    existing = get_existing_count()
    if existing > 0:
        print(f"\n  [INFO] Found {existing} existing signals. Will append to file.")
    
    # Initialize coordinator
    print("\n  Initializing Multi-Agent System...")
    coordinator = AgentCoordinator()
    print("  [OK] System ready!\n")
    
    start_time = datetime.now()
    
    for round_num in range(1, total_rounds + 1):
        round_start = datetime.now()
        hours_elapsed = (round_start - start_time).total_seconds() / 3600
        
        await run_collection_round(
            coordinator, 
            symbols, 
            collection_hour=int(hours_elapsed),
            round_num=round_num,
            total_rounds=total_rounds
        )
        
        # Wait for next round (unless it's the last one)
        if round_num < total_rounds:
            next_round_time = round_start + timedelta(minutes=interval_minutes)
            wait_seconds = (next_round_time - datetime.now()).total_seconds()
            
            if wait_seconds > 0:
                print(f"\n  Waiting {wait_seconds/60:.1f} min until next round...")
                print(f"  Next round at: {next_round_time.strftime('%H:%M:%S')}")
                print("  (Press Ctrl+C to stop early and generate reports)")
                
                try:
                    await asyncio.sleep(wait_seconds)
                except KeyboardInterrupt:
                    print("\n\n  [INTERRUPTED] Generating reports with collected data...")
                    break
    
    # Final summary
    total_collected = get_existing_count()
    duration = (datetime.now() - start_time).total_seconds() / 3600
    
    print("\n" + "=" * 60)
    print("     COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"\n  Total signals:    {total_collected}")
    print(f"  Duration:         {duration:.2f} hours")
    print(f"  Success rate:     {count_successful():.1f}%")
    print(f"  Output file:      {OUTPUT_CSV}")
    print("=" * 60)
    
    # Auto-generate analysis and charts
    print("\n  Generating analysis and charts...")
    generate_reports()


def count_successful() -> float:
    """Calculate success rate"""
    if not OUTPUT_CSV.exists():
        return 0
    
    total = 0
    success = 0
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('success', 'False').lower() == 'true':
                success += 1
    
    return (success / total * 100) if total > 0 else 0


def generate_reports():
    """Generate analysis and charts from collected data"""
    # Copy to expected location for other scripts
    import shutil
    my_signals = RESULTS_DIR / "my_signals.csv"
    if OUTPUT_CSV.exists():
        shutil.copy(OUTPUT_CSV, my_signals)
        print(f"  [OK] Copied data to {my_signals}")
    
    # Run analysis script
    try:
        import subprocess
        print("  Running analyze_my_results.py...")
        subprocess.run([sys.executable, "-m", "scripts.analyze_my_results"], 
                      cwd=str(project_root), check=True)
        
        print("  Running generate_my_charts.py...")
        subprocess.run([sys.executable, "-m", "scripts.generate_my_charts"],
                      cwd=str(project_root), check=True)
        
        print("  Running generate_extended_charts.py...")
        subprocess.run([sys.executable, "-m", "scripts.generate_extended_charts"],
                      cwd=str(project_root), check=True)
        
        print("\n  [OK] All reports generated!")
        print(f"  Charts at: {project_root / 'data' / 'my_charts'}")
        
    except Exception as e:
        print(f"  [WARNING] Could not auto-generate reports: {e}")
        print("  Run manually:")
        print("    python -m scripts.analyze_my_results")
        print("    python -m scripts.generate_my_charts")


def main():
    parser = argparse.ArgumentParser(description="Scheduled signal collection")
    parser.add_argument('--hours', type=float, default=12, 
                       help='Collection duration in hours (default: 12)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Minutes between collection rounds (default: 30)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (1 hour, 15 min interval)')
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    if args.quick:
        hours = 1
        interval = 15
    else:
        hours = args.hours
        interval = args.interval
    
    try:
        asyncio.run(main_collection(hours, interval))
    except KeyboardInterrupt:
        print("\n\n  [STOPPED] Collection interrupted.")
        total = get_existing_count()
        if total > 0:
            print(f"  {total} signals were collected.")
            print("  To generate reports from existing data, run:")
            print("    python -m scripts.scheduled_collection --generate-only")


if __name__ == "__main__":
    main()
