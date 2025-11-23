"""
Performance metrics for backtesting
"""
import numpy as np
from typing import List, Dict, Any


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    excess_returns = np.array(returns) - (risk_free_rate / periods_per_year)
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility)
    
    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    excess_returns = np.array(returns) - (risk_free_rate / periods_per_year)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_deviation = np.std(downside_returns)
    
    return (np.mean(excess_returns) / downside_deviation) * np.sqrt(periods_per_year)


def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Cumulative returns over time
        
    Returns:
        Maximum drawdown (negative value)
    """
    if not cumulative_returns:
        return 0.0
    
    cumulative = np.array(cumulative_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    
    return np.min(drawdown)


def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float = 1.0) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown)
    
    Args:
        total_return: Total return over period
        max_drawdown: Maximum drawdown (positive value)
        years: Number of years
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    return annualized_return / abs(max_drawdown)


def calculate_metrics_summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics summary
    
    Args:
        trades: List of trade dictionaries with 'pnl_pct' key
        
    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_trade_duration': 0.0
        }
    
    returns = [trade['pnl_pct'] for trade in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    # Basic metrics
    win_rate = len(wins) / len(returns)
    
    # Profit factor
    gross_profit = sum(abs(r) for r in wins) if wins else 0
    gross_loss = sum(abs(r) for r in losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=52)  # Weekly
    sortino = calculate_sortino_ratio(returns, periods_per_year=52)
    
    # Drawdown
    cumulative_returns = np.cumsum(returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    
    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 2),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'max_drawdown': round(max_dd, 4),
        'avg_win': round(np.mean(wins), 4) if wins else 0.0,
        'avg_loss': round(np.mean(losses), 4) if losses else 0.0,
        'largest_win': round(max(wins), 4) if wins else 0.0,
        'largest_loss': round(min(losses), 4) if losses else 0.0
    }
