"""
Tests for backtesting engine
"""
import pytest
from backtesting.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_metrics_summary
)


def test_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    # Positive returns
    returns = [0.01, 0.02, -0.01, 0.03, 0.02]
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=52)
    assert sharpe > 0
    
    # Zero returns
    returns = [0.0, 0.0, 0.0]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe == 0.0


def test_sortino_ratio():
    """Test Sortino ratio calculation"""
    returns = [0.02, -0.01, 0.03, -0.02, 0.04]
    sortino = calculate_sortino_ratio(returns, periods_per_year=52)
    assert sortino >= 0


def test_max_drawdown():
    """Test max drawdown calculation"""
    cumulative_returns = [0.01, 0.03, 0.02, 0.04, 0.03, 0.05]
    max_dd = calculate_max_drawdown(cumulative_returns)
    assert max_dd <= 0  # Should be negative or zero


def test_metrics_summary():
    """Test comprehensive metrics calculation"""
    trades = [
        {'pnl_pct': 0.05},
        {'pnl_pct': -0.02},
        {'pnl_pct': 0.03},
        {'pnl_pct': 0.04},
        {'pnl_pct': -0.01}
    ]
    
    metrics = calculate_metrics_summary(trades)
    
    # Validate structure
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'sharpe_ratio' in metrics
    
    # Validate values
    assert metrics['total_trades'] == 5
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['profit_factor'] >= 0


def test_empty_trades():
    """Test metrics with no trades"""
    metrics = calculate_metrics_summary([])
    
    assert metrics['total_trades'] == 0
    assert metrics['win_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
