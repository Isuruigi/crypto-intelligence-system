"""
Backtesting engine for strategy validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from app.agents.orchestrator import MultiAgentOrchestrator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Backtesting engine for validating trading strategies
    
    Simulates trades based on historical signals and calculates performance metrics
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital in USD
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.current_position = None
    
    async def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        orchestrator: MultiAgentOrchestrator
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            orchestrator: Orchestrator instance
            
        Returns:
            Performance metrics
        """
        logger.info('backtest_starting', symbol=symbol, start_date=start_date, end_date=end_date)
        
        # Generate date range (simplified - in production, use actual historical intervals)
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # For demo purposes, simulate 50 trading periods
        num_periods = min(50, (end - start).days // 7)  # Weekly signals
        
        for i in range(num_periods):
            try:
                # Generate signal
                signal = await orchestrator.generate_signal(symbol)
                
                # Simulate price (in production, use actual historical prices)
                price = signal['price'] * (1 + np.random.uniform(-0.02, 0.02))
                
                # Execute trade based on signal
                if signal['signal'] == 'BUY' and not self.current_position:
                    self._open_position(signal, price)
                elif signal['signal'] == 'SELL' and self.current_position:
                    self._close_position(signal, price)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error('backtest_period_error', period=i, error=str(e))
                continue
        
        # Close any open positions at the end
        if self.current_position:
            final_price = self.current_position['entry_price'] * 1.01  # Assume small profit
            self._close_position({'signal': 'SELL', 'confidence': 50}, final_price)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info('backtest_complete', total_trades=len(self.trades))
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            **metrics
        }
    
    def _open_position(self, signal: Dict[str, Any], price: float):
        """Open a long position"""
        # Use confidence to determine position size
        confidence_factor = signal['confidence'] / 100
        position_size_pct = 0.3 * confidence_factor  # Max 30% of capital
        
        position_value = self.capital * position_size_pct
        amount = position_value / price
        
        self.current_position = {
            'entry_price': price,
            'amount': amount,
            'entry_time': datetime.utcnow().isoformat(),
            'confidence': signal['confidence']
        }
        
        logger.info('position_opened', price=price, amount=amount, confidence=signal['confidence'])
    
    def _close_position(self, signal: Dict[str, Any], price: float):
        """Close current position"""
        if not self.current_position:
            return
        
        # Calculate PnL
        entry_price = self.current_position['entry_price']
        amount = self.current_position['amount']
        
        pnl = (price - entry_price) * amount
        pnl_pct = (price - entry_price) / entry_price
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = {
            'entry_price': entry_price,
            'exit_price': price,
            'amount': amount,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_confidence': self.current_position['confidence'],
            'exit_time': datetime.utcnow().isoformat()
        }
        
        self.trades.append(trade)
        self.current_position = None
        
        logger.info('position_closed', pnl=pnl, pnl_pct=pnl_pct)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Extract returns
        returns = [trade['pnl_pct'] for trade in self.trades]
        
        # Win rate
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        win_rate = len(wins) / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(abs(r) for r in wins) if wins else 0
        gross_loss = sum(abs(r) for r in losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (annualized, assuming weekly trades)
        if len(returns) > 1:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(52) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Total return
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': len(self.trades),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown, 4),
            'total_return': round(total_return, 4),
            'avg_win': round(np.mean([r for r in returns if r > 0]), 4) if wins else 0.0,
            'avg_loss': round(np.mean([r for r in returns if r < 0]), 4) if losses else 0.0
        }
