"""
Comprehensive risk management system for the trading platform.
Implements Kelly Criterion, correlation analysis, and dynamic position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sqlalchemy import and_

from src.core.database import db_manager
from src.core.models import Position, PortfolioSnapshot, MarketData, TradingSignal
from src.ml.ensemble import ModelPrediction
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeRecommendation:
    """Container for position sizing recommendation."""
    symbol: str
    recommended_allocation: float  # Percentage of portfolio
    kelly_allocation: float
    risk_adjusted_allocation: float
    max_position_size: float
    reasoning: str
    risk_factors: Dict[str, float]


@dataclass
class RiskMetrics:
    """Container for portfolio risk metrics."""
    total_exposure: float
    leverage: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float
    volatility: float


class KellyCriterionCalculator:
    """Implements Kelly Criterion for optimal position sizing."""
    
    def __init__(self):
        self.min_samples = 20  # Minimum number of historical trades for calculation
    
    def calculate_kelly_fraction(
        self,
        symbol: str,
        prediction: ModelPrediction,
        lookback_days: int = 90
    ) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            symbol: Trading pair symbol
            prediction: ML model prediction
            lookback_days: Historical period to analyze
            
        Returns:
            Kelly fraction (0.0 to 1.0)
        """
        try:
            # Get historical performance for this symbol
            win_rate, avg_win, avg_loss = self._get_historical_performance(symbol, lookback_days)
            
            if win_rate is None or avg_win is None or avg_loss is None:
                # Fallback to prediction-based calculation
                return self._prediction_based_kelly(prediction)
            
            # Kelly formula: f = (bp - q) / b
            # where:
            # b = odds received on the bet (avg_win / avg_loss)
            # p = probability of winning (win_rate)
            # q = probability of losing (1 - win_rate)
            
            if avg_loss == 0:
                return 0.0
            
            b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b if b != 0 else 0.0
            
            # Cap Kelly fraction at reasonable limits
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Max 25%
            
            # Adjust based on model confidence
            confidence_adjustment = prediction.confidence_score
            kelly_fraction *= confidence_adjustment
            
            logger.debug(f"Kelly fraction for {symbol}: {kelly_fraction:.3f} "
                        f"(win_rate: {win_rate:.3f}, avg_win: {avg_win:.3f}, avg_loss: {avg_loss:.3f})")
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction for {symbol}: {e}")
            return 0.05  # Conservative fallback
    
    def _get_historical_performance(
        self,
        symbol: str,
        lookback_days: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get historical win rate and average win/loss for a symbol."""
        try:
            with db_manager.get_session() as session:
                # Get closed positions for the symbol
                start_date = datetime.utcnow() - timedelta(days=lookback_days)
                
                positions = session.query(Position).filter(
                    and_(
                        Position.symbol == symbol,
                        Position.is_open == False,
                        Position.closed_at >= start_date
                    )
                ).all()
                
                if len(positions) < self.min_samples:
                    return None, None, None
                
                # Calculate performance metrics
                returns = [float(p.total_pnl) / float(p.average_entry_price * p.quantity) 
                          for p in positions if p.total_pnl is not None]
                
                if not returns:
                    return None, None, None
                
                winning_trades = [r for r in returns if r > 0]
                losing_trades = [r for r in returns if r < 0]
                
                win_rate = len(winning_trades) / len(returns)
                avg_win = np.mean(winning_trades) if winning_trades else 0.0
                avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.01
                
                return win_rate, avg_win, avg_loss
                
        except Exception as e:
            logger.error(f"Error getting historical performance for {symbol}: {e}")
            return None, None, None
    
    def _prediction_based_kelly(self, prediction: ModelPrediction) -> float:
        """Calculate Kelly fraction based on model prediction when no historical data."""
        try:
            # Estimate win probability from model confidence and expected return
            expected_return = prediction.expected_return
            confidence = prediction.confidence_score
            
            # Simple heuristic: higher expected return and confidence = higher probability
            win_probability = 0.5 + (expected_return * confidence * 10)  # Scale factor
            win_probability = max(0.1, min(win_probability, 0.9))  # Clamp to reasonable range
            
            # Assume 1:1 risk/reward ratio for simplicity
            odds = 1.0
            
            # Kelly calculation
            kelly_fraction = (odds * win_probability - (1 - win_probability)) / odds
            kelly_fraction = max(0.0, min(kelly_fraction, 0.15))  # Conservative cap
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error in prediction-based Kelly calculation: {e}")
            return 0.05


class CorrelationAnalyzer:
    """Analyzes correlations between positions to manage portfolio risk."""
    
    def __init__(self):
        self.correlation_window = 30  # Days for correlation calculation
    
    def get_portfolio_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for current portfolio positions."""
        try:
            # Get current positions
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.is_open == True
                ).all()
                
                if len(positions) < 2:
                    return pd.DataFrame()
                
                symbols = [p.symbol for p in positions]
                
                # Get price data for correlation calculation
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=self.correlation_window)
                
                price_data = {}
                for symbol in symbols:
                    prices = session.query(MarketData.timestamp, MarketData.close_price).filter(
                        and_(
                            MarketData.symbol == symbol,
                            MarketData.timestamp >= start_date,
                            MarketData.timestamp <= end_date
                        )
                    ).order_by(MarketData.timestamp).all()
                    
                    if prices:
                        price_data[symbol] = pd.Series(
                            [float(p.close_price) for p in prices],
                            index=[p.timestamp for p in prices]
                        )
                
                if len(price_data) < 2:
                    return pd.DataFrame()
                
                # Create price DataFrame
                price_df = pd.DataFrame(price_data)
                price_df = price_df.fillna(method='ffill').dropna()
                
                # Calculate returns
                returns_df = price_df.pct_change().dropna()
                
                # Calculate correlation matrix
                correlation_matrix = returns_df.corr()
                
                return correlation_matrix
                
        except Exception as e:
            logger.error(f"Error calculating portfolio correlations: {e}")
            return pd.DataFrame()
    
    def get_correlation_risk_score(self) -> float:
        """Calculate overall correlation risk for the portfolio."""
        try:
            correlation_matrix = self.get_portfolio_correlations()
            
            if correlation_matrix.empty:
                return 0.0
            
            # Calculate average correlation (excluding diagonal)
            correlations = correlation_matrix.values
            n = len(correlations)
            
            if n < 2:
                return 0.0
            
            # Sum all correlations except diagonal, then divide by number of pairs
            total_correlation = np.sum(correlations) - np.trace(correlations)  # Remove diagonal
            num_pairs = n * (n - 1)  # Total pairs excluding self-correlation
            
            avg_correlation = total_correlation / num_pairs if num_pairs > 0 else 0.0
            
            # Risk score: higher correlation = higher risk
            risk_score = abs(avg_correlation)
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk score: {e}")
            return 0.0
    
    def check_correlation_limit(self, new_symbol: str, existing_positions: List[str]) -> bool:
        """Check if adding a new position would violate correlation limits."""
        try:
            if not existing_positions:
                return True
            
            # Get correlation with existing positions
            max_correlation = 0.0
            
            for existing_symbol in existing_positions:
                correlation = self._get_pairwise_correlation(new_symbol, existing_symbol)
                max_correlation = max(max_correlation, abs(correlation))
            
            # Check against threshold
            return max_correlation <= settings.risk.max_correlation_threshold
            
        except Exception as e:
            logger.error(f"Error checking correlation limit for {new_symbol}: {e}")
            return True  # Allow trade if check fails
    
    def _get_pairwise_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols."""
        try:
            with db_manager.get_session() as session:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=self.correlation_window)
                
                # Get price data for both symbols
                prices1 = session.query(MarketData.timestamp, MarketData.close_price).filter(
                    and_(
                        MarketData.symbol == symbol1,
                        MarketData.timestamp >= start_date,
                        MarketData.timestamp <= end_date
                    )
                ).order_by(MarketData.timestamp).all()
                
                prices2 = session.query(MarketData.timestamp, MarketData.close_price).filter(
                    and_(
                        MarketData.symbol == symbol2,
                        MarketData.timestamp >= start_date,
                        MarketData.timestamp <= end_date
                    )
                ).order_by(MarketData.timestamp).all()
                
                if not prices1 or not prices2:
                    return 0.0
                
                # Convert to pandas series
                series1 = pd.Series(
                    [float(p.close_price) for p in prices1],
                    index=[p.timestamp for p in prices1]
                )
                
                series2 = pd.Series(
                    [float(p.close_price) for p in prices2],
                    index=[p.timestamp for p in prices2]
                )
                
                # Align timestamps and calculate returns
                aligned_df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
                returns_df = aligned_df.pct_change().dropna()
                
                if len(returns_df) < 10:  # Need minimum data points
                    return 0.0
                
                # Calculate correlation
                correlation = returns_df['s1'].corr(returns_df['s2'])
                
                return correlation if not np.isnan(correlation) else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def update_correlation_matrix(self):
        """
        Async method to update correlation matrix.
        This is a wrapper around the synchronous correlation calculation.
        """
        try:
            # Update correlation data for the portfolio
            correlation_matrix = self.get_portfolio_correlations()
            
            if not correlation_matrix.empty:
                # Store updated correlation data if needed
                # This could be expanded to cache the matrix or store historical correlation data
                logger.info(f"Updated correlation matrix for {len(correlation_matrix)} assets")
            else:
                logger.info("No correlation data to update (insufficient positions)")
                
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")


class VolatilityManager:
    """Manages position sizing based on volatility conditions."""
    
    def __init__(self):
        self.volatility_window = 20  # Days for volatility calculation
    
    def get_volatility_adjustment(self, symbol: str) -> float:
        """
        Calculate volatility-based adjustment factor for position sizing.
        
        Returns:
            Adjustment factor (0.5 to 1.5) where:
            - < 1.0 means reduce position size (high volatility)
            - > 1.0 means increase position size (low volatility)
        """
        try:
            with db_manager.get_session() as session:
                # Get recent price data
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=self.volatility_window * 2)  # Extra buffer
                
                prices = session.query(MarketData.close_price).filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.timestamp >= start_date,
                        MarketData.timestamp <= end_date
                    )
                ).order_by(MarketData.timestamp).all()
                
                if len(prices) < self.volatility_window:
                    return 1.0  # No adjustment if insufficient data
                
                # Calculate returns and volatility
                price_series = pd.Series([float(p.close_price) for p in prices])
                returns = price_series.pct_change().dropna()
                
                current_volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
                long_term_volatility = returns.std()
                
                # Volatility ratio
                volatility_ratio = current_volatility / long_term_volatility if long_term_volatility > 0 else 1.0
                
                # Adjustment factor (inverse relationship with volatility)
                # High volatility = reduce position size
                # Low volatility = increase position size
                adjustment = settings.risk.volatility_scaling_factor / volatility_ratio
                
                # Clamp to reasonable range
                adjustment = max(0.5, min(adjustment, 1.5))
                
                logger.debug(f"Volatility adjustment for {symbol}: {adjustment:.3f} "
                           f"(vol_ratio: {volatility_ratio:.3f})")
                
                return adjustment
                
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment for {symbol}: {e}")
            return 1.0


class RiskManager:
    """
    Main risk management system combining all risk components.
    """
    
    def __init__(self):
        self.kelly_calculator = KellyCriterionCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.volatility_manager = VolatilityManager()
    
    def calculate_position_size(
        self,
        symbol: str,
        prediction: ModelPrediction,
        portfolio_value: float
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size based on multiple risk factors.
        
        Args:
            symbol: Trading pair symbol
            prediction: ML model prediction
            portfolio_value: Current portfolio value
            
        Returns:
            Position sizing recommendation
        """
        try:
            # 1. Kelly Criterion base allocation
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(symbol, prediction)
            
            # 2. Volatility adjustment
            volatility_adjustment = self.volatility_manager.get_volatility_adjustment(symbol)
            
            # 3. Correlation check
            current_positions = self._get_current_position_symbols()
            correlation_allowed = self.correlation_analyzer.check_correlation_limit(
                symbol, current_positions
            )
            
            # 4. Portfolio concentration limits
            current_allocation = self._get_current_allocation(symbol)
            max_single_position = settings.max_portfolio_allocation
            
            # Calculate risk-adjusted allocation
            risk_adjusted_allocation = kelly_fraction * volatility_adjustment
            
            # Apply portfolio limits
            if current_allocation + risk_adjusted_allocation > max_single_position:
                risk_adjusted_allocation = max(0, max_single_position - current_allocation)
            
            # Apply correlation constraints
            if not correlation_allowed:
                risk_adjusted_allocation *= 0.5  # Reduce by 50% if high correlation
            
            # Apply minimum/maximum bounds
            risk_adjusted_allocation = max(
                settings.min_portfolio_allocation,
                min(risk_adjusted_allocation, settings.max_portfolio_allocation)
            )
            
            # Calculate maximum position size in dollars
            max_position_size = portfolio_value * risk_adjusted_allocation
            
            # Risk factors summary
            risk_factors = {
                'kelly_fraction': kelly_fraction,
                'volatility_adjustment': volatility_adjustment,
                'correlation_risk': self.correlation_analyzer.get_correlation_risk_score(),
                'current_allocation': current_allocation,
                'correlation_allowed': correlation_allowed
            }
            
            # Generate reasoning
            reasoning = self._generate_sizing_reasoning(risk_factors, prediction)
            
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_allocation=risk_adjusted_allocation,
                kelly_allocation=kelly_fraction,
                risk_adjusted_allocation=risk_adjusted_allocation,
                max_position_size=max_position_size,
                reasoning=reasoning,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            
            # Conservative fallback
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_allocation=settings.trading.min_portfolio_allocation,
                kelly_allocation=0.0,
                risk_adjusted_allocation=settings.trading.min_portfolio_allocation,
                max_position_size=portfolio_value * settings.trading.min_portfolio_allocation,
                reasoning="Error in calculation, using conservative fallback",
                risk_factors={}
            )
    
    async def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        portfolio_value: float = None
    ) -> PositionSizeRecommendation:
        """
        Async wrapper for position size calculation.
        
        Args:
            symbol: Trading pair symbol
            signal_strength: Signal confidence (0-1)
            portfolio_value: Current portfolio value (optional)
        """
        try:
            # Get portfolio value if not provided
            if portfolio_value is None:
                # This would need to be implemented to get actual portfolio value
                portfolio_value = 100000.0  # Default value for now
            
            # Create a mock prediction object for compatibility
            from src.ml.ensemble import ModelPrediction
            mock_prediction = ModelPrediction(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                signal_type='BUY' if signal_strength > 0 else 'SELL',
                confidence_score=abs(signal_strength),
                expected_return=signal_strength * 0.1,  # Simple scaling
                risk_score=1.0 - abs(signal_strength),
                model_scores={'ensemble': signal_strength}
            )
            
            # Use the synchronous method
            return self.calculate_position_size(symbol, mock_prediction, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error in async position size calculation: {e}")
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_allocation=0.01,  # 1% fallback
                kelly_allocation=0.01,
                risk_adjusted_allocation=0.01,
                max_position_size=portfolio_value * 0.01 if portfolio_value else 1000.0,
                reasoning="Error in calculation, using conservative fallback",
                risk_factors={}
            )
    
    def _get_current_position_symbols(self) -> List[str]:
        """Get symbols of current open positions."""
        try:
            with db_manager.get_session() as session:
                positions = session.query(Position.symbol).filter(
                    Position.is_open == True
                ).all()
                
                return [p.symbol for p in positions]
                
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []
    
    def _get_current_allocation(self, symbol: str) -> float:
        """Get current allocation percentage for a symbol."""
        try:
            with db_manager.get_session() as session:
                # Get current position
                position = session.query(Position).filter(
                    and_(
                        Position.symbol == symbol,
                        Position.is_open == True
                    )
                ).first()
                
                if not position:
                    return 0.0
                
                # Get latest portfolio snapshot
                portfolio_snapshot = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.desc()
                ).first()
                
                if not portfolio_snapshot:
                    return 0.0
                
                # Calculate position value
                position_value = float(position.quantity * position.current_price)
                portfolio_value = float(portfolio_snapshot.total_value)
                
                return position_value / portfolio_value if portfolio_value > 0 else 0.0
                
        except Exception as e:
            logger.error(f"Error getting current allocation for {symbol}: {e}")
            return 0.0
    
    def _generate_sizing_reasoning(
        self,
        risk_factors: Dict[str, float],
        prediction: ModelPrediction
    ) -> str:
        """Generate human-readable reasoning for position sizing."""
        try:
            reasoning_parts = []
            
            # Kelly Criterion
            kelly = risk_factors.get('kelly_fraction', 0)
            if kelly > 0.1:
                reasoning_parts.append(f"Strong Kelly signal ({kelly:.1%})")
            elif kelly > 0.05:
                reasoning_parts.append(f"Moderate Kelly signal ({kelly:.1%})")
            else:
                reasoning_parts.append(f"Weak Kelly signal ({kelly:.1%})")
            
            # Volatility
            vol_adj = risk_factors.get('volatility_adjustment', 1.0)
            if vol_adj < 0.8:
                reasoning_parts.append("reduced for high volatility")
            elif vol_adj > 1.2:
                reasoning_parts.append("increased for low volatility")
            
            # Correlation
            if not risk_factors.get('correlation_allowed', True):
                reasoning_parts.append("reduced due to high correlation with existing positions")
            
            # Model confidence
            if prediction.confidence_score > 0.8:
                reasoning_parts.append("high model confidence")
            elif prediction.confidence_score < 0.5:
                reasoning_parts.append("low model confidence")
            
            return "; ".join(reasoning_parts).capitalize()
            
        except Exception as e:
            logger.error(f"Error generating sizing reasoning: {e}")
            return "Standard risk-based sizing"
    
    def get_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        try:
            with db_manager.get_session() as session:
                # Get latest portfolio snapshot
                latest_snapshot = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.desc()
                ).first()
                
                if not latest_snapshot:
                    return self._empty_risk_metrics()
                
                # Get recent snapshots for historical analysis
                recent_snapshots = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.desc()
                ).limit(30).all()
                
                if len(recent_snapshots) < 5:
                    return self._snapshot_to_risk_metrics(latest_snapshot)
                
                # Calculate additional metrics from historical data
                returns = []
                values = []
                
                for i in range(1, len(recent_snapshots)):
                    current = recent_snapshots[i-1]
                    previous = recent_snapshots[i]
                    
                    if previous.total_value > 0:
                        daily_return = (
                            float(current.total_value) / float(previous.total_value) - 1
                        )
                        returns.append(daily_return)
                        values.append(float(current.total_value))
                
                # Calculate VaR (Value at Risk) 95%
                var_95 = np.percentile(returns, 5) if returns else 0.0  # 5th percentile
                
                # Calculate Sharpe ratio
                if returns:
                    avg_return = np.mean(returns)
                    volatility = np.std(returns)
                    sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
                
                # Calculate max drawdown
                if values:
                    peak = values[0]
                    max_drawdown = 0.0
                    current_drawdown = 0.0
                    
                    for value in values:
                        if value > peak:
                            peak = value
                        
                        drawdown = (peak - value) / peak if peak > 0 else 0.0
                        max_drawdown = max(max_drawdown, drawdown)
                        current_drawdown = drawdown  # Latest drawdown
                else:
                    max_drawdown = 0.0
                    current_drawdown = 0.0
                
                # Get correlation risk
                correlation_risk = self.correlation_analyzer.get_correlation_risk_score()
                
                # Calculate concentration risk
                concentration_risk = self._calculate_concentration_risk()
                
                return RiskMetrics(
                    total_exposure=float(latest_snapshot.total_exposure or 0),
                    leverage=float(latest_snapshot.leverage or 1.0),
                    var_95=var_95,
                    max_drawdown=max_drawdown,
                    current_drawdown=current_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    correlation_risk=correlation_risk,
                    concentration_risk=concentration_risk,
                    volatility=np.std(returns) if returns else 0.0
                )
                
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return self._empty_risk_metrics()
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics."""
        return RiskMetrics(
            total_exposure=0.0,
            leverage=1.0,
            var_95=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            volatility=0.0
        )
    
    def _snapshot_to_risk_metrics(self, snapshot: PortfolioSnapshot) -> RiskMetrics:
        """Convert portfolio snapshot to risk metrics."""
        return RiskMetrics(
            total_exposure=float(snapshot.total_exposure or 0),
            leverage=float(snapshot.leverage or 1.0),
            var_95=float(snapshot.var_95 or 0),
            max_drawdown=float(snapshot.max_drawdown or 0),
            current_drawdown=float(snapshot.current_drawdown or 0),
            sharpe_ratio=float(snapshot.sharpe_ratio or 0),
            correlation_risk=self.correlation_analyzer.get_correlation_risk_score(),
            concentration_risk=self._calculate_concentration_risk(),
            volatility=0.0  # Would need historical data
        )
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk."""
        try:
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.is_open == True
                ).all()
                
                if not positions:
                    return 0.0
                
                # Calculate position weights
                total_value = sum(
                    float(p.quantity * p.current_price) for p in positions
                    if p.current_price is not None
                )
                
                if total_value == 0:
                    return 0.0
                
                weights = [
                    float(p.quantity * p.current_price) / total_value
                    for p in positions if p.current_price is not None
                ]
                
                # Herfindahl-Hirschman Index for concentration
                hhi = sum(w**2 for w in weights)
                
                # Normalize to 0-1 scale (1 = maximum concentration)
                max_hhi = 1.0  # All money in one position
                concentration_risk = hhi / max_hhi
                
                return concentration_risk
                
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def check_risk_limits(self, new_position_size: float, portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if a new position would violate risk limits.
        
        Returns:
            (allowed, reason)
        """
        try:
            # Get current risk metrics
            risk_metrics = self.get_portfolio_risk_metrics()
            
            # Check maximum drawdown
            if risk_metrics.current_drawdown > settings.trading.max_drawdown_threshold:
                return False, f"Portfolio drawdown ({risk_metrics.current_drawdown:.1%}) exceeds limit"
            
            # Check total exposure
            new_exposure = risk_metrics.total_exposure + new_position_size
            max_exposure = portfolio_value * 1.0  # 100% exposure limit
            
            if new_exposure > max_exposure:
                return False, f"Position would exceed maximum exposure limit"
            
            # Check correlation risk
            if risk_metrics.correlation_risk > settings.risk.max_correlation_threshold:
                return False, f"Portfolio correlation risk too high ({risk_metrics.correlation_risk:.2f})"
            
            # Check concentration risk
            if risk_metrics.concentration_risk > 0.5:  # Max 50% concentration
                return False, f"Portfolio too concentrated ({risk_metrics.concentration_risk:.1%})"
            
            return True, "All risk checks passed"
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, "Error in risk limit check"

    async def update_correlation_matrix(self):
        """Update the correlation matrix for current portfolio."""
        try:
            await self.correlation_analyzer.update_correlation_matrix()
            logger.info("Correlation matrix updated successfully")
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    async def check_correlation_risk(self, symbol: str, allocation_percentage: float) -> Dict[str, any]:
        """
        Check if adding a position would violate correlation risk limits.
        
        Args:
            symbol: Symbol to check
            allocation_percentage: Proposed allocation as percentage of portfolio
            
        Returns:
            Dict with 'allowed' boolean and 'reason' string
        """
        try:
            # Get current position symbols
            current_positions = self._get_current_position_symbols()
            
            # Check correlation limit using the correlation analyzer
            correlation_allowed = self.correlation_analyzer.check_correlation_limit(
                symbol, current_positions
            )
            
            if not correlation_allowed:
                return {
                    'allowed': False,
                    'reason': f"High correlation risk with existing positions"
                }
            
            # Check if allocation would exceed concentration limits
            current_allocation = self._get_current_allocation(symbol)
            total_allocation = current_allocation + allocation_percentage
            
            if total_allocation > settings.max_portfolio_allocation:
                return {
                    'allowed': False,
                    'reason': f"Total allocation would exceed limit: {total_allocation:.2%} > {settings.max_portfolio_allocation:.2%}"
                }
            
            return {
                'allowed': True,
                'reason': 'Correlation and concentration checks passed'
            }
            
        except Exception as e:
            logger.error(f"Error checking correlation risk for {symbol}: {e}")
            return {
                'allowed': False,
                'reason': f"Risk check error: {str(e)}"
            }


# Global risk manager instance
risk_manager = RiskManager()
