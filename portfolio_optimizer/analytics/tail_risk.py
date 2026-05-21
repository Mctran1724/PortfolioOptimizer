"""Tail risk diagnostics (Value at Risk, Expected Shortfall / CVaR, Max Drawdown, Calmar Ratio, Ulcer Index)."""

import pandas as pd
import numpy as np
from typing import Union

def calculate_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95
) -> float:
    """
    Calculates Value at Risk (VaR) using the historical simulation method.
    VaR is represented as a positive percentage loss (e.g. 0.05 means a 5% loss boundary).
    """
    if len(returns) == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    # Quantile of the return distribution
    var_val = np.percentile(returns, alpha * 100)
    # Return as positive number representing the loss
    return float(-var_val)

def calculate_cvar(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95
) -> float:
    """
    Calculates Conditional Value at Risk (CVaR) / Expected Shortfall.
    CVaR is the average return in the worst (1 - confidence_level) percentage of cases.
    Returned as a positive percentage loss.
    """
    if len(returns) == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    var_cutoff = -calculate_var(returns, confidence_level)
    
    # Worst case returns (returns below the negative VaR cutoff)
    worst_returns = returns[returns <= var_cutoff]
    
    if len(worst_returns) == 0:
        return float(-var_cutoff)
        
    return float(-np.mean(worst_returns))

def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculates the Maximum Drawdown of a returns series.
    Returns drawdown as a positive percentage (e.g., 0.20 for a 20% decline).
    """
    if len(returns) == 0:
        return 0.0
    
    # Reconstruct cumulative wealth index
    cumulative = (1.0 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1.0 + returns)
    
    # Calculate rolling maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdowns
    drawdowns = (cumulative - running_max) / running_max
    
    return float(-np.min(drawdowns))

def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculates the Calmar Ratio.
    Calmar Ratio = Annualized Return / Maximum Drawdown
    """
    max_dd = calculate_max_drawdown(returns)
    if max_dd == 0.0:
        return 0.0
        
    annualized_return = np.mean(returns) * periods_per_year
    return float(annualized_return / max_dd)

def calculate_ulcer_index(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculates the Ulcer Index (UI), penalizing both the depth and duration of drawdowns.
    UI = sqrt( Mean( Drawdown_t ^ 2 ) )
    """
    if len(returns) == 0:
        return 0.0
        
    cumulative = (1.0 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    
    # Quadratic mean of drawdowns
    squared_drawdowns = drawdowns ** 2
    mean_squared = np.mean(squared_drawdowns)
    return float(np.sqrt(mean_squared))
