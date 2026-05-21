"""Traditional risk-adjusted performance metrics (Sharpe, Sortino, Treynor, Information Ratio)."""

import pandas as pd
import numpy as np
from typing import Union, Optional

def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray], 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 252
) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    Sharpe Ratio = (Mean(R_p) - R_f_daily) / Std(R_p) * sqrt(periods_per_year)
    """
    # Annualized rates to daily/period rates
    rf_daily = risk_free_rate / periods_per_year
    
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)
    
    if std_returns == 0.0:
        return 0.0
        
    daily_sharpe = mean_excess / std_returns
    return float(daily_sharpe * np.sqrt(periods_per_year))

def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculates the annualized Sortino Ratio.
    Sortino Ratio = (Mean(R_p) - R_f_daily) / DownsideDeviation * sqrt(periods_per_year)
    DownsideDeviation is the root mean square of negative deviations below MAR.
    """
    rf_daily = risk_free_rate / periods_per_year
    mar_daily = mar / periods_per_year
    
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Calculate downside deviation
    downside_diff = returns - mar_daily
    negative_diff = np.minimum(0, downside_diff)
    downside_variance = np.mean(negative_diff ** 2)
    downside_std = np.sqrt(downside_variance)
    
    if downside_std == 0.0:
        return 0.0
        
    daily_sortino = mean_excess / downside_std
    return float(daily_sortino * np.sqrt(periods_per_year))

def calculate_treynor_ratio(
    returns: Union[pd.Series, np.ndarray],
    beta: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculates the annualized Treynor Ratio.
    Treynor Ratio = (Annualized_Return - R_f) / Beta
    """
    if beta == 0.0:
        return 0.0
        
    mean_return = np.mean(returns) * periods_per_year
    return float((mean_return - risk_free_rate) / beta)

def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculates the Information Ratio.
    Information Ratio = Mean(returns - benchmark_returns) / Std(returns - benchmark_returns) * sqrt(periods_per_year)
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length.")
        
    active_returns = returns - benchmark_returns
    mean_active = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0.0:
        return 0.0
        
    daily_ir = mean_active / tracking_error
    return float(daily_ir * np.sqrt(periods_per_year))
