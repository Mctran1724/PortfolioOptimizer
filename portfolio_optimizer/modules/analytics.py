# Factor models, Sortino, CVaR
"""
Analytics module for portfolio analysis including factor models, 
Sortino ratio, CVaR, and other risk metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from typing import Dict, List, Tuple, Union

class PortfolioAnalytics:
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with returns DataFrame
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
    
    def calculate_sortino_ratio(self, returns: pd.Series = None, 
                              risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        """
        if returns is None:
            returns = self.returns
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        if downside_deviation == 0:
            return np.inf
        
        return np.mean(excess_returns) / downside_deviation
    
    def calculate_cvar(self, returns: pd.Series = None, 
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR)
        """
        if returns is None:
            returns = self.returns
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_var(self, returns: pd.Series = None, 
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        """
        if returns is None:
            returns = self.returns
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
    
    def calculate_max_drawdown(self, returns: pd.Series = None) -> float:
        """
        Calculate maximum drawdown
        """
        if returns is None:
            returns = self.returns
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def factor_model_regression(self, returns: pd.Series, 
                              factor_returns: pd.DataFrame) -> Dict:
        """
        Perform factor model regression (CAPM, Fama-French, etc.)
        """
        # Add constant for intercept
        X = sm.add_constant(factor_returns)
        y = returns
        
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'beta': model.params.drop('const').to_dict(),
            'r_squared': model.rsquared,
            'p_values': model.pvalues.to_dict(),
            'residuals': model.resid
        }
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio
        """
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return np.inf
        
        return active_returns.mean() / tracking_error
    
    def rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling metrics (volatility, Sharpe, etc.)
        """
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (self.returns.rolling(window).mean() * 252) / rolling_vol
        
        metrics = pd.DataFrame({
            'rolling_volatility': rolling_vol.mean(axis=1),
            'rolling_sharpe': rolling_sharpe.mean(axis=1)
        })
        
        return metrics

if __name__ == "__main__":
    # Example usage
    # returns = pd.DataFrame(np.random.randn(1000, 3), columns=['A', 'B', 'C'])
    # analytics = PortfolioAnalytics(returns)
    # print(f"Sortino Ratio: {analytics.calculate_sortino_ratio()}")
    # print(f"CVaR (95%): {analytics.calculate_cvar()}")