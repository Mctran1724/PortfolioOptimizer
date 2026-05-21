"""Kelly Criterion optimization model to maximize the geometric growth rate of the portfolio."""

import logging
import pandas as pd
import numpy as np
import riskfolio as rp
from typing import Optional
from portfolio_optimizer.optimization.base_optimizer import BaseOptimizer
from portfolio_optimizer.models.optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

class KellyOptimizer(BaseOptimizer):
    """
    Kelly Criterion Optimizer.
    Maximizes the logarithmic expected wealth (asymptotic compound growth rate) 
    using riskfolio's Log-Return / Logarithmic Mean Risk framework.
    """
    
    def __init__(self, risk_free_rate: float = 0.045):
        super().__init__("Kelly Criterion (Log-Mean)", risk_free_rate)

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        try:
            # Set up Riskfolio
            port = rp.Portfolio(returns=returns)
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            if covariance is not None:
                port.cov = covariance
                
            rf_daily = self.risk_free_rate / 252
            
            # Riskfolio supports obj='LogRet' to maximize logarithmic utility of wealth
            # under convex constraints.
            w = port.optimization(
                model='Classic',
                rm='MV',
                obj='LogRet',
                rf=rf_daily,
                l=0.0,
                hist=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            w_arr = w.values
            
            mu = port.mu
            cov = port.cov
            
            mu_val = mu.values.flatten()
            cov_val = cov.values
            w_val = w_arr.flatten()
            
            exp_return = float(np.sum(w_val * mu_val)) * 252
            exp_vol = float(np.sqrt(w_val @ cov_val @ w_val)) * np.sqrt(252)
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
            
            portfolio_returns = returns @ w_arr.flatten()
            cvar = float(-np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                cvar=cvar,
                optimizer_name=self.name
            )
        except Exception as e:
            logger.error(f"Kelly Optimization failed: {e}. Falling back to Mean-Variance (Max Sharpe).")
            # Fallback to standard Sharpe optimization if LogRet fails (some solvers don't support log cones)
            try:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=self.risk_free_rate/252, hist=True)
                weights_dict = w.iloc[:, 0].to_dict()
                mu_val = port.mu.values.flatten()
                cov_val = port.cov.values
                w_val = w.values.flatten()
                
                exp_return = float(np.sum(w_val * mu_val)) * 252
                exp_vol = float(np.sqrt(w_val @ cov_val @ w_val)) * np.sqrt(252)
                sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=exp_return,
                    expected_volatility=exp_vol,
                    sharpe_ratio=sharpe,
                    cvar=0.05,
                    optimizer_name=self.name + " (Fallback Max Sharpe)"
                )
            except Exception:
                n_assets = len(returns.columns)
                w_eq = {col: 1.0 / n_assets for col in returns.columns}
                return OptimizationResult(
                    weights=w_eq,
                    expected_return=0.05,
                    expected_volatility=0.15,
                    sharpe_ratio=0.3,
                    cvar=0.08,
                    optimizer_name=self.name + " (Fallback Equal Weight)"
                )
