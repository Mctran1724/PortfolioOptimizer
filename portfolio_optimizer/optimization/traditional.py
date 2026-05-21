"""Traditional optimization models: Mean-Variance Optimization and Vanilla Risk Parity."""

import logging
import pandas as pd
import numpy as np
import riskfolio as rp
from typing import Optional, Dict
from portfolio_optimizer.optimization.base_optimizer import BaseOptimizer
from portfolio_optimizer.models.optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

class MeanVarianceOptimizer(BaseOptimizer):
    """Mean-Variance Optimization (MVO) to maximize Sharpe Ratio or minimize volatility."""
    
    def __init__(self, risk_free_rate: float = 0.045, objective: str = "Sharpe"):
        super().__init__("Mean-Variance Optimization", risk_free_rate)
        self.objective = objective  # "Sharpe", "MinRisk", or "MaxRet"

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        try:
            # Setup Riskfolio Portfolio
            port = rp.Portfolio(returns=returns)
            
            # Estimate inputs
            port.assets_stats(method_mu='hist', method_cov='hist')
            if covariance is not None:
                # Override covariance matrix (e.g. from Marcenko-Pastur filtering)
                port.cov = covariance
                
            # Apply risk-free rate
            rf_daily = self.risk_free_rate / 252
            
            # Optimization
            # rm='MV' stands for Mean-Variance (standard deviation)
            w = port.optimization(
                model='Classic', 
                rm='MV', 
                obj=self.objective, 
                rf=rf_daily, 
                l=0.0, 
                hist=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            
            # Get return & risk stats
            mu = port.mu
            cov = port.cov
            w_arr = w.values
            
            mu_val = mu.values.flatten()
            cov_val = cov.values
            w_val = w_arr.flatten()
            
            exp_return = float(np.sum(w_val * mu_val)) * 252
            exp_vol = float(np.sqrt(w_val @ cov_val @ w_val)) * np.sqrt(252)
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
            
            # Calculate historical CVaR for diagnostics
            portfolio_returns = returns @ w_arr.flatten()
            cvar = float(-np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                expected_volatility=exp_vol,
                sharpe_ratio=sharpe,
                cvar=cvar,
                optimizer_name=self.name,
                additional_metrics={"objective": self.objective}
            )
        except Exception as e:
            logger.error(f"MVO Optimization failed: {e}. Falling back to equal weighting.")
            # Equal weight fallback
            n_assets = len(returns.columns)
            w_eq = {col: 1.0 / n_assets for col in returns.columns}
            return self._equal_weight_fallback(returns, w_eq)

    def _equal_weight_fallback(self, returns: pd.DataFrame, weights: Dict[str, float]) -> OptimizationResult:
        w_arr = np.array([weights[c] for c in returns.columns])
        port_returns = returns.values @ w_arr
        mean_ret = np.mean(port_returns) * 252
        vol = np.std(port_returns, ddof=1) * np.sqrt(252)
        cvar = float(-np.mean(port_returns[port_returns <= np.percentile(port_returns, 5)]))
        sharpe = (mean_ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        
        return OptimizationResult(
            weights=weights,
            expected_return=mean_ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            cvar=cvar,
            optimizer_name=self.name + " (Equal Weight Fallback)"
        )


class RiskParityOptimizer(BaseOptimizer):
    """Vanilla Risk Parity / Risk Budgeting to equalize asset volatility risk contributions."""
    
    def __init__(self, risk_free_rate: float = 0.045):
        super().__init__("Vanilla Risk Parity", risk_free_rate)

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        try:
            port = rp.Portfolio(returns=returns)
            port.assets_stats(method_mu='hist', method_cov='hist')
            if covariance is not None:
                port.cov = covariance
                
            rf_daily = self.risk_free_rate / 252
            
            # rp_optimization calculates equal risk contribution
            w = port.rp_optimization(
                model='Classic', 
                rm='MV', 
                rf=rf_daily, 
                hist=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            
            # Compute performance stats
            mu = port.mu
            cov = port.cov
            w_arr = w.values
            
            mu_val = mu.values.flatten()
            cov_val = cov.values
            w_val = w_arr.flatten()
            
            exp_return = float(np.sum(w_val * mu_val)) * 252
            exp_vol = float(np.sqrt(w_val @ cov_val @ w_val)) * np.sqrt(252)
            sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
            
            # Calculate historical CVaR for diagnostics
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
            logger.error(f"Risk Parity Optimization failed: {e}. Falling back to equal weighting.")
            # Equal weight fallback
            n_assets = len(returns.columns)
            w_eq = {col: 1.0 / n_assets for col in returns.columns}
            
            w_arr = np.array([1.0 / n_assets] * n_assets)
            port_returns = returns.values @ w_arr
            mean_ret = np.mean(port_returns) * 252
            vol = np.std(port_returns, ddof=1) * np.sqrt(252)
            cvar = float(-np.mean(port_returns[port_returns <= np.percentile(port_returns, 5)]))
            sharpe = (mean_ret - self.risk_free_rate) / vol if vol > 0 else 0.0
            
            return OptimizationResult(
                weights=w_eq,
                expected_return=mean_ret,
                expected_volatility=vol,
                sharpe_ratio=sharpe,
                cvar=cvar,
                optimizer_name=self.name + " (Equal Weight Fallback)"
            )
