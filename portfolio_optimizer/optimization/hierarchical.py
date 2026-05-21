"""Hierarchical portfolio optimization models: Hierarchical Risk Parity (HRP) and Hierarchical Equal Risk Contribution (HERC)."""

import logging
import pandas as pd
import numpy as np
import riskfolio as rp
from typing import Optional
from portfolio_optimizer.optimization.base_optimizer import BaseOptimizer
from portfolio_optimizer.models.optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

class HRPOptimizer(BaseOptimizer):
    """Hierarchical Risk Parity (HRP) Optimizer."""
    
    def __init__(self, risk_free_rate: float = 0.045, linkage: str = "single"):
        super().__init__("Hierarchical Risk Parity", risk_free_rate)
        self.linkage = linkage  # "single", "complete", "average", "ward"

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        try:
            # Setup HCPortfolio
            port = rp.HCPortfolio(returns=returns)
            
            # Estimate optimal portfolio
            # rm='MV' (Standard deviation)
            w = port.optimization(
                model='HRP',
                codependence='pearson',
                rm='MV',
                rf=self.risk_free_rate / 252,
                linkage=self.linkage,
                leaf_order=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            w_arr = w.values
            
            # Compute performance stats
            mean_ret = np.mean(returns.values @ w_arr) * 252
            vol = np.std(returns.values @ w_arr, ddof=1) * np.sqrt(252)
            sharpe = (mean_ret - self.risk_free_rate) / vol if vol > 0 else 0.0
            
            portfolio_returns = returns @ w_arr.flatten()
            cvar = float(-np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=mean_ret,
                expected_volatility=vol,
                sharpe_ratio=sharpe,
                cvar=cvar,
                optimizer_name=self.name,
                additional_metrics={"linkage": self.linkage}
            )
        except Exception as e:
            logger.error(f"HRP Optimization failed: {e}. Falling back to equal weighting.")
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


class HERCOptimizer(BaseOptimizer):
    """Hierarchical Equal Risk Contribution (HERC) Optimizer."""
    
    def __init__(self, risk_free_rate: float = 0.045, linkage: str = "ward"):
        super().__init__("Hierarchical Equal Risk Contribution", risk_free_rate)
        self.linkage = linkage

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        try:
            port = rp.HCPortfolio(returns=returns)
            
            w = port.optimization(
                model='HERC',
                codependence='pearson',
                rm='MV',
                rf=self.risk_free_rate / 252,
                linkage=self.linkage,
                leaf_order=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            w_arr = w.values
            
            mean_ret = np.mean(returns.values @ w_arr) * 252
            vol = np.std(returns.values @ w_arr, ddof=1) * np.sqrt(252)
            sharpe = (mean_ret - self.risk_free_rate) / vol if vol > 0 else 0.0
            
            portfolio_returns = returns @ w_arr.flatten()
            cvar = float(-np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=mean_ret,
                expected_volatility=vol,
                sharpe_ratio=sharpe,
                cvar=cvar,
                optimizer_name=self.name,
                additional_metrics={"linkage": self.linkage}
            )
        except Exception as e:
            logger.error(f"HERC Optimization failed: {e}. Falling back to equal weighting.")
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
