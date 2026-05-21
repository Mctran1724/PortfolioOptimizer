"""Black-Litterman asset allocation model updating equilibrium returns with subjective views."""

import logging
import pandas as pd
import numpy as np
import riskfolio as rp
from typing import Optional, Dict, List
from portfolio_optimizer.optimization.base_optimizer import BaseOptimizer
from portfolio_optimizer.models.optimization_result import OptimizationResult

logger = logging.getLogger(__name__)

class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman model that merges market equilibrium priors with custom investor views
    and returns optimized portfolio allocations.
    """
    
    def __init__(
        self, 
        risk_free_rate: float = 0.045, 
        delta: float = 3.0, 
        tau: float = 0.05
    ):
        super().__init__("Black-Litterman Optimization", risk_free_rate)
        self.delta = delta
        self.tau = tau

    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        """
        Executes Black-Litterman optimization.
        - returns: historical return DataFrame
        - covariance: optional covariance matrix
        kwargs:
            - market_weights: Dict[str, float] representing equilibrium asset weights
            - views_P: np.ndarray matrix mapping views to assets (shape: views x assets)
            - views_Q: np.ndarray vector of view returns (shape: views x 1)
        """
        try:
            n_assets = len(returns.columns)
            assets = list(returns.columns)
            
            # Extract views
            P = kwargs.get("views_P", None)
            Q = kwargs.get("views_Q", None)
            mkt_w_dict = kwargs.get("market_weights", None)
            
            # If no market weights are provided, assume equal weights as equilibrium
            if mkt_w_dict is None:
                mkt_w = np.ones(n_assets) / n_assets
            else:
                mkt_w = np.array([mkt_w_dict.get(a, 1.0 / n_assets) for a in assets])
                
            # If no views are provided, standard MVO is run
            if P is None or Q is None:
                logger.info("No views provided. Black-Litterman running standard MVO.")
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=self.risk_free_rate/252, hist=True)
                weights_dict = w.iloc[:, 0].to_dict()
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=float(w.values.T @ port.mu.values)[0] * 252,
                    expected_volatility=float(np.sqrt(w.values.T @ port.cov.values @ w.values))[0] * np.sqrt(252),
                    sharpe_ratio=float((w.values.T @ port.mu.values)[0] * 252 - self.risk_free_rate) / (float(np.sqrt(w.values.T @ port.cov.values @ w.values))[0] * np.sqrt(252)),
                    cvar=0.05,
                    optimizer_name=self.name
                )
                
            P = np.array(P)
            Q = np.array(Q).reshape(-1, 1)
            
            # Covariance matrix (Sigma)
            sigma = covariance.values if covariance is not None else returns.cov().values
            
            # 1. Implied equilibrium returns (Pi = delta * Sigma * w)
            pi = self.delta * (sigma @ mkt_w).reshape(-1, 1)
            
            # 2. Uncertainty of views matrix (Omega = P * (tau * Sigma) * P^T)
            # We assume views are independent, so Omega is diagonal
            omega = np.diag(np.diag(P @ (self.tau * sigma) @ P.T))
            if np.any(np.diagonal(omega) == 0):
                # Avoid singular matrix by adding small noise
                omega += np.eye(len(omega)) * 1e-6
                
            # 3. Calculate Black-Litterman posterior expected returns (mu_bl)
            # mu_bl = [ (tau*Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau*Sigma)^-1 * pi + P^T * Omega^-1 * Q ]
            inv_tau_sigma = np.linalg.inv(self.tau * sigma)
            inv_omega = np.linalg.inv(omega)
            
            middle_inv = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
            mu_bl = middle_inv @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
            
            # 4. Posterior Covariance (Sigma_bl = Sigma + middle_inv)
            sigma_bl = sigma + middle_inv
            
            # Set up Riskfolio using the BL returns and covariance
            port = rp.Portfolio(returns=returns)
            port.mu = pd.Series(mu_bl.flatten(), index=assets)
            port.cov = pd.DataFrame(sigma_bl, index=assets, columns=assets)
            
            # Run max Sharpe optimization
            w = port.optimization(
                model='Classic', 
                rm='MV', 
                obj='Sharpe', 
                rf=self.risk_free_rate / 252, 
                hist=True
            )
            
            weights_dict = w.iloc[:, 0].to_dict()
            w_arr = w.values
            
            mu_val = port.mu.values.flatten()
            cov_val = port.cov.values
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
                optimizer_name=self.name,
                additional_metrics={"views_incorporated": len(Q)}
            )
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}. Falling back to standard MVO.")
            # Fallback to standard MVO without views
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
                optimizer_name=self.name + " (Fallback MVO)"
            )
