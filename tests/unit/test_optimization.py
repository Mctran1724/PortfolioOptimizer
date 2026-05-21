"""Unit tests for optimization solvers."""

import numpy as np
import pandas as pd
from portfolio_optimizer.optimization.traditional import MeanVarianceOptimizer, RiskParityOptimizer
from portfolio_optimizer.optimization.hierarchical import HRPOptimizer, HERCOptimizer
from portfolio_optimizer.optimization.kelly_criterion import KellyOptimizer
from portfolio_optimizer.optimization.shortfall_minimization import ShortfallMinimizationOptimizer
from portfolio_optimizer.utils.helpers import generate_sample_returns

def test_optimizers():
    tickers = ["QQQ", "SPY", "TLT", "TIP"]
    returns = generate_sample_returns(tickers, periods=100)
    
    # 1. Mean-Variance Optimization
    mvo = MeanVarianceOptimizer()
    res_mvo = mvo.optimize(returns)
    assert res_mvo.optimizer_name == "Mean-Variance Optimization"
    assert len(res_mvo.weights) == len(tickers)
    
    # 2. Risk Parity
    rp_opt = RiskParityOptimizer()
    res_rp = rp_opt.optimize(returns)
    assert "Risk Parity" in res_rp.optimizer_name
    assert len(res_rp.weights) == len(tickers)
    
    # 3. HRP
    hrp = HRPOptimizer()
    res_hrp = hrp.optimize(returns)
    assert "Hierarchical" in res_hrp.optimizer_name
    assert len(res_hrp.weights) == len(tickers)
    
    # 4. HERC
    herc = HERCOptimizer()
    res_herc = herc.optimize(returns)
    assert "Hierarchical" in res_herc.optimizer_name
    assert len(res_herc.weights) == len(tickers)
    
    # 5. Kelly
    kelly = KellyOptimizer()
    res_kelly = kelly.optimize(returns)
    assert "Kelly" in res_kelly.optimizer_name
    assert len(res_kelly.weights) == len(tickers)
    
    # 6. CVaR Min
    cvar_min = ShortfallMinimizationOptimizer(risk_measure="CVaR")
    res_cvar = cvar_min.optimize(returns)
    assert "Shortfall" in res_cvar.optimizer_name
    assert len(res_cvar.weights) == len(tickers)
