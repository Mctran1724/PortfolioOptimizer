"""Unit tests for risk and pricing metrics."""

import numpy as np
import pandas as pd
from portfolio_optimizer.analytics.risk_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_treynor_ratio
from portfolio_optimizer.analytics.tail_risk import calculate_var, calculate_cvar, calculate_max_drawdown, calculate_ulcer_index
from portfolio_optimizer.analytics.fixed_income import calculate_bond_duration_and_convexity, simulate_svf_crediting_rate
from portfolio_optimizer.analytics.annuities import calculate_indexed_annuity_payoff

def test_risk_metrics():
    # Simple returns with 0% risk free rate
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert isinstance(sharpe, float)
    
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert isinstance(sortino, float)
    
    treynor = calculate_treynor_ratio(returns, beta=1.0, risk_free_rate=0.0)
    assert isinstance(treynor, float)

def test_tail_risk():
    returns = np.array([0.01, 0.02, -0.05, 0.03, -0.01])
    
    var_95 = calculate_var(returns, confidence_level=0.95)
    cvar_95 = calculate_cvar(returns, confidence_level=0.95)
    max_dd = calculate_max_drawdown(returns)
    ui = calculate_ulcer_index(returns)
    
    assert var_95 >= 0.0
    assert cvar_95 >= var_95
    assert max_dd >= 0.0
    assert ui >= 0.0

def test_fixed_income():
    stats = calculate_bond_duration_and_convexity(
        face_value=1000.0,
        coupon_rate=0.05,
        yield_to_maturity=0.04,
        years_to_maturity=10.0,
        payment_frequency=2
    )
    
    assert stats["bond_price"] > 1000.0  # Premium bond since coupon > ytm
    assert stats["macaulay_duration"] > 0
    assert stats["modified_duration"] > 0
    assert stats["convexity"] > 0

def test_svf_smoothing():
    rate = simulate_svf_crediting_rate(
        current_crediting_rate=0.035,
        book_value=100.0,
        market_value=95.0,
        duration=2.5,
        smoothing_factor=0.15
    )
    # Market value is below book value, so crediting rate should decrease
    assert rate < 0.035
    assert rate >= 0.0

def test_annuity_payoff():
    # Test index return of 10% with 8% cap, 80% participation, 0% floor
    payoff_positive = calculate_indexed_annuity_payoff(np.array([0.10]), participation_rate=0.8, cap_rate=0.08, floor_rate=0.0)
    assert np.isclose(payoff_positive[0], 0.08)
    
    # Test index return of -10% with 0% floor
    payoff_negative = calculate_indexed_annuity_payoff(np.array([-0.10]), participation_rate=0.8, cap_rate=0.08, floor_rate=0.0)
    assert np.isclose(payoff_negative[0], 0.0)
