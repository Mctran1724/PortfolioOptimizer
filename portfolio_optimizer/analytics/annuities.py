"""Annuity pricing models including indexing payoffs and Monte Carlo simulations for variable annuity riders."""

import numpy as np
from typing import Union, List, Dict, Any

def calculate_indexed_annuity_payoff(
    index_returns: Union[np.ndarray, list],
    participation_rate: float = 0.80,
    cap_rate: float = 0.08,
    spread_rate: float = 0.0,
    floor_rate: float = 0.0
) -> np.ndarray:
    """
    Calculates the returns credited to an indexed annuity based on index returns.
    Payoff = Max(floor_rate, Min(cap_rate, index_returns * participation_rate - spread_rate))
    """
    returns_arr = np.array(index_returns)
    credited = returns_arr * participation_rate - spread_rate
    credited = np.minimum(cap_rate, credited)
    credited = np.maximum(floor_rate, credited)
    return credited

def monte_carlo_annuity_pricing(
    index_start: float = 100.0,
    participation_rate: float = 0.80,
    cap_rate: float = 0.08,
    floor_rate: float = 0.0,
    volatility: float = 0.15,
    risk_free_rate: float = 0.045,
    term_years: float = 5.0,
    simulations: int = 10000,
    annual_steps: int = 1
) -> Dict[str, Any]:
    """
    Simulates index paths and calculates the fair value of an Indexed Annuity.
    Uses Geometric Brownian Motion to simulate index price paths.
    """
    np.random.seed(42)
    dt = 1.0 / annual_steps
    total_steps = int(term_years * annual_steps)
    
    # Simulate paths: shape (simulations, total_steps + 1)
    paths = np.zeros((simulations, total_steps + 1))
    paths[:, 0] = index_start
    
    for t in range(1, total_steps + 1):
        z = np.random.normal(0, 1, simulations)
        paths[:, t] = paths[:, t-1] * np.exp(
            (risk_free_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
        )
        
    # Calculate returns for each period
    # If annual steps = 1, period is yearly.
    period_returns = (paths[:, 1:] - paths[:, :-1]) / paths[:, :-1]
    
    # Calculate credited returns per period
    credited_returns = np.zeros_like(period_returns)
    for t in range(total_steps):
        credited_returns[:, t] = calculate_indexed_annuity_payoff(
            period_returns[:, t],
            participation_rate,
            cap_rate,
            0.0,
            floor_rate
        )
        
    # Reconstruct annuity account value paths
    annuity_values = np.zeros((simulations, total_steps + 1))
    annuity_values[:, 0] = index_start
    for t in range(1, total_steps + 1):
        annuity_values[:, t] = annuity_values[:, t-1] * (1.0 + credited_returns[:, t-1])
        
    # Ending values
    ending_annuity_vals = annuity_values[:, -1]
    ending_index_vals = paths[:, -1]
    
    # Discounted fair value today
    discount_factor = np.exp(-risk_free_rate * term_years)
    fair_value = np.mean(ending_annuity_vals) * discount_factor
    
    # Valuation of the downside protection (the synthetic put option value)
    # Put Payoff = Max(0, Annuity Value - Index Value)
    protection_payoff = np.maximum(0, ending_annuity_vals - ending_index_vals)
    protection_value = np.mean(protection_payoff) * discount_factor
    
    return {
        "fair_value": float(fair_value),
        "expected_ending_value": float(np.mean(ending_annuity_vals)),
        "protection_option_value": float(protection_value),
        "median_ending_value": float(np.median(ending_annuity_vals)),
        "simulated_index_ending_value": float(np.mean(ending_index_vals))
    }
