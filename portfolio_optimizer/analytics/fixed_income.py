"""Fixed-income analytics including duration, convexity, breakeven inflation, and Stable Value Fund models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

def calculate_bond_duration_and_convexity(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    years_to_maturity: float,
    payment_frequency: int = 2
) -> Dict[str, float]:
    """
    Calculates Macaulay Duration, Modified Duration, and Convexity for a standard coupon bond.
    - face_value: Face value of the bond (e.g. 1000)
    - coupon_rate: Annual coupon rate (e.g. 0.05)
    - yield_to_maturity: Annualized yield to maturity (e.g. 0.04)
    - years_to_maturity: Time to maturity in years (e.g. 10)
    - payment_frequency: Number of coupon payments per year (default 2 for semi-annual)
    """
    total_periods = int(years_to_maturity * payment_frequency)
    period_yield = yield_to_maturity / payment_frequency
    coupon_payment = (coupon_rate * face_value) / payment_frequency
    
    cash_flows = []
    times = []
    
    for t in range(1, total_periods + 1):
        cf = coupon_payment
        if t == total_periods:
            cf += face_value
        cash_flows.append(cf)
        times.append(t / payment_frequency)
        
    cash_flows = np.array(cash_flows)
    times = np.array(times)
    
    # Present values of cash flows
    pv_factors = 1 / (1 + period_yield) ** (times * payment_frequency)
    pv_cash_flows = cash_flows * pv_factors
    bond_price = np.sum(pv_cash_flows)
    
    if bond_price == 0:
        return {"macaulay_duration": 0.0, "modified_duration": 0.0, "convexity": 0.0}
        
    # Macaulay Duration = Sum(t * PV_t) / Price
    macaulay_duration = np.sum(times * pv_cash_flows) / bond_price
    
    # Modified Duration = Macaulay Duration / (1 + y/m)
    modified_duration = macaulay_duration / (1 + period_yield)
    
    # Convexity = Sum( t * (t + 1/m) * PV_t ) / (Price * (1 + y/m)^2)
    t_terms = times * (times + 1/payment_frequency)
    convexity = np.sum(t_terms * pv_cash_flows) / (bond_price * (1 + period_yield) ** 2)
    
    return {
        "bond_price": float(bond_price),
        "macaulay_duration": float(macaulay_duration),
        "modified_duration": float(modified_duration),
        "convexity": float(convexity)
    }

def calculate_breakeven_inflation(nominal_yield: float, tips_yield: float) -> float:
    """
    Calculates the breakeven inflation rate.
    Breakeven Inflation = Nominal Yield - Real (TIPS) Yield
    """
    return float(nominal_yield - tips_yield)

def simulate_svf_crediting_rate(
    current_crediting_rate: float,
    book_value: float,
    market_value: float,
    duration: float,
    smoothing_factor: float = 0.15
) -> float:
    """
    Calculates the Stable Value Fund (SVF) crediting rate adjustment based on book-to-market convergence:
    CR_new = CR_old + (MV - BV) / (BV * duration) * smoothing_factor
    """
    if book_value <= 0.0 or duration <= 0.0:
        return current_crediting_rate
        
    adjustment = (market_value - book_value) / (book_value * duration)
    new_rate = current_crediting_rate + adjustment * smoothing_factor
    return max(0.0, float(new_rate))
