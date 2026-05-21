"""Helper functions for data manipulation, mathematical operations, and YAML configuration loading."""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads a YAML configuration file. Falls back to default config if none provided."""
    if config_path is None:
        # Determine path relative to this helper file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "config", "default_config.yaml")
        
    if not os.path.exists(config_path):
        # Return sensible hardcoded defaults if config file is missing
        return {
            "default_tickers": {
                "equities": ["QQQ", "SPY"],
                "reits": ["VNQ"],
                "fixed_income": ["TLT", "TIP"]
            },
            "stable_value_fund": {
                "proxy_bond_ticker": "SHY",
                "crediting_rate_smoothing": 0.15,
                "initial_crediting_rate": 0.035
            },
            "annuity": {
                "participation_rate": 0.80,
                "cap_rate": 0.08,
                "spread_rate": 0.0,
                "floor_rate": 0.0
            },
            "risk_free_rate": 0.045,
            "default_confidence_level": 0.95,
            "default_target_beta": 1.10,
            "default_max_drawdown_limit": 0.20,
            "default_mar": 0.0,
        }
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Removes NaN values and aligns indices of returns DataFrame."""
    return returns.dropna(how="all").fillna(0.0)

def generate_sample_returns(tickers: list, periods: int = 1000, start_date: str = "2022-01-01") -> pd.DataFrame:
    """Generates synthetic log-normal asset returns for fallback or testing purposes."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=periods, freq="B")
    
    # Define mean and vol for sample asset returns (simulating different behaviors)
    # e.g. QQQ: high return high vol, TLT: low return low vol negative correlation, etc.
    stats = {
        "QQQ": (0.0005, 0.015),
        "SPY": (0.0004, 0.012),
        "VNQ": (0.0002, 0.014),
        "TLT": (0.0001, 0.009),
        "TIP": (0.0001, 0.006),
        "SHY": (0.00005, 0.002),
        "SVF_Proxy": (0.00012, 0.0005),  # smoothed
        "Annuity_Proxy": (0.00025, 0.004) # modified index returns
    }
    
    data = {}
    for ticker in tickers:
        mu, sigma = stats.get(ticker, (0.0002, 0.01))
        # Add basic negative correlation between QQQ and TLT/TIP
        if ticker in ["TLT", "TIP", "SHY"] and "QQQ" in data:
            # negatively correlated noise component
            noise = -0.3 * data["QQQ"] + np.random.normal(mu, sigma, size=periods)
            data[ticker] = noise
        else:
            data[ticker] = np.random.normal(mu, sigma, size=periods)
            
    df = pd.DataFrame(data, index=dates)
    return df
