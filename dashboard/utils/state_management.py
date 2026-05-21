"""Session state management utilities for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
from portfolio_optimizer.models.optimization_result import OptimizationResult

# Session state keys and their default initial values
STATE_DEFAULTS = {
    "tickers": ["QQQ", "SPY", "VNQ", "TLT", "TIP"],
    "historical_prices": None,
    "historical_returns": None,
    "optimization_results": {},
    "fred_api_key": "",
    "risk_free_rate": 0.045,
    "confidence_level": 0.95,
    "target_beta": 1.10,
    "max_drawdown_limit": 0.20,
    "reit_ticker": "VNQ",
    "bl_views": [],  # List of dicts describing active views
    "svf_params": {
        "crediting_rate": 0.035,
        "book_value": 100.0,
        "market_value": 98.0,
        "duration": 2.5,
        "smoothing": 0.15
    },
    "annuity_params": {
        "participation": 0.80,
        "cap": 0.08,
        "floor": 0.0,
        "spread": 0.0,
        "vol": 0.15,
        "term": 5.0,
        "simulations": 5000
    }
}

def initialize_state() -> None:
    """Ensures all required session state variables are initialized."""
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def get_state(key: str) -> Any:
    """Gets a value from session state, ensuring initialization first."""
    initialize_state()
    return st.session_state[key]

def set_state(key: str, value: Any) -> None:
    """Sets a value in session state, ensuring initialization first."""
    initialize_state()
    st.session_state[key] = value

def reset_optimization_results() -> None:
    """Resets the cached optimization results in session state."""
    st.session_state["optimization_results"] = {}
