"""Constants used across the Portfolio Optimization Framework."""

# Asset type identifiers
EQUITY = "Equity"
REIT = "REIT"
FIXED_INCOME = "FixedIncome"
SVF = "StableValueFund"
ANNUITY = "Annuity"

# Default names for display
ASSET_TYPE_NAMES = {
    EQUITY: "Equities / Equity ETFs",
    REIT: "Real Estate Investment Trusts",
    FIXED_INCOME: "Fixed Income (Bonds / TIPS)",
    SVF: "Stable Value Funds",
    ANNUITY: "In-Plan Annuities / Insurance Products",
}

# FRED Economic Data Series Tickers
FRED_TICKERS = {
    "T5YIE": "5-Year Breakeven Inflation Rate",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "DGS5": "5-Year Treasury Rate",
    "DGS10": "10-Year Treasury Rate",
    "DFII5": "5-Year TIPS Real Yield",
    "DFII10": "10-Year TIPS Real Yield",
}

# Default settings
DEFAULT_RISK_FREE_RATE = 0.045
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MAR = 0.0
