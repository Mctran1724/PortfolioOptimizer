"""Asset class hierarchy representing traditional and non-equity assets."""

from abc import ABC
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from portfolio_optimizer.utils.constants import EQUITY, REIT, FIXED_INCOME, SVF, ANNUITY

class Asset(ABC):
    """Abstract Base Class for all financial assets."""
    
    def __init__(self, symbol: str, name: str, asset_type: str):
        self.symbol = symbol
        self.name = name
        self.asset_type = asset_type
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol='{self.symbol}', name='{self.name}')"

class EquityAsset(Asset):
    """Represents a traditional equity stock or exchange-traded fund (ETF)."""
    
    def __init__(self, symbol: str, name: str, beta: float = 1.0):
        super().__init__(symbol, name, EQUITY)
        self.beta = beta

class REITAsset(Asset):
    """Represents a Real Estate Investment Trust (REIT) with specialized metrics."""
    
    def __init__(
        self, 
        symbol: str, 
        name: str, 
        ffo: float = 0.0,  # Funds From Operations
        nav: float = 0.0,  # Net Asset Value
        ffo_payout_ratio: float = 0.0, 
        debt_to_assets: float = 0.0,
        dividend_yield: float = 0.0
    ):
        super().__init__(symbol, name, REIT)
        self.ffo = ffo
        self.nav = nav
        self.ffo_payout_ratio = ffo_payout_ratio
        self.debt_to_assets = debt_to_assets
        self.dividend_yield = dividend_yield

class FixedIncomeAsset(Asset):
    """Represents a fixed income instrument (Nominal Treasury, Corporate Bond, or TIPS)."""
    
    def __init__(
        self, 
        symbol: str, 
        name: str, 
        duration: float = 0.0,  # Modified Duration
        convexity: float = 0.0, 
        yield_to_maturity: float = 0.0,
        is_tips: bool = False
    ):
        super().__init__(symbol, name, FIXED_INCOME)
        self.duration = duration
        self.convexity = convexity
        self.yield_to_maturity = yield_to_maturity
        self.is_tips = is_tips

class SVFAsset(Asset):
    """Represents a Stable Value Fund (SVF) utilizing book value accounting."""
    
    def __init__(
        self, 
        symbol: str, 
        name: str, 
        crediting_rate: float = 0.035, 
        book_value: float = 100.0,
        market_value: float = 100.0,
        duration: float = 2.5
    ):
        super().__init__(symbol, name, SVF)
        self.crediting_rate = crediting_rate
        self.book_value = book_value
        self.market_value = market_value
        self.duration = duration

    def simulate_crediting_rate_adjustment(self, smoothing_factor: float = 0.15) -> float:
        """
        Simulates the updated crediting rate based on the book-to-market convergence equation:
        CR_new = CR_old + (MV - BV) / (BV * duration) * smoothing_factor
        """
        if self.book_value <= 0 or self.duration <= 0:
            return self.crediting_rate
        
        rate_diff = (self.market_value - self.book_value) / (self.book_value * self.duration)
        new_rate = self.crediting_rate + rate_diff * smoothing_factor
        # Floor crediting rate at 0%
        return max(0.0, float(new_rate))

class AnnuityAsset(Asset):
    """Represents an In-Plan Annuity product (e.g. Fixed Indexed Annuity)."""
    
    def __init__(
        self, 
        symbol: str, 
        name: str, 
        participation_rate: float = 0.80, 
        cap_rate: float = 0.08, 
        spread_rate: float = 0.0, 
        floor_rate: float = 0.0,
        has_gmwb: bool = False
    ):
        super().__init__(symbol, name, ANNUITY)
        self.participation_rate = participation_rate
        self.cap_rate = cap_rate
        self.spread_rate = spread_rate
        self.floor_rate = floor_rate
        self.has_gmwb = has_gmwb

    def calculate_payoff(self, index_return: float) -> float:
        """
        Calculates the annuity credit based on index performance, cap, participation rate, and floor.
        payoff = Max(floor_rate, Min(cap_rate, index_return * participation_rate - spread_rate))
        """
        credit = index_return * self.participation_rate - self.spread_rate
        credit = min(self.cap_rate, credit)
        credit = max(self.floor_rate, credit)
        return float(credit)
