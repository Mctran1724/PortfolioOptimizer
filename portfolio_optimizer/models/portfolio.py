"""Portfolio class to represent a collection of assets with weights and dollar amounts."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from portfolio_optimizer.models.asset import Asset, EquityAsset, REITAsset, FixedIncomeAsset, SVFAsset, AnnuityAsset
from portfolio_optimizer.utils.constants import EQUITY, REIT, FIXED_INCOME, SVF, ANNUITY

class Portfolio:
    """Represents an investment portfolio of traditional and non-equity assets with values."""
    
    def __init__(
        self, 
        assets: Union[Dict[str, Asset], List[Asset]], 
        weights: Optional[Dict[str, float]] = None,
        amounts: Optional[Dict[str, float]] = None,
        total_value: Optional[float] = None
    ):
        # Handle assets as list or dict
        if isinstance(assets, list):
            self.assets = {asset.symbol: asset for asset in assets}
        else:
            self.assets = assets.copy()
            
        self.amounts = {}
        self.weights = {}
        self.total_value = 0.0

        if amounts is not None:
            self.amounts = {k: float(v) for k, v in amounts.items()}
            self.total_value = float(sum(self.amounts.values()))
            if self.total_value > 0:
                self.weights = {k: v / self.total_value for k, v in self.amounts.items()}
            else:
                self.weights = {k: 0.0 for k in self.amounts.keys()}
        elif weights is not None:
            # Handle weights normalization if sum is not 1.0
            total_weight = sum(weights.values())
            if total_weight > 0 and not np.isclose(total_weight, 1.0):
                self.weights = {k: v / total_weight for k, v in weights.items()}
            else:
                self.weights = weights.copy()
            
            if total_value is not None:
                self.total_value = float(total_value)
                self.amounts = {k: w * self.total_value for k, w in self.weights.items()}
            else:
                self.total_value = 0.0
                self.amounts = {k: 0.0 for k in self.weights.keys()}
        else:
            # If neither is provided, initialize empty weights/amounts for all assets
            self.weights = {k: 0.0 for k in self.assets.keys()}
            self.amounts = {k: 0.0 for k in self.assets.keys()}
            self.total_value = 0.0

        # Fill in missing weights/amounts for any assets not specified
        for symbol in self.assets:
            if symbol not in self.weights:
                self.weights[symbol] = 0.0
            if symbol not in self.amounts:
                self.amounts[symbol] = 0.0

    def get_asset_weight(self, symbol: str) -> float:
        """Returns the weight of the asset in the portfolio."""
        return self.weights.get(symbol, 0.0)

    def get_asset_percentage(self, symbol: str) -> float:
        """Returns the percentage weight (0-100) of the asset in the portfolio."""
        return self.get_asset_weight(symbol) * 100.0

    def get_asset_amount(self, symbol: str) -> float:
        """Returns the dollar amount invested in the asset."""
        return self.amounts.get(symbol, 0.0)

    def calculate_portfolio_beta(self) -> float:
        """Calculates the weighted average Beta of the equity portions of the portfolio."""
        portfolio_beta = 0.0
        for symbol, asset in self.assets.items():
            if isinstance(asset, EquityAsset):
                portfolio_beta += self.weights.get(symbol, 0.0) * asset.beta
            elif isinstance(asset, REITAsset):
                # REITs typically have some equity beta (e.g. 0.5-0.8)
                portfolio_beta += self.weights.get(symbol, 0.0) * 0.7
            # SVF, Annuity, and Fixed Income are assumed to have near-zero equity beta
        return float(portfolio_beta)

    def calculate_portfolio_duration(self) -> float:
        """Calculates the weighted average duration of the fixed income sleeve of the portfolio."""
        total_fi_weight = 0.0
        weighted_duration = 0.0
        
        for symbol, asset in self.assets.items():
            weight = self.weights.get(symbol, 0.0)
            if isinstance(asset, FixedIncomeAsset):
                weighted_duration += weight * asset.duration
                total_fi_weight += weight
            elif isinstance(asset, SVFAsset):
                weighted_duration += weight * asset.duration
                total_fi_weight += weight
                
        if total_fi_weight == 0:
            return 0.0
            
        return float(weighted_duration / total_fi_weight)

    def get_allocation_by_asset_type(self) -> Dict[str, float]:
        """Aggregates portfolio weights by asset type."""
        allocations = {EQUITY: 0.0, REIT: 0.0, FIXED_INCOME: 0.0, SVF: 0.0, ANNUITY: 0.0}
        for symbol, asset in self.assets.items():
            weight = self.weights.get(symbol, 0.0)
            if asset.asset_type in allocations:
                allocations[asset.asset_type] += weight
        return allocations
