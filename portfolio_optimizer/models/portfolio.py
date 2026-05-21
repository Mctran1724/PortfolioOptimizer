"""Portfolio class to represent a collection of assets with weights."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from portfolio_optimizer.models.asset import Asset, EquityAsset, REITAsset, FixedIncomeAsset, SVFAsset, AnnuityAsset
from portfolio_optimizer.utils.constants import EQUITY, REIT, FIXED_INCOME, SVF, ANNUITY

class Portfolio:
    """Represents an investment portfolio of traditional and non-equity assets."""
    
    def __init__(self, assets: Dict[str, Asset], weights: Dict[str, float]):
        self.assets = assets
        # Normalize weights to sum to 1.0 if not already
        total_weight = sum(weights.values())
        if total_weight > 0 and not np.isclose(total_weight, 1.0):
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = weights.copy()

    def get_asset_weight(self, symbol: str) -> float:
        """Returns the weight of the asset in the portfolio."""
        return self.weights.get(symbol, 0.0)

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
