"""Base class for all portfolio optimizers in the framework."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from portfolio_optimizer.models.optimization_result import OptimizationResult

class BaseOptimizer(ABC):
    """Abstract Base Class defining the interface for all portfolio optimization engines."""
    
    def __init__(self, name: str, risk_free_rate: float = 0.045):
        self.name = name
        self.risk_free_rate = risk_free_rate

    @abstractmethod
    def optimize(
        self, 
        returns: pd.DataFrame, 
        covariance: Optional[pd.DataFrame] = None, 
        **kwargs
    ) -> OptimizationResult:
        """
        Executes the optimization algorithm on historical asset returns.
        - returns: pd.DataFrame of historical asset returns (sorted by date index)
        - covariance: Optional pd.DataFrame of the asset covariance matrix (useful for filtered covariances)
        Returns an OptimizationResult object containing weights and diagnostics.
        """
        pass
