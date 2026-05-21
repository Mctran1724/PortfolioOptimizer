"""Data model representing the results of a portfolio optimization."""

import pandas as pd
from typing import Dict, Any, Optional

class OptimizationResult:
    """Contains weight allocations and risk-return diagnostics from optimization solver."""
    
    def __init__(
        self,
        weights: Dict[str, float],
        expected_return: float,
        expected_volatility: float,
        sharpe_ratio: float,
        cvar: float,
        optimizer_name: str,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        self.weights = weights
        self.expected_return = expected_return
        self.expected_volatility = expected_volatility
        self.sharpe_ratio = sharpe_ratio
        self.cvar = cvar
        self.optimizer_name = optimizer_name
        self.additional_metrics = additional_metrics or {}

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(optimizer='{self.optimizer_name}', "
            f"Expected Return={self.expected_return:.4f}, "
            f"Expected Volatility={self.expected_volatility:.4f}, "
            f"Sharpe={self.sharpe_ratio:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts result metrics to dictionary."""
        return {
            "optimizer": self.optimizer_name,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "cvar": self.cvar,
            **self.additional_metrics
        }
