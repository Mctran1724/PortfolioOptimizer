"""Builder for defining and applying portfolio optimization constraints in Riskfolio-Lib."""

import numpy as np
import pandas as pd
import riskfolio as rp
from typing import Dict, List, Optional, Tuple

class ConstraintsBuilder:
    """Helper to construct and apply linear and asset constraints to a Riskfolio Portfolio."""
    
    @staticmethod
    def apply_asset_bounds(
        port: rp.Portfolio, 
        lower_bounds: Dict[str, float], 
        upper_bounds: Dict[str, float]
    ) -> None:
        """
        Applies individual upper and lower bounds on asset weights.
        - lower_bounds: Dict maps ticker -> min weight
        - upper_bounds: Dict maps ticker -> max weight
        """
        assets = list(port.returns.columns)
        
        # Build series
        w_min = []
        w_max = []
        
        for asset in assets:
            w_min.append(lower_bounds.get(asset, 0.0))
            w_max.append(upper_bounds.get(asset, 1.0))
            
        port.w_min = pd.Series(w_min, index=assets)
        port.w_max = pd.Series(w_max, index=assets)

    @staticmethod
    def apply_linear_constraints(
        port: rp.Portfolio,
        constraints_list: List[Tuple[np.ndarray, float, str]]
    ) -> None:
        """
        Applies a list of linear inequalities to the portfolio.
        Each constraint is a tuple: (coefficients_array, limit_value, sense)
        where sense is either "<=" or ">=".
        
        Converts inequalities into A_ineq * w <= B_ineq.
        """
        assets = list(port.returns.columns)
        n_assets = len(assets)
        
        A_rows = []
        B_vals = []
        
        for coeffs, limit, sense in constraints_list:
            if len(coeffs) != n_assets:
                raise ValueError(f"Constraint coefficients length {len(coeffs)} does not match number of assets {n_assets}.")
                
            if sense == "<=":
                A_rows.append(coeffs)
                B_vals.append(limit)
            elif sense == ">=":
                A_rows.append(-coeffs)
                B_vals.append(-limit)
            else:
                raise ValueError(f"Invalid constraint sense: {sense}. Must be '<=' or '>='.")
                
        if A_rows:
            A_matrix = np.vstack(A_rows)
            B_vector = np.array(B_vals).reshape(-1, 1)
            
            port.ainequality = pd.DataFrame(A_matrix, columns=assets)
            port.binequality = pd.DataFrame(B_vector)

    @classmethod
    def apply_beta_constraint(
        cls,
        port: rp.Portfolio,
        asset_betas: Dict[str, float],
        max_portfolio_beta: float
    ) -> None:
        """
        Applies a constraint to limit the weighted average beta of the portfolio:
        Sum( w_i * beta_i ) <= max_portfolio_beta
        """
        assets = list(port.returns.columns)
        betas = np.array([asset_betas.get(asset, 0.0) for asset in assets])
        
        # Apply as linear constraint: betas^T * w <= max_portfolio_beta
        cls.apply_linear_constraints(port, [(betas, max_portfolio_beta, "<=")])
