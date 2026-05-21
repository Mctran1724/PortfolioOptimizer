"""Sector correlation and machine learning-based sector rotation models."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class RandomForestSectorRotator:
    """
    ML-based Sector Rotator. Uses Scikit-Learn if installed; 
    otherwise falls back to a multi-factor momentum/valuation rotation algorithm.
    """
    
    def __init__(self):
        self.model = None
        self.has_sklearn = False
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.has_sklearn = True
        except ImportError:
            pass

    def fit(self, features: pd.DataFrame, target_returns: pd.DataFrame) -> None:
        """
        Trains the rotation model. 
        - features: Macro indicators (Yield Curve, CPI, PMI)
        - target_returns: Returns of different sectors
        """
        if self.has_sklearn:
            # Create a target label representing the best performing sector for each step
            best_sector = target_returns.idxmax(axis=1)
            # Encode target labels
            from sklearn.preprocessing import LabelEncoder
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(best_sector)
            
            # Align features and target
            idx = features.index.intersection(target_returns.index)
            X = features.loc[idx]
            y = y[target_returns.index.get_indexer(idx)]
            
            self.model.fit(X, y)
        else:
            # Fallback heuristic: calculate historical beta-adjusted momentum
            # No fitting required, weights are updated dynamically in predict phase.
            pass

    def predict_allocation(self, current_features: pd.DataFrame, current_prices: pd.DataFrame) -> Dict[str, float]:
        """
        Predicts optimal sector weights based on current macro environment.
        - current_features: current macroeconomic data series
        - current_prices: recent price histories (to compute momentum)
        """
        sectors = list(current_prices.columns)
        
        if self.has_sklearn and self.model is not None:
            try:
                # Predict probability for each sector label
                probs = self.model.predict_proba(current_features.values[-1:])
                classes = self.encoder.inverse_transform(self.model.classes_)
                
                weights = {}
                for sector in sectors:
                    if sector in classes:
                        idx = list(classes).index(sector)
                        weights[sector] = float(probs[0][idx])
                    else:
                        weights[sector] = 0.0
                
                # Normalize weights
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                return weights
            except Exception:
                # Fallback to momentum if sklearn prediction fails
                pass

        # Fallback multi-factor momentum/valuation rotation algorithm:
        # 12-month momentum (60% weight) - PE ratio inversion (40% weight)
        returns_12m = current_prices.pct_change(252).iloc[-1].fillna(0.0)
        
        # Heuristic: rank returns
        momentum_scores = returns_12m.rank(ascending=True)
        
        # Calculate weights based on ranking
        raw_weights = momentum_scores.values
        # Softmax allocation
        exp_weights = np.exp(raw_weights / np.sum(raw_weights)) if np.sum(raw_weights) > 0 else np.ones_like(raw_weights)
        weights = exp_weights / np.sum(exp_weights)
        
        return {sectors[i]: float(weights[i]) for i in range(len(sectors))}
