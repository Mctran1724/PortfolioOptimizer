"""Data validation utilities for verifying the shape and cleanliness of asset returns."""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def validate_returns_data(df: pd.DataFrame, min_periods: int = 30) -> pd.DataFrame:
    """
    Validates returns DataFrame. 
    1. Ensures sorted datetime index.
    2. Drops columns with all NaNs.
    3. Fills remaining NaNs with 0.0.
    4. Replaces infinite values with 0.0.
    5. Checks if minimum observations limit is met.
    """
    if df.empty:
        raise ValueError("Returns DataFrame is empty.")

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Index must be convertible to DatetimeIndex. Error: {e}")
            
    # Sort by date index
    df = df.sort_index()
    
    # Drop columns with all NaNs
    df = df.dropna(how="all", axis=1)
    
    if len(df) < min_periods:
        raise ValueError(f"Insufficient returns observations. Have {len(df)}, need at least {min_periods}.")

    # Replace Inf with NaN and fill all NaNs with 0.0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    # Log warnings for zero variance columns
    for col in df.columns:
        if df[col].std() == 0.0:
            logger.warning(f"Asset '{col}' has zero variance (constant returns). This can cause numerical solver instability.")

    return df
