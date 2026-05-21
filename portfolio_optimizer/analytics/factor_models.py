"""Factor exposure analysis (Fama-French Five-Factor regression) and Marcenko-Pastur correlation filtering."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, Tuple

def run_fama_french_regression(
    asset_returns: pd.Series, 
    factor_returns: pd.DataFrame
) -> Dict[str, Any]:
    """
    Regresses asset returns against Fama-French 5 Factors (Mkt-RF, SMB, HML, RMW, CMA).
    Returns a dictionary of coefficients (exposures), t-stats, R-squared, and Alpha.
    """
    # Aligned data
    combined = pd.concat([asset_returns, factor_returns], axis=1).dropna()
    if len(combined) < 20:
        # Fallback dummy coefficients if data is too sparse
        return {
            "Alpha": 0.0,
            "Beta_Mkt": 1.0,
            "Beta_SMB": 0.0,
            "Beta_HML": 0.0,
            "Beta_RMW": 0.0,
            "Beta_CMA": 0.0,
            "R2": 0.0,
            "t_stats": {}
        }
        
    y = combined.iloc[:, 0]
    X = combined.drop(combined.columns[0], axis=1)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # Map constants
    params = model.params
    p_values = model.pvalues
    t_stats = model.tvalues
    
    return {
        "Alpha": float(params.get("const", 0.0)),
        "Beta_Mkt": float(params.get("Mkt-RF", params.get("Mkt", 0.0))),
        "Beta_SMB": float(params.get("SMB", 0.0)),
        "Beta_HML": float(params.get("HML", 0.0)),
        "Beta_RMW": float(params.get("RMW", 0.0)),
        "Beta_CMA": float(params.get("CMA", 0.0)),
        "R2": float(model.rsquared),
        "t_stats": t_stats.to_dict(),
        "p_values": p_values.to_dict()
    }

def marcenko_pastur_filter(correlation_matrix: pd.DataFrame, t_over_n: float) -> pd.DataFrame:
    """
    Denoises a correlation matrix using Marcenko-Pastur Random Matrix Theory.
    - correlation_matrix: input correlation DataFrame
    - t_over_n: aspect ratio Q = T/N where T is number of rows and N is number of columns.
    """
    if t_over_n <= 1.0:
        # Cannot reliably denoise if N > T, return original
        return correlation_matrix
        
    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix.values)
    
    # Theoretical maximum eigenvalue for pure noise
    # Max eigenvalue lambda_plus = sigma^2 * (1 + sqrt(1/Q))^2
    # Estimate sigma^2 as the mean of eigenvalues below the theoretical threshold
    # Since we don't know sigma upfront, we can estimate it iteratively or use a standard shortcut:
    # sigma^2 is roughly the variance explanation of the noise cluster, which is 1 - variance explained by signal.
    # A standard practical implementation is to estimate sigma^2 from the eigenvalues.
    q = t_over_n
    
    # We find sigma^2 by fitting the eigenvalues below lambda_plus.
    # For simplicity and robustness, we can set sigma = 1.0 (standard for correlation matrix) or estimate it.
    # Let's use a standard approximation of sigma^2 = 1.0 - max(eigenvalues) / sum(eigenvalues) or estimate from the lower part.
    # We will assume a baseline sigma^2 = 1.0 - (eigenvalues[-1] / len(eigenvalues))
    sigma_sq = 1.0 - (max(eigenvalues) / sum(eigenvalues))
    
    lambda_plus = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2
    
    # Denoising: Replace eigenvalues below lambda_plus with their average
    noise_indices = np.where(eigenvalues <= lambda_plus)[0]
    if len(noise_indices) > 0:
        avg_noise_eval = np.mean(eigenvalues[noise_indices])
        eigenvalues_filtered = eigenvalues.copy()
        eigenvalues_filtered[noise_indices] = avg_noise_eval
    else:
        eigenvalues_filtered = eigenvalues
        
    # Reconstruct correlation matrix
    recon = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
    
    # Normalize diagonal elements to 1.0 (to maintain correlation properties)
    diag = np.diag(recon)
    scaling_factor = np.sqrt(diag)
    recon_normalized = recon / np.outer(scaling_factor, scaling_factor)
    
    return pd.DataFrame(recon_normalized, index=correlation_matrix.index, columns=correlation_matrix.columns)
