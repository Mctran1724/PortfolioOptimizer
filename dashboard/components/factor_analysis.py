"""Visualizations for Fama-French factor exposures and regression diagnostics."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

def render_factor_exposures_bar(factor_results: Dict[str, Any], title: str = "Factor Exposures (Betas)") -> None:
    """Renders a horizontal bar chart displaying Fama-French factor exposures."""
    # Map factor names
    factors = ["Market (Mkt-RF)", "Size (SMB)", "Value (HML)", "Profitability (RMW)", "Investment (CMA)"]
    betas = [
        factor_results.get("Beta_Mkt", 0.0),
        factor_results.get("Beta_SMB", 0.0),
        factor_results.get("Beta_HML", 0.0),
        factor_results.get("Beta_RMW", 0.0),
        factor_results.get("Beta_CMA", 0.0)
    ]
    
    # Custom color: Green for positive, Red for negative
    colors = ['#34D399' if b >= 0 else '#F87171' for b in betas]
    
    fig = go.Figure(data=[go.Bar(
        y=factors,
        x=betas,
        orientation='h',
        marker_color=colors,
        hovertemplate='Exposure (Beta): %{x:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        margin=dict(t=40, b=30, l=10, r=10),
        font=dict(family="Inter", size=12),
        xaxis=dict(
            title="Factor Beta Coefficient",
            gridcolor='rgba(255, 255, 255, 0.05)',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=1
        ),
        yaxis=dict(
            autorange="reversed"  # Keeping market factor at the top
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_factor_regression_table(factor_results: Dict[str, Any]) -> None:
    """Displays a clean metrics summary table of Fama-French regression outputs."""
    factors = ["Intercept (Alpha)", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    
    # Get values
    betas = [
        factor_results.get("Alpha", 0.0) * 252, # Annualize Alpha
        factor_results.get("Beta_Mkt", 0.0),
        factor_results.get("Beta_SMB", 0.0),
        factor_results.get("Beta_HML", 0.0),
        factor_results.get("Beta_RMW", 0.0),
        factor_results.get("Beta_CMA", 0.0)
    ]
    
    t_stats = factor_results.get("t_stats", {})
    p_vals = factor_results.get("p_values", {})
    
    t_list = [
        t_stats.get("const", 0.0),
        t_stats.get("Mkt-RF", t_stats.get("Mkt", 0.0)),
        t_stats.get("SMB", 0.0),
        t_stats.get("HML", 0.0),
        t_stats.get("RMW", 0.0),
        t_stats.get("CMA", 0.0)
    ]
    
    p_list = [
        p_vals.get("const", 0.0),
        p_vals.get("Mkt-RF", p_vals.get("Mkt", 0.0)),
        p_vals.get("SMB", 0.0),
        p_vals.get("HML", 0.0),
        p_vals.get("RMW", 0.0),
        p_vals.get("CMA", 0.0)
    ]
    
    df = pd.DataFrame({
        "Factor / Coefficient": factors,
        "Estimate (Beta / Ann. Alpha)": [f"{b:.4f}" if i==0 else f"{b:.3f}" for i, b in enumerate(betas)],
        "t-Statistic": [f"{t:.2f}" for t in t_list],
        "p-Value": [f"{p:.4f}" for p in p_list],
        "Significance": ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in p_list]
    })
    
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Alpha is annualized. Significance flags: *** p<0.01, ** p<0.05, * p<0.1. Model R2 = " + f"{factor_results.get('R2', 0.0):.2f}")
