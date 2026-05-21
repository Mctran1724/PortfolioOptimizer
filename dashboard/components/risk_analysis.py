"""Tail-risk visualizations including correlation heatmaps, drawdowns, and loss distribution tails."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from portfolio_optimizer.analytics.tail_risk import calculate_var, calculate_cvar

def render_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """Renders a correlation matrix heatmap with custom color schemes."""
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1.0,
        zmax=1.0,
        hovertemplate='Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(family="Inter", size=12),
        width=500,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_drawdown_waterline(returns: pd.Series) -> None:
    """Renders the historical drawdown chart (waterline area plot)."""
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100.0,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.15)',
        line=dict(color='#EF4444', width=1.5),
        name="Drawdown"
    ))
    
    fig.update_layout(
        title="Historical Portfolio Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        margin=dict(t=40, b=30, l=10, r=10),
        font=dict(family="Inter", size=12),
        yaxis=dict(
            title="Decline from Peak (%)",
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_tail_risk_histogram(returns: pd.Series, confidence_level: float = 0.95) -> None:
    """Renders a returns distribution histogram highlighting VaR and CVaR boundaries."""
    returns_pct = returns * 100.0
    var_val = calculate_var(returns, confidence_level) * 100.0
    cvar_val = calculate_cvar(returns, confidence_level) * 100.0
    
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=100,
        name="Daily Returns",
        marker_color='#6366F1',
        opacity=0.65
    ))
    
    # Add VaR cutoff line
    fig.add_vline(
        x=-var_val,
        line_width=2.5,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"VaR ({confidence_level * 100:.0f}%): -{var_val:.2f}%",
        annotation_position="top left",
        annotation_font=dict(color="#F59E0B")
    )
    
    # Add CVaR cutoff line
    fig.add_vline(
        x=-cvar_val,
        line_width=2.5,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"CVaR: -{cvar_val:.2f}%",
        annotation_position="top right",
        annotation_font=dict(color="#EF4444")
    )
    
    fig.update_layout(
        title="Return Distribution & Tail-Risk Boundaries",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        margin=dict(t=45, b=30, l=10, r=10),
        font=dict(family="Inter", size=12),
        xaxis=dict(
            title="Daily Return (%)",
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        yaxis=dict(
            title="Frequency Count",
            gridcolor='rgba(255, 255, 255, 0.05)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
