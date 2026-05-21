"""Visualizations for asset allocation weights and cumulative return performance."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict

def render_weights_donut(weights: Dict[str, float]) -> None:
    """Renders an interactive Plotly donut chart of asset weights."""
    df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
    df = df[df["Weight"] > 0.0]
    
    # Sort weights descending
    df = df.sort_values(by="Weight", ascending=False)
    
    # Custom color palette (matching Indigo / Purple / Pink gradients)
    colors = ['#6366F1', '#8B5CF6', '#EC4899', '#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#64748B']
    
    fig = go.Figure(data=[go.Pie(
        labels=df["Asset"],
        values=df["Weight"],
        hole=0.55,
        textinfo='label+percent',
        hoverinfo='label+value',
        marker=dict(colors=colors, line=dict(color='#0F172A', width=2))
    )])
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(family="Inter", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_cumulative_returns_chart(backtests: Dict[str, pd.Series]) -> None:
    """Renders cumulative returns time-series chart comparing optimized strategies."""
    fig = go.Figure()
    
    # Custom color map
    color_map = {
        "Benchmark (SPY)": "#94A3B8",
        "Concentrated (QQQ)": "#F472B6",
        "Mean-Variance Optimization": "#60A5FA",
        "Vanilla Risk Parity": "#34D399",
        "Hierarchical Risk Parity": "#A78BFA",
        "Kelly Criterion (Log-Mean)": "#F59E0B",
        "Shortfall Minimization (CVaR)": "#F87171"
    }
    
    for name, series in backtests.items():
        # Compute cumulative return
        cum_ret = (1 + series).cumprod() - 1
        
        # Determine color
        color = color_map.get(name, colors := np.random.choice(list(color_map.values())))
        
        fig.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret * 100.0,
            mode='lines',
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
        
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.4)',
        margin=dict(t=30, b=30, l=10, r=10),
        font=dict(family="Inter", size=12),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)',
            zeroline=False
        ),
        yaxis=dict(
            title="Cumulative Return (%)",
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.05)',
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
