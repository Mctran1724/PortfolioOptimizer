"""UI components for displaying portfolio metrics in a premium grid layout."""

import streamlit as st
from typing import Dict, Any

def render_metrics_grid(metrics: Dict[str, Any]) -> None:
    """Renders a grid of custom glassmorphism metric cards."""
    
    # Extract keys with safe defaults
    sharpe = metrics.get("sharpe_ratio", 0.0)
    sortino = metrics.get("sortino_ratio", 0.0)
    treynor = metrics.get("treynor_ratio", 0.0)
    info_ratio = metrics.get("information_ratio", None)
    cvar = metrics.get("cvar", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    ret = metrics.get("expected_return", 0.0)
    vol = metrics.get("expected_volatility", 0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe:.3f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Excess Return / Volatility
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Return</div>
                <div class="metric-value">{ret * 100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Annualized mean return
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value">{sortino:.3f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Excess Return / Downside Vol
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Expected Volatility</div>
                <div class="metric-value">{vol * 100:.2f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Annualized volatility
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Conditional VaR (5%)</div>
                <div class="metric-value">{(cvar * 100):.2f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Expected loss in worst 5% days
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Treynor Ratio</div>
                <div class="metric-value">{treynor:.3f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Excess Return / Systematic Beta
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Maximum Drawdown</div>
                <div class="metric-value">{(max_dd * 100):.2f}%</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Peak-to-trough drop limit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        ir_text = f"{info_ratio:.3f}" if info_ratio is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Information Ratio</div>
                <div class="metric-value">{ir_text}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    Active Return / Tracking Error
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
