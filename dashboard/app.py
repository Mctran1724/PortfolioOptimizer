"""Main entry point for the Streamlit dashboard application."""

import os
import sys

# Add the project root to sys.path to allow absolute imports of portfolio_optimizer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from utils.state_management import initialize_state, get_state

# Page Configuration
st.set_page_config(
    page_title="Investment Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling
def load_custom_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
initialize_state()
load_custom_css()

# Sidebar branding
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #818CF8;">Antigravity Quant</h2>
        <p style="font-size: 0.8rem; color: #64748B;">Multi-Asset Risk & Portfolio Optimizer</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.05); margin: 0 0 20px 0;"/>
    """,
    unsafe_allow_html=True
)

# Main Page Content
st.markdown(
    """
    <div style="margin-bottom: 30px;">
        <h1 class="gradient-text" style="font-size: 3rem; margin-bottom: 5px;">
            Quantitative Portfolio Optimization Framework
        </h1>
        <p style="font-size: 1.2rem; color: #94A3B8;">
            Institutional-Grade Multi-Asset Risk Mitigation for Tax-Advantaged (Roth 401k) SDBA Portfolios
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Strategic Framework Overview")
    st.markdown(
        """
        This framework executes portfolio optimization and hedging models directly tailored for corporate employees 
        holding highly-concentrated technology stock positions. By utilizing a **Self-Directed Brokerage Account (SDBA)** 
        within a tax-advantaged account, we can dynamically reallocate capital without incurring capital gains tax drag.
        
        #### Key Quantitative Pillars:
        1. **Asymmetrical Risk Optimization**: Moving beyond symmetrical variance (Sharpe Ratio) toward Downside Deviation (Sortino Ratio) and Conditional Value at Risk (CVaR).
        2. **Noise Denoising**: Filtering noisy correlation matrices via the **Marcenko-Pastur Random Matrix Theorem** before feeding them to optimizers.
        3. **Factor Hedging**: Regressing portfolio assets against the **Fama-French Five-Factor Model** to construct precise hedges.
        4. **Alternative Asset Integration**: Sourcing and analyzing non-equity risk premia, including:
            * **Fixed Income sleeves** (TIPS and Nominal Treasuries dynamically adjusted via breakeven inflation curves)
            * **Stable Value Funds (SVF)** (exploiting unique corporate 401(k) book-value smoothing)
            * **In-Plan Annuities** (modeling indexed participation, cap, and floor structures)
        """
    )
    
    st.markdown("### 🗺️ Dashboard Navigation Guide")
    st.markdown(
        """
        - **01 Data Ingestion**: Setup tickers, fetch yfinance historical returns, FRED macroeconomic interest rates, and check data quality.
        - **02 Analytics**: Analyze asset-level metrics (duration/convexity, FFO yields, SVF crediting rates, annuity option pricing, and Fama-French betas).
        - **03 Optimization**: Run modern portfolio algorithms (MVO, Risk Parity, HRP, HERC, Kelly Criterion, and CVaR Minimization) with customized constraints.
        - **04 Backtesting**: Run out-of-sample backtests to compare optimized risk-mitigated strategies against buy-and-hold benchmarks.
        """
    )

with col2:
    st.markdown("### 📊 Active Universe Status")
    
    # Render interactive state summary cards
    tickers = get_state("tickers")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Monitored Assets</div>
            <div class="metric-value">{len(tickers)}</div>
            <div style="font-size: 0.9rem; color: #818CF8; margin-top: 5px;">
                {', '.join(tickers)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    rf_rate = get_state("risk_free_rate")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Risk-Free Rate (Annualized)</div>
            <div class="metric-value">{rf_rate * 100:.2f}%</div>
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 5px;">
                Implied from FRED DGS5 Yield Curve
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    has_data = get_state("historical_returns") is not None
    data_status = "READY" if has_data else "PENDING INGESTION"
    status_class = "delta-up" if has_data else "delta-down"
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dataset Status</div>
            <div class="metric-value {status_class}">{data_status}</div>
            <div style="font-size: 0.9rem; color: #64748B; margin-top: 5px;">
                Go to page <b>01 Data Ingestion</b> to load returns.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
        Quantitative Portfolio Optimization Framework &bull; Designed by Antigravity Quant Team &copy; 2026
    </div>
    """,
    unsafe_allow_html=True
)
