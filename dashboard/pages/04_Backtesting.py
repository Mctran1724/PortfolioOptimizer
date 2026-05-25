"""Dashboard page: Out-of-Sample Strategy Backtesting."""

import streamlit as st
import pandas as pd
import numpy as np
from utils.state_management import initialize_state, get_state
from components.portfolio_viz import render_cumulative_returns_chart
from portfolio_optimizer.analytics.risk_metrics import calculate_sharpe_ratio, calculate_sortino_ratio
from portfolio_optimizer.analytics.tail_risk import calculate_max_drawdown, calculate_cvar

initialize_state()

st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="font-size: 2.5rem; margin-bottom: 5px;">04 Strategy Backtesting</h1>
        <p style="font-size: 1.1rem; color: #94A3B8;">
            Compare the out-of-sample historical performance of optimized strategies against passive indices.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

returns_df = get_state("historical_returns")
opt_results = get_state("optimization_results")

if returns_df is None:
    st.warning("No returns data found. Please ingest historical data on page **01 Data Ingestion** before running backtests.")
elif not opt_results:
    st.info("No optimized portfolios found. Please configure and run an optimizer on page **03 Portfolio Optimization** to see comparison results.")
else:
    st.markdown("### 🏃 Backtest Execution")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Select Strategies to Compare")
        
        # Benchmarks are always included
        include_concentrated = st.checkbox("Include Concentrated Tech (QQQ)", value=True)
        include_spy = st.checkbox("Include Benchmark (SPY)", value=True)
        
        # User selection of optimized portfolios
        available_opts = list(opt_results.keys())
        selected_opts = st.multiselect(
            "Select Optimized Portfolios",
            options=available_opts,
            default=available_opts
        )
        
        run_bt = st.button("Run Historical Backtest", width='stretch')
        
    with col2:
        if run_bt or "backtest_runs" in st.session_state:
            # Run backtest
            backtests = {}
            assets = list(returns_df.columns)
            
            # 1. Base Benchmarks
            if include_concentrated and "QQQ" in returns_df.columns:
                backtests["Concentrated (QQQ)"] = returns_df["QQQ"]
            if include_spy and "SPY" in returns_df.columns:
                backtests["Benchmark (SPY)"] = returns_df["SPY"]
                
            # 2. Optimized Portfolios
            for opt_name in selected_opts:
                res = opt_results[opt_name]
                w_arr = np.array([res.weights.get(a, 0.0) for a in assets])
                
                # Daily portfolio returns
                port_ret = returns_df.values @ w_arr
                backtests[opt_name] = pd.Series(port_ret, index=returns_df.index)
                
            st.session_state["backtest_runs"] = backtests
            
            # Render comparison chart
            st.markdown("#### Cumulative Returns Performance Path")
            render_cumulative_returns_chart(backtests)
            
            # Calculate summary stats table
            st.markdown("#### Strategy Comparison Diagnostics")
            
            summary_data = []
            rf_rate = get_state("risk_free_rate")
            
            for name, series in backtests.items():
                cum_ret = (1.0 + series).prod() - 1
                ann_ret = np.mean(series) * 252
                vol = np.std(series, ddof=1) * np.sqrt(252)
                sharpe = calculate_sharpe_ratio(series, risk_free_rate=rf_rate)
                sortino = calculate_sortino_ratio(series, risk_free_rate=rf_rate)
                max_dd = calculate_max_drawdown(series)
                cvar_5 = calculate_cvar(series, confidence_level=0.95)
                
                summary_data.append({
                    "Strategy / Portfolio": name,
                    "Cumulative Return": f"{cum_ret * 100:.2f}%",
                    "Annualized Return": f"{ann_ret * 100:.2f}%",
                    "Volatility (Risk)": f"{vol * 100:.2f}%",
                    "Sharpe Ratio": f"{sharpe:.3f}",
                    "Sortino Ratio": f"{sortino:.3f}",
                    "Max Drawdown": f"{max_dd * 100:.2f}%",
                    "Conditional VaR (5%)": f"{cvar_5 * 100:.2f}%"
                })
                
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, hide_index=True, width='stretch')
            
        else:
            st.info("Select portfolios on the left panel and click **'Run Historical Backtest'** to view cumulative performance paths.")
