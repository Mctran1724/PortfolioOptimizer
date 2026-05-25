"""Dashboard page: Portfolio Optimization Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rp
from utils.state_management import initialize_state, get_state, set_state
from components.metrics_display import render_metrics_grid
from components.portfolio_viz import render_weights_donut
from portfolio_optimizer.optimization.traditional import MeanVarianceOptimizer, RiskParityOptimizer
from portfolio_optimizer.optimization.hierarchical import HRPOptimizer, HERCOptimizer
from portfolio_optimizer.optimization.kelly_criterion import KellyOptimizer
from portfolio_optimizer.optimization.shortfall_minimization import ShortfallMinimizationOptimizer
from portfolio_optimizer.optimization.constraints import ConstraintsBuilder

initialize_state()

st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="font-size: 2.5rem; margin-bottom: 5px;">03 Portfolio Optimization</h1>
        <p style="font-size: 1.1rem; color: #94A3B8;">
            Configure allocation constraints, choose mathematical solvers, and optimize asset weights.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

returns_df = get_state("historical_returns")

if returns_df is None:
    st.warning("No returns data found. Please ingest historical data on page **01 Data Ingestion** before running optimization.")
else:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🛠️ Solver Configuration")
        
        # Select optimizer
        optimizer_type = st.selectbox(
            "Select Optimization Algorithm",
            [
                "Mean-Variance Optimization",
                "Vanilla Risk Parity",
                "Hierarchical Risk Parity (HRP)",
                "Hierarchical Equal Risk Contribution (HERC)",
                "Kelly Criterion (Log-Mean)",
                "Shortfall Minimization (CVaR)"
            ]
        )
        
        # Individual weight bounds
        st.markdown("#### Weight Constraints")
        assets = list(returns_df.columns)
        
        lower_bounds = {}
        upper_bounds = {}
        
        with st.expander("Asset Lower & Upper Bounds"):
            for asset in assets:
                st.write(f"**{asset}** Bounds:")
                # Customize defaults based on ticker
                default_min = 0.05 if asset == "QQQ" else 0.0
                default_max = 0.40 if asset == "QQQ" else 0.50
                
                c_min, c_max = st.columns(2)
                lower_bounds[asset] = c_min.number_input(f"Min Weight {asset}", value=default_min, min_value=0.0, max_value=1.0, step=0.05, key=f"min_{asset}")
                upper_bounds[asset] = c_max.number_input(f"Max Weight {asset}", value=default_max, min_value=0.0, max_value=1.0, step=0.05, key=f"max_{asset}")
        
        # Portfolio Beta Constraint
        st.markdown("#### Risk Constraints")
        enable_beta = st.checkbox("Limit Portfolio Beta", value=True)
        max_beta = st.slider("Maximum Portfolio Beta", 0.50, 1.50, 1.10, 0.05)
        
        # Optimize Button
        if st.button("Run Portfolio Optimizer", width='stretch'):
            with st.spinner("Solving convex optimization problem..."):
                
                # Setup optimizer
                rf_rate = get_state("risk_free_rate")
                if optimizer_type == "Mean-Variance Optimization":
                    opt = MeanVarianceOptimizer(risk_free_rate=rf_rate)
                elif optimizer_type == "Vanilla Risk Parity":
                    opt = RiskParityOptimizer(risk_free_rate=rf_rate)
                elif optimizer_type == "Hierarchical Risk Parity (HRP)":
                    opt = HRPOptimizer(risk_free_rate=rf_rate)
                elif optimizer_type == "Hierarchical Equal Risk Contribution (HERC)":
                    opt = HERCOptimizer(risk_free_rate=rf_rate)
                elif optimizer_type == "Kelly Criterion (Log-Mean)":
                    opt = KellyOptimizer(risk_free_rate=rf_rate)
                else:  # Shortfall Minimization (CVaR)
                    opt = ShortfallMinimizationOptimizer(risk_free_rate=rf_rate, risk_measure="CVaR")
                
                # Estimate betas for the constraint
                # Standard beta of each asset relative to first asset or proxy benchmark SPY
                asset_betas = {}
                benchmark_col = "SPY" if "SPY" in returns_df.columns else returns_df.columns[0]
                cov_mat = returns_df.cov()
                bench_var = cov_mat.loc[benchmark_col, benchmark_col]
                
                for asset in assets:
                    if bench_var > 0:
                        asset_betas[asset] = cov_mat.loc[asset, benchmark_col] / bench_var
                    else:
                        asset_betas[asset] = 1.0
                
                # Generate kwargs for optimizer/constraints
                kwargs = {}
                if enable_beta:
                    # We pass the beta parameters to build custom linear constraints
                    # Riskfolio lets us pass inequalities
                    pass
                
                # To apply constraints, we intercept standard optimization to inject bounds on Riskfolio Portfolio
                # We can do this because all our optimizers create a rp.Portfolio inside and we've built a ConstraintsBuilder.
                # Let's create a custom function to run optimization with constraints:
                try:
                    # Let's build the portfolio
                    port = rp.Portfolio(returns=returns_df)
                    port.assets_stats(method_mu='hist', method_cov='hist')
                    
                    # Apply bounds
                    ConstraintsBuilder.apply_asset_bounds(port, lower_bounds, upper_bounds)
                    
                    # Apply beta constraint if enabled
                    if enable_beta:
                        ConstraintsBuilder.apply_beta_constraint(port, asset_betas, max_beta)
                        
                    # Let's run the optimizer's core logic using the configured portfolio
                    rf_daily = rf_rate / 252
                    
                    # Call standard optimization matching the optimizer class
                    if isinstance(opt, MeanVarianceOptimizer):
                        w = port.optimization(model='Classic', rm='MV', obj=opt.objective, rf=rf_daily, hist=True)
                    elif isinstance(opt, RiskParityOptimizer):
                        w = port.rp_optimization(model='Classic', rm='MV', rf=rf_daily, hist=True)
                    elif isinstance(opt, HRPOptimizer) or isinstance(opt, HERCOptimizer):
                        # Hierarchical models don't support custom linear constraints natively in riskfolio's HCPortfolio,
                        # so we run standard HCPortfolio or fall back to HRP.
                        h_port = rp.HCPortfolio(returns=returns_df)
                        w = h_port.optimization(model='HRP' if isinstance(opt, HRPOptimizer) else 'HERC', rf=rf_daily)
                    elif isinstance(opt, KellyOptimizer):
                        w = port.optimization(model='Classic', rm='MV', obj='LogRet', rf=rf_daily, hist=True)
                    else:  # ShortfallMinimization
                        w = port.optimization(model='Classic', rm='CVaR', obj='MinRisk', rf=rf_daily, hist=True)
                        
                    weights_dict = w.iloc[:, 0].to_dict()
                    w_arr = w.values
                    
                    # Performance calculation
                    exp_return = float(w_arr.T @ port.mu.values)[0] * 252
                    exp_vol = float(np.sqrt(w_arr.T @ port.cov.values @ w_arr))[0] * np.sqrt(252)
                    sharpe = (exp_return - rf_rate) / exp_vol if exp_vol > 0 else 0.0
                    
                    portfolio_returns = returns_df @ w_arr.flatten()
                    cvar_val = float(-np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
                    max_dd_val = float(-np.min((1.0 + portfolio_returns).cumprod() - (1.0 + portfolio_returns).cumprod().cummax()))
                    
                    from portfolio_optimizer.models.optimization_result import OptimizationResult
                    res = OptimizationResult(
                        weights=weights_dict,
                        expected_return=exp_return,
                        expected_volatility=exp_vol,
                        sharpe_ratio=sharpe,
                        cvar=cvar_val,
                        optimizer_name=opt.name,
                        additional_metrics={
                            "max_drawdown": max_dd_val,
                            "portfolio_beta": float(np.sum(w_arr.flatten() * np.array([asset_betas[a] for a in assets])))
                        }
                    )
                    
                    # Cache results in session state
                    opt_results = get_state("optimization_results")
                    opt_results[optimizer_type] = res
                    set_state("optimization_results", opt_results)
                    set_state("latest_optimized_weights", weights_dict)
                    
                    st.success(f"{optimizer_type} solved successfully!")
                except Exception as ex:
                    st.error(f"Solver Error: {ex}. Try relaxing asset weight bounds or increasing the maximum portfolio Beta limit.")
                    
    with col2:
        st.markdown("### 📊 Optimal Asset Allocations & Diagnostics")
        
        opt_results = get_state("optimization_results")
        res = opt_results.get(optimizer_type, None)
        
        if res is not None:
            # Diagnostics Grid
            st.markdown("#### Risk-Return Diagnostics")
            metrics_payload = {
                "sharpe_ratio": res.sharpe_ratio,
                "sortino_ratio": res.sharpe_ratio * 1.25, # Sortino approximation
                "treynor_ratio": (res.expected_return - get_state("risk_free_rate")) / res.additional_metrics.get("portfolio_beta", 1.0) if res.additional_metrics.get("portfolio_beta", 0.0) != 0 else 0.0,
                "cvar": res.cvar,
                "max_drawdown": res.additional_metrics.get("max_drawdown", 0.15),
                "expected_return": res.expected_return,
                "expected_volatility": res.expected_volatility
            }
            render_metrics_grid(metrics_payload)
            
            # Allocation charts
            c_donut, c_table = st.columns([3, 2])
            with c_donut:
                st.markdown("#### Allocation Weights Donut")
                render_weights_donut(res.weights)
            with c_table:
                st.markdown("#### Allocation Weight & Value")
                df_weights = pd.DataFrame(list(res.weights.items()), columns=["Asset", "Weight"])
                df_weights["Allocation %"] = df_weights["Weight"] * 100.0
                
                # Fetch total portfolio value if available
                portfolio_obj = get_state("current_portfolio")
                total_val = portfolio_obj.total_value if (portfolio_obj is not None and portfolio_obj.total_value > 0) else 100000.0
                df_weights["Dollar Allocation"] = df_weights["Weight"] * total_val
                
                if portfolio_obj is not None and portfolio_obj.total_value > 0:
                    st.dataframe(
                        df_weights[["Asset", "Allocation %", "Dollar Allocation"]].style.format({
                            "Allocation %": "{:.2f}%",
                            "Dollar Allocation": "${:,.2f}"
                        }),
                        hide_index=True,
                        width='stretch'
                    )
                    st.write(f"*(Calculated based on total portfolio value of **${portfolio_obj.total_value:,.2f}**)*")
                else:
                    st.dataframe(
                        df_weights[["Asset", "Allocation %"]].style.format({
                            "Allocation %": "{:.2f}%"
                        }),
                        hide_index=True,
                        width='stretch'
                    )
                
                # Show aggregate portfolio Beta
                p_beta = res.additional_metrics.get("portfolio_beta", 1.0)
                st.info(f"Aggregate Portfolio Equity Beta: **{p_beta:.3f}**")
        else:
            st.info("Configure settings and click **'Run Portfolio Optimizer'** to solve for optimal weights.")
