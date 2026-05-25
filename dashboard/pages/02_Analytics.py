"""Dashboard page: Quantitative Asset Analytics."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.state_management import initialize_state, get_state, set_state
from portfolio_optimizer.data.cache import DataCache
from portfolio_optimizer.analytics.fixed_income import calculate_bond_duration_and_convexity, calculate_breakeven_inflation, simulate_svf_crediting_rate
from portfolio_optimizer.analytics.annuities import monte_carlo_annuity_pricing, calculate_indexed_annuity_payoff

initialize_state()

st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="font-size: 2.5rem; margin-bottom: 5px;">02 Quantitative Asset Analytics</h1>
        <p style="font-size: 1.1rem; color: #94A3B8;">
            Evaluate non-equity risk premia, interest-rate sensitivities, and simulate in-plan insurance derivatives.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 Fixed-Income & TIPS",
    "🏢 REIT Analysis",
    "🛡️ Stable Value Funds (SVF)",
    "📈 In-Plan Annuity Option Pricing"
])

# ----------------------------------------------------
# TAB 1: Fixed-Income & TIPS
# ----------------------------------------------------
with tab1:
    st.markdown("### ⛓️ Duration, Convexity & Breakeven Inflation")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Bond Sensitivity Calculator")
        face_val = st.number_input("Bond Face Value ($)", value=1000.0, step=100.0)
        coupon_rate = st.slider("Annual Coupon Rate (%)", 0.0, 15.0, 4.0, 0.25) / 100
        ytm = st.slider("Yield to Maturity (YTM) (%)", 0.0, 15.0, 4.5, 0.25) / 100
        years = st.number_input("Years to Maturity", value=10.0, step=1.0)
        freq = st.selectbox("Coupon Frequency (per year)", [1, 2, 4, 12], index=1)
        
        # Calculate
        stats = calculate_bond_duration_and_convexity(face_val, coupon_rate, ytm, years, freq)
        
        st.markdown("---")
        st.write(f"**Bond Price**: ${stats['bond_price']:.2f}")
        st.write(f"**Macaulay Duration**: {stats['macaulay_duration']:.3f} years")
        st.write(f"**Modified Duration**: {stats['modified_duration']:.3f} years")
        st.write(f"**Convexity**: {stats['convexity']:.4f}")
        
    with col2:
        st.markdown("#### Market Breakeven Inflation Curve")
        # Load FRED macro data
        cache = DataCache()
        macro_df = cache.load("macro")
        
        if macro_df is not None:
            # We want to display T5YIE (5y breakeven), T10YIE (10y breakeven)
            st.markdown("Expected inflation rates derived from the US Treasury market yields:")
            fig = go.Figure()
            
            if "T5YIE" in macro_df.columns:
                fig.add_trace(go.Scatter(
                    x=macro_df.index, y=macro_df["T5YIE"] * 100.0 if macro_df["T5YIE"].max() < 0.2 else macro_df["T5YIE"],
                    name="5-Year Breakeven Inflation", line=dict(color="#6366F1", width=2)
                ))
            if "T10YIE" in macro_df.columns:
                fig.add_trace(go.Scatter(
                    x=macro_df.index, y=macro_df["T10YIE"] * 100.0 if macro_df["T10YIE"].max() < 0.2 else macro_df["T10YIE"],
                    name="10-Year Breakeven Inflation", line=dict(color="#C084FC", width=2)
                ))
                
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.4)',
                yaxis=dict(title="Inflation Rate (%)", gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(t=20, b=20, l=10, r=10),
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Show latest values
            val_5y = macro_df["T5YIE"].iloc[-1]
            val_10y = macro_df["T10YIE"].iloc[-1]
            val_5y_pct = val_5y * 100.0 if val_5y < 0.2 else val_5y
            val_10y_pct = val_10y * 100.0 if val_10y < 0.2 else val_10y
            st.info(f"Latest Market Rates: 5-Year Breakeven = **{val_5y_pct:.2f}%**, 10-Year Breakeven = **{val_10y_pct:.2f}%**.")
        else:
            st.warning("Go to page **01 Data Ingestion** to download FRED inflation expectation curves.")

# ----------------------------------------------------
# TAB 2: REIT Valuations
# ----------------------------------------------------
with tab2:
    st.markdown("### 🏢 Funds From Operations (FFO) & Real Estate Valuation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Valuation Modeler")
        reit_ticker = st.selectbox("Select REIT Ticker", ["VNQ", "O", "PLD"])
        
        # Load defaults or customize
        reit_defaults = {
            "VNQ": {"ffo": 4.5, "nav": 86.50, "payout": 85.0, "debt": 38.0, "yield": 4.2},
            "O": {"ffo": 4.0, "nav": 58.00, "payout": 88.0, "debt": 41.0, "yield": 5.8},
            "PLD": {"ffo": 5.2, "nav": 120.00, "payout": 78.0, "debt": 32.0, "yield": 2.9}
        }
        
        def_vals = reit_defaults.get(reit_ticker, reit_defaults["VNQ"])
        
        ffo = st.number_input("Funds From Operations (FFO) per share ($)", value=def_vals["ffo"])
        nav = st.number_input("Net Asset Value (NAV) per share ($)", value=def_vals["nav"])
        payout = st.slider("FFO Payout Ratio (%)", 50.0, 100.0, def_vals["payout"]) / 100
        debt = st.slider("Debt-to-Assets (%)", 10.0, 80.0, def_vals["debt"]) / 100
        
        # Simulate price
        price = st.number_input("Current REIT Share Price ($)", value=nav * 0.95) # 5% discount
        
    with col2:
        st.markdown("#### Valuation Diagnostics")
        ffo_yield = ffo / price
        premium_discount = (price - nav) / nav
        
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">FFO Yield</div>
                <div class="metric-value">{ffo_yield * 100:.2f}%</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 5px;">
                    Compares REIT cash generation to market share price.
                </div>
            </div>
            <div class="metric-card" style="margin-bottom: 15px;">
                <div class="metric-label">NAV Premium / Discount</div>
                <div class="metric-value" style="color: {'#34D399' if premium_discount >= 0 else '#F87171'};">
                    {premium_discount * 100:.2f}% { 'Premium' if premium_discount >= 0 else 'Discount' }
                </div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 5px;">
                    Discount is a typical buy indicator for high-quality properties.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Risk assessment
        st.markdown("#### Leverage Risk Assessment")
        if debt > 0.45:
            st.error(f"High Leverage Warning: Debt-to-Assets ratio ({debt*100:.1f}%) exceeds the 45% threshold. Highly vulnerable to rate hikes.")
        else:
            st.success(f"Debt-to-Assets ratio ({debt*100:.1f}%) is in the conservative structural safety zone.")

# ----------------------------------------------------
# TAB 3: Stable Value Funds
# ----------------------------------------------------
with tab3:
    st.markdown("### 🛡️ SVF Book-Value Smoothing & Crediting Rate Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Adjust SVF Modeling Inputs")
        cr_old = st.slider("Current Crediting Rate (%)", 1.0, 8.0, 3.5, 0.1) / 100
        bv = st.number_input("SVF Book Value ($)", value=100.0)
        mv = st.slider("Underlying Bond Portfolio Market Value ($)", 85.0, 115.0, 96.0, 0.5)
        d_svf = st.slider("Sleeve Duration (years)", 1.5, 4.0, 2.5, 0.1)
        smoothing = st.slider("Smoothing / Convergence Factor", 0.05, 0.50, 0.15, 0.05)
        
        # Calculate
        cr_new = simulate_svf_crediting_rate(cr_old, bv, mv, d_svf, smoothing)
        
        st.markdown("---")
        st.write(f"**Crediting Rate Adjustment**: {(cr_new - cr_old)*100:+.2f}%")
        st.write(f"**New Crediting Rate**: {cr_new * 100:.2f}%")
        
    with col2:
        st.markdown("#### Dynamics of Book-Value Smoothing")
        st.markdown(
            """
            **Stable Value Funds (SVF)** exploit contract wraps that insulate 401(k) plan participants from daily bond 
            price volatility. While money market funds yield overnight rates, SVFs invest in intermediate-term bonds 
            to capture term premium, utilizing book value accounting to smooth gains/losses.
            
            The crediting rate adjustment formula simulates this smoothing mechanism:
            $$CR_{new} = CR_{old} + \\frac{MV - BV}{BV \\cdot d} \\times \\text{smoothing}$$
            """
        )
        
        # Generate chart comparing crediting rate changes based on market value shifts
        mv_range = np.linspace(90.0, 110.0, 50)
        rates = [simulate_svf_crediting_rate(cr_old, bv, m, d_svf, smoothing) * 100.0 for m in mv_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mv_range, y=rates, name="New Crediting Rate", line=dict(color="#10B981", width=2.5)))
        # Current reference line
        fig.add_vline(x=bv, line_dash="dash", line_color="#EF4444", annotation_text="Book Value Equilibrium ($100)")
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.4)',
            xaxis=dict(title="Underlying Bond Market Value ($)", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title="Adjusted Crediting Rate (%)", gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, width='stretch')

# ----------------------------------------------------
# TAB 4: In-Plan Annuity Option Pricing
# ----------------------------------------------------
with tab4:
    st.markdown("### 📈 Fixed Indexed Annuity (FIA) Monte Carlo Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### FIA Contract Settings")
        part_rate = st.slider("Participation Rate (%)", 50, 100, 80, 5) / 100
        cap_rate = st.slider("Indexed Cap Rate (%)", 2, 15, 8, 1) / 100
        floor_rate = st.slider("Floor Guarantee (%)", 0, 5, 0, 1) / 100
        
        st.markdown("#### Monte Carlo Parameters")
        term = st.slider("Contract Term (Years)", 1, 10, 5, 1)
        sim_val = st.number_input("Monte Carlo Iterations", value=5000, step=1000)
        vol = st.slider("Index Volatility (%)", 5.0, 35.0, 15.0, 1.0) / 100
        rf_rate = get_state("risk_free_rate")
        
        if st.button("Run Stochastic Valuation", width='stretch'):
            with st.spinner("Valuing index option paths..."):
                res = monte_carlo_annuity_pricing(
                    index_start=100.0,
                    participation_rate=part_rate,
                    cap_rate=cap_rate,
                    floor_rate=floor_rate,
                    volatility=vol,
                    risk_free_rate=rf_rate,
                    term_years=term,
                    simulations=sim_val
                )
                st.session_state["annuity_pricing_result"] = res
                st.success("Stochastic pricing completed!")
                
    with col2:
        res = st.session_state.get("annuity_pricing_result", None)
        
        if res is not None:
            st.markdown("#### Valuation Summary")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom: 15px;">
                        <div class="metric-label">Annuity Fair Value (Per $100)</div>
                        <div class="metric-value">${res['fair_value']:.2f}</div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 5px;">
                            Discounted expectation of payoff.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom: 15px;">
                        <div class="metric-label">Downside Put Option Value</div>
                        <div class="metric-value">${res['protection_option_value']:.2f}</div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 5px;">
                            Value of the index floor guarantee.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown(
                f"""
                - **Expected ending annuity index**: **{res['expected_ending_value']:.2f}** (vs **{res['simulated_index_ending_value']:.2f}** for unhedged index).
                - The stochastic pricing suggests the embedded put option (protecting against index declines below 0%) accounts for **{res['protection_option_value']/res['fair_value']*100:.1f}%** of the annuity's fair value.
                - This actuarial guarantee provides the Roth 401(k) portfolio with a hard floor defensive boundary.
                """
            )
        else:
            st.info("Adjust settings and click **'Run Stochastic Valuation'** to price the index wrap options.")
