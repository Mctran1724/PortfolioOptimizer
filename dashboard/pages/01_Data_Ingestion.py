"""Dashboard page: Data Ingestion and Validation."""

import os

import pandas as pd
import streamlit as st
from utils.state_management import get_state, initialize_state, set_state

from portfolio_optimizer.data.cache import DataCache
from portfolio_optimizer.data.ingestion import DataIngestionEngine
from portfolio_optimizer.data.validators import validate_returns_data

initialize_state()

# Page title
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="font-size: 2.5rem; margin-bottom: 5px;">01 Data Ingestion</h1>
        <p style="font-size: 1.1rem; color: #94A3B8;">
            Configure portfolio universe, download historical asset prices, and validate return data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Universe Configuration")

    # Text input for tickers list
    tickers_str = st.text_input(
        "Asset Tickers (comma-separated)",
        value=", ".join(get_state("tickers")),
        help="List of equity and bond ETF tickers to include in the portfolio.",
    )

    # Process tickers input
    tickers_list = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    # Sync portfolio_amounts in session state with tickers_list
    current_amounts = get_state("portfolio_amounts") or {}
    updated_amounts = {}
    for t in tickers_list:
        updated_amounts[t] = current_amounts.get(t, 10000.0)

    st.markdown("#### 💰 Current Holdings (Dollar Values)")
    with st.expander("Configure Asset Values", expanded=True):
        for asset in tickers_list:
            updated_amounts[asset] = st.number_input(
                f"Holding value for {asset} ($)",
                value=float(updated_amounts[asset]),
                min_value=0.0,
                step=1000.0,
                key=f"amt_input_{asset}"
            )

    # Save the updated amounts to session state
    set_state("portfolio_amounts", updated_amounts)

    # Build Asset and Portfolio objects to cache in session state
    from portfolio_optimizer.models.portfolio import Portfolio
    from portfolio_optimizer.models.asset import Asset, EquityAsset, REITAsset, FixedIncomeAsset, SVFAsset, AnnuityAsset
    
    # Estimate betas for the assets if returns are available
    asset_betas = {}
    returns_df = get_state("historical_returns")
    if returns_df is not None:
        benchmark_col = "SPY" if "SPY" in returns_df.columns else returns_df.columns[0]
        cov_mat_raw = returns_df.cov()
        if benchmark_col in cov_mat_raw.columns:
            bench_var = cov_mat_raw.loc[benchmark_col, benchmark_col]
            for asset in tickers_list:
                if asset in cov_mat_raw.columns and bench_var > 0:
                    asset_betas[asset] = cov_mat_raw.loc[asset, benchmark_col] / bench_var
                else:
                    asset_betas[asset] = 1.0
        else:
            for asset in tickers_list:
                asset_betas[asset] = 1.0
    else:
        for asset in tickers_list:
            asset_betas[asset] = 1.0

    def create_asset_by_ticker(ticker: str, beta: float = 1.0) -> Asset:
        ticker_upper = ticker.upper()
        if "VNQ" in ticker_upper or "REIT" in ticker_upper:
            return REITAsset(ticker, f"{ticker} REIT")
        elif "TLT" in ticker_upper or "TIP" in ticker_upper or "BND" in ticker_upper:
            is_tips = "TIP" in ticker_upper
            return FixedIncomeAsset(ticker, f"{ticker} Fixed Income", duration=6.0, is_tips=is_tips)
        elif "SVF" in ticker_upper:
            return SVFAsset(ticker, "Stable Value Fund")
        elif "ANNUITY" in ticker_upper:
            return AnnuityAsset(ticker, "Indexed Annuity")
        else:
            return EquityAsset(ticker, f"{ticker} Equity", beta=beta)

    assets_dict = {ticker: create_asset_by_ticker(ticker, beta=asset_betas.get(ticker, 1.0)) for ticker in tickers_list}
    portfolio_obj = Portfolio(assets=assets_dict, amounts=updated_amounts)
    set_state("current_portfolio", portfolio_obj)

    # FRED Key Input
    fred_key = st.text_input(
        "FRED API Key (Optional)",
        value=get_state("fred_api_key"),
        type="password",
        help="Input your Federal Reserve St. Louis Economic Data key to fetch live macroeconomic yields. If left blank, synthetic yields will be generated.",
    )

    # Start and End Dates
    start_date = st.date_input("Start Ingestion Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Ingestion Date", pd.to_datetime("2026-05-01"))

    # Force Cache Clear checkbox
    clear_cache = st.checkbox("Clear Cache on Download", value=False)

    # Action Button
    if st.button("Download & Ingest Dataset", width="stretch"):
        with st.spinner("Downloading data from Yahoo Finance and FRED..."):
            # Setup engines
            engine = DataIngestionEngine(fred_api_key=fred_key if fred_key else None)
            cache = DataCache()

            if clear_cache:
                cache.clear()

            # Fetch prices
            prices = engine.fetch_historical_prices(
                tickers_list,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )

            # Calculate daily returns
            returns = prices.pct_change().dropna()

            # Validate
            try:
                validated_returns = validate_returns_data(returns, min_periods=30)

                # Fetch FRED Macro
                macro_series = ["T5YIE", "T10YIE", "DGS10", "DFII10"]
                macro_df = engine.fetch_fred_macro_data(
                    macro_series,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )

                # Save to cache
                cache.save(prices, "prices")
                cache.save(validated_returns, "returns")
                cache.save(macro_df, "macro")

                # Save to session state
                set_state("tickers", tickers_list)
                set_state("fred_api_key", fred_key)
                set_state("historical_prices", prices)
                set_state("historical_returns", validated_returns)

                st.success("Dataset successfully downloaded, validated, and cached!")
            except Exception as e:
                st.error(f"Data validation error: {e}")

with col2:
    portfolio_obj = get_state("current_portfolio")
    if portfolio_obj is not None:
        st.markdown("### 💰 Current Portfolio Holdings & Weights")
        st.markdown(
            f"""
            <div class="metric-card" style="padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="metric-label">Total Portfolio Value</div>
                        <div class="metric-value" style="font-size: 1.8rem; color: #818CF8;">${portfolio_obj.total_value:,.2f}</div>
                    </div>
                    <div>
                        <div class="metric-label">Portfolio Beta</div>
                        <div class="metric-value" style="font-size: 1.8rem; color: #34D399;">{portfolio_obj.calculate_portfolio_beta():.3f}</div>
                    </div>
                    <div>
                        <div class="metric-label">Weighted FI Duration</div>
                        <div class="metric-value" style="font-size: 1.8rem; color: #F59E0B;">{portfolio_obj.calculate_portfolio_duration():.2f} years</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display table of assets, asset types, dollar amounts, and weights
        holdings_data = []
        for symbol, asset in portfolio_obj.assets.items():
            holdings_data.append({
                "Asset Symbol": symbol,
                "Asset Type": asset.asset_type,
                "Dollar Value": portfolio_obj.get_asset_amount(symbol),
                "Allocation %": portfolio_obj.get_asset_percentage(symbol)
            })
        
        df_holdings = pd.DataFrame(holdings_data)
        st.dataframe(
            df_holdings.style.format({
                "Dollar Value": "${:,.2f}",
                "Allocation %": "{:.2f}%"
            }),
            width='stretch',
            hide_index=True
        )
        st.markdown("---")

    st.markdown("### 📊 Ingested Data Quality Report")

    returns_df = get_state("historical_returns")
    prices_df = get_state("historical_prices")

    if returns_df is not None and prices_df is not None:
        st.markdown(
            f"""
            <div class="metric-card" style="padding: 15px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="metric-label">Observation Count</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{len(returns_df)} days</div>
                    </div>
                    <div>
                        <div class="metric-label">Asset Count</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{len(returns_df.columns)} assets</div>
                    </div>
                    <div>
                        <div class="metric-label">Date Range</div>
                        <div style="font-size: 1.1rem; font-weight:600; color:#E2E8F0; margin-top:5px;">
                            {returns_df.index[0].strftime("%Y-%m-%d")} to {returns_df.index[-1].strftime("%Y-%m-%d")}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display sample prices
        st.markdown("#### Adjusted Closing Prices (Recent Samples)")
        st.dataframe(prices_df.tail(), width='stretch')

        # Display asset correlations
        st.markdown("#### Return Correlations (Raw Matrix)")
        corr = returns_df.corr()
        st.dataframe(
            corr.style.background_gradient(cmap="coolwarm", axis=None),
            width='stretch',
        )

    else:
        st.warning(
            "No data ingested yet. Click **'Download & Ingest Dataset'** on the left panel to load historical financial series."
        )

        # Try to load cached data
        cache = DataCache()
        cached_returns = cache.load("returns")
        cached_prices = cache.load("prices")

        if cached_returns is not None and cached_prices is not None:
            if st.button("Load Cached Local Data", width='stretch'):
                set_state("historical_returns", cached_returns)
                set_state("historical_prices", cached_prices)
                set_state("tickers", list(cached_returns.columns))
                st.rerun()
