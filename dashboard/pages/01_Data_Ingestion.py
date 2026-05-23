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
    if st.button("Download & Ingest Dataset", use_container_width=True):
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
        st.dataframe(prices_df.tail(), use_container_width=True)

        # Display asset correlations
        st.markdown("#### Return Correlations (Raw Matrix)")
        corr = returns_df.corr()
        st.dataframe(
            corr.style.background_gradient(cmap="coolwarm", axis=None),
            use_container_width=True,
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
            if st.button("Load Cached Local Data", use_container_width=True):
                set_state("historical_returns", cached_returns)
                set_state("historical_prices", cached_prices)
                set_state("tickers", list(cached_returns.columns))
                st.rerun()
