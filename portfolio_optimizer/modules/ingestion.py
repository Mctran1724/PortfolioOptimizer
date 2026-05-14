# yfinance, fredapi, FinanceToolkit
"""
Data ingestion module for retrieving financial data from various sources.
Handles yfinance, FRED API, and FinanceToolkit integrations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from financetoolkit import Toolkit
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataIngestion:
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
    
    def get_stock_data(self, symbols, start_date, end_date):
        """
        Fetch stock data using yfinance
        """
        data = yf.download(symbols, start=start_date, end=end_date)
        return data['Adj Close'] if 'Adj Close' in data.columns else data
    
    def get_fred_data(self, series_ids, start_date, end_date):
        """
        Fetch economic data from FRED
        """
        if not self.fred:
            raise ValueError("FRED API key not configured")
        
        data = {}
        for series_id in series_ids:
            data[series_id] = self.fred.get_series(series_id, 
                                                  observation_start=start_date,
                                                  observation_end=end_date)
        return pd.DataFrame(data)
    
    def get_financial_ratios(self, symbols, start_date, end_date):
        """
        Fetch financial ratios using FinanceToolkit
        """
        toolkit = Toolkit(symbols, api_key=self.fred_api_key, 
                         start_date=start_date, end_date=end_date)
        ratios = toolkit.ratios.get_ratios()
        return ratios

if __name__ == "__main__":
    # Example usage
    ingestor = DataIngestion()
    # stock_data = ingestor.get_stock_data(['AAPL', 'MSFT'], '2020-01-01', '2023-12-31')
    # print(stock_data.head())