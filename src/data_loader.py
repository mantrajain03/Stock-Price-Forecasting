import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    """
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure no missing values [cite: 41]
    df = df.ffill().dropna()
    
    # Flatten multi-index columns if present (yfinance update fix)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    print(f"Data shape: {df.shape}")
    return df

def save_data(df, filename):
    df.to_csv(filename)