import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds SMA, EMA, RSI, MACD, Bollinger Bands, and Returns.
    """
    data = df.copy()
    
    # Simple Moving Averages [cite: 58]
    for window in [20, 50, 200]:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()

    # Exponential Moving Averages [cite: 59]
    for span in [12, 26]:
        data[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()

    # RSI (Relative Strength Index) [cite: 60]
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD [cite: 61]
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands [cite: 62]
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    
    # Daily Returns [cite: 63]
    data['Returns'] = data['Close'].pct_change()
    
    # Drop NaN created by indicators (e.g., 200-day SMA)
    return data.dropna()