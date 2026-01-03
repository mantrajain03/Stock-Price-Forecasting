import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config
import os

def plot_results(train, test, arima_pred, lstm_pred, title="Model Comparison"):
    """
    Plots Actual vs ARIMA vs LSTM.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot training data
    plt.plot(train.index, train.values, label='Training Data', color='gray', alpha=0.5)
    
    # Plot test data (Actual)
    plt.plot(test.index, test.values, label='Actual Price', color='blue')
    
    # Plot ARIMA
    plt.plot(test.index, arima_pred, label='ARIMA Predictions', color='red', linestyle='--')
    
    # Plot LSTM
    # Note: LSTM might have length mismatch due to sequence window, handle indexing carefully
    lstm_idx = test.index[:len(lstm_pred)]
    plt.plot(lstm_idx, lstm_pred, label='LSTM Predictions', color='green', linestyle='--')
    
    plt.title(f'{config.TICKER} Stock Price Prediction: {title}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(config.REPORTS_DIR, 'final_comparison.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()