# Stock Price Forecasting System: ARIMA vs. LSTM

## 📌 Project Overview
This project implements a comprehensive stock price forecasting system comparing two distinct approaches:
1.  **ARIMA (AutoRegressive Integrated Moving Average):** A statistical method for time series forecasting.
2.  **LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) capable of learning long-term dependencies.

The system performs an end-to-end workflow including data ingestion from Yahoo Finance, extensive Exploratory Data Analysis (EDA), feature engineering (technical indicators), model training, and rigorous evaluation using metrics like RMSE and MAPE.

## 📂 File Structure
```text
stock_price_forecasting/
├── data/               # Raw and processed data
├── models/             # Saved .h5 and .pkl models
├── notebooks/          # Jupyter notebooks for step-by-step analysis
├── reports/            # Generated figures and summary reports
├── src/                # Source code modules (data, features, models)
├── main.py             # Main execution script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation