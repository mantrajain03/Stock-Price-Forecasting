# Project Report: Stock Price Forecasting Analysis

## 1. Executive Summary
This project compared the performance of statistical (ARIMA) and Deep Learning (LSTM) models in predicting Apple Inc. (AAPL) stock prices. The study utilized 5 years of historical daily data. The Long Short-Term Memory (LSTM) network demonstrated superior capability in capturing non-linear market patterns compared to the linear ARIMA model.

## 2. Methodology
* **Data Source:** Yahoo Finance API (Daily OHLCV data).
* **Feature Engineering:**
    * Technical Indicators: RSI (14-day), MACD, Bollinger Bands, and Moving Averages (20, 50, 200).
    * Preprocessing: Data was normalized to [0, 1] using MinMaxScaler for the neural network.
* **Models:**
    * **ARIMA:** Parameters selected via grid search (Auto-ARIMA) based on AIC scores.
    * **LSTM:** 2-layer sequential architecture (50 units) with Dropout (0.2) to prevent overfitting.

## 3. Results and Performance
The models were evaluated on the final 20% of the dataset (Test Set).

| Metric | LSTM Model | ARIMA Model | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAPE** | **2.19%** | High (>10%) | LSTM error is exceptionally low (<5%). |
| **R² Score** | **0.9286** | ~0.0 | LSTM explains 92.8% of variance; ARIMA failed to capture trend. |
| **RMSE** | **4.62** | High | LSTM predictions are much closer to actual price. |

### Key Observations
1.  **Trend Capture:** The LSTM model successfully identified and adapted to the uptrend in 2023, while ARIMA largely predicted a flat trajectory (mean reversion).
2.  **Volatility:** The LSTM model reacted well to short-term volatility, closely tracking the "noise" of daily price movements.
3.  **Stationarity:** The ARIMA model struggled because stock prices are inherently non-stationary, and differencing removed the trend information required for long-term forecasting.

## 4. Conclusion
The **LSTM model is the recommended approach** for this specific forecasting task. With a Mean Absolute Percentage Error (MAPE) of just **2.19%**, it meets the criteria for "Excellent" accuracy defined in the project success metrics.

## 5. Future Work
To further improve performance:
* Integrate **Sentiment Analysis** from financial news to predict sudden price shocks.
* Implement **Hyperparameter Tuning** (Keras Tuner) to optimize the LSTM layer count and learning rate.