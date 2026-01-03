### 3. Final Report (`reports/summary_report.md`)
This file is the "Executive Summary" suitable for stakeholders, detailing the findings and conclusions [cite: 186-194].

```markdown
# Final Project Report: Stock Price Forecasting

## 1. Executive Summary
This project developed and compared two predictive models for forecasting the stock price of **[Insert Ticker Here]**. The goal was to evaluate whether a deep learning approach (LSTM) outperforms a traditional statistical method (ARIMA) for financial time series data.

## 2. Dataset Description
* **Source:** Yahoo Finance API
* **Range:** 2018-01-01 to 2024-01-01
* **Features:** Open, High, Low, Close, Volume
* **Engineered Features:**
    * Moving Averages (SMA 20/50/200)
    * RSI (14-day)
    * MACD
    * Bollinger Bands

## 3. Methodology

### 3.1 ARIMA (Statistical)
The time series was first transformed to be stationary. The `auto_arima` function was utilized to discover the optimal parameters. The model focused on linear relationships in past errors and lag values.

### 3.2 LSTM (Deep Learning)
A sequential neural network was designed to capture non-linear temporal dependencies. Data was normalized to the [0, 1] range. A 60-day sliding window approach was used to predict the next day's closing price.

## 4. Results and Comparison

### 4.1 Performance Metrics
*See `reports/figures/final_comparison.png` for the visual overlay.*

* **ARIMA Performance:** Demonstrated strength in following the general trend but lagged in reacting to sudden volatility.
* **LSTM Performance:** Generally captured tighter price movements but required significantly more computational time to train.

### 4.2 Limitations
* **ARIMA:** Assumes linear relationships and struggles with complex, non-linear market behaviors.
* **LSTM:** Prone to overfitting on small datasets and acts as a "black box" regarding feature importance.

## 5. Conclusion & Recommendations
Based on the MAPE (Mean Absolute Percentage Error) scores, the **[Insert Better Model]** proved more effective for this specific timeframe. However, for a production trading system, an ensemble approach combining both signals is recommended to balance trend stability (ARIMA) with pattern recognition (LSTM).