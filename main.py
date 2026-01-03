import pandas as pd
import numpy as np
import os
from src import config, data_loader, features, arima_model, lstm_model, evaluation, visualization

def main():
    print("=== Starting Stock Price Forecasting System ===")
    
    # 1. Data Collection [cite: 37]
    df = data_loader.download_data(config.TICKER, config.START_DATE, config.END_DATE)
    
    # 2. Feature Engineering [cite: 55]
    df_features = features.add_technical_indicators(df)
    
    # Use 'Close' price for forecasting
    data = df_features[['Close']]
    
    # Split Data [cite: 83]
    train_size = int(len(data) * config.TRAIN_SPLIT)
    train_data, test_data = data[:train_size], data[train_size:]
    
    print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

    # ==========================
    # 3. ARIMA Implementation [cite: 71]
    # ==========================
    print("\n--- Starting ARIMA Model ---")
    arima = arima_model.ArimaForecaster()
    
    # Train
    arima.train(train_data['Close'])
    
    # Predict
    arima_pred, conf_int = arima.predict(steps=len(test_data))
    arima_pred = pd.Series(arima_pred.values, index=test_data.index)
    
    # Evaluate ARIMA
    arima_metrics = evaluation.calculate_metrics(test_data['Close'], arima_pred, "ARIMA")
    
    # Save ARIMA
    arima.save_model(os.path.join(config.MODELS_DIR, 'arima_model.pkl'))

    # ==========================
    # 4. LSTM Implementation [cite: 98]
    # ==========================
    print("\n--- Starting LSTM Model ---")
    lstm = lstm_model.LSTMForecaster(sequence_length=config.SEQ_LEN)
    
    # Prepare data (LSTM requires sequences, so we need some train data for the first test prediction)
    # We pass the WHOLE dataset to preprocess, then split manually to preserve order for testing
    X, y, scaled_data = lstm.preprocess(data)
    
    # LSTM Split (Adjusting for sequence length which reduces total samples)
    # The 'y' array aligns with the target day.
    # If len(data) = 1000, len(y) = 940 (if seq=60). 
    # We split based on the index relative to the original split point.
    
    split_idx = int(len(X) * config.TRAIN_SPLIT)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Build & Train
    lstm.build_model(input_shape=(X_train.shape[1], 1))
    lstm.train(X_train, y_train, validation_data=(X_test, y_test))
    
    # Predict
    lstm_pred = lstm.predict(X_test)
    
    # LSTM Evaluation (Align dimensions)
    # The test_data index needs to be aligned with LSTM predictions
    # LSTM predictions start at split_idx + seq_len relative to start of data
    # For simplicity in this demo, we compare LSTM pred against y_test (inverse transformed)
    
    y_test_inv = lstm.scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_metrics = evaluation.calculate_metrics(y_test_inv, lstm_pred, "LSTM")
    
    # Save LSTM
    lstm.save_model(os.path.join(config.MODELS_DIR, 'lstm_model.h5'))

    # ==========================
    # 5. Results & Visualization [cite: 161]
    # ==========================
    
    # Align indices for plotting
    # ARIMA predicts exactly the test_data range.
    # LSTM predicts the range covered by X_test.
    
    # Visualize
    # We pass the slice of test data that corresponds to LSTM predictions for fair visual
    visualization.plot_results(
        train_data['Close'], 
        test_data['Close'].iloc[-len(lstm_pred):], 
        arima_pred[-len(lstm_pred):], 
        lstm_pred.flatten(),
        title="ARIMA vs LSTM Comparison"
    )
    
    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()