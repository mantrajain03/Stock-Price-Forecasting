import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src import config

class LSTMForecaster:
    def __init__(self, sequence_length=60):
        self.seq_len = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) # [cite: 101]

    def preprocess(self, data):
        """
        Normalizes data and creates sequences[cite: 102].
        """
        # Fit scaler
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.seq_len, len(scaled_data)):
            X.append(scaled_data[i-self.seq_len:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        # Reshape to [samples, timesteps, features] [cite: 103]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data

    def build_model(self, input_shape):
        """
        Constructs the LSTM architecture as defined in requirements [cite: 106-114].
        """
        model = Sequential()
        
        # Layer 1
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Layer 2
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense Layers
        model.add(Dense(25))
        model.add(Dense(1)) # Output layer
        
        # Compilation [cite: 116-118]
        model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), 
                      loss='mean_squared_error')
        
        self.model = model
        return model

    def train(self, X_train, y_train, validation_data=None):
        """
        Trains the model with Early Stopping [cite: 120-124].
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X_test):
        """
        Makes predictions and inverse transforms them [cite: 133-134].
        """
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions)

    def save_model(self, filepath):
        """Saves model in .h5 format [cite: 196]"""
        self.model.save(filepath)