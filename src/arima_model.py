import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
from src import config
import matplotlib.pyplot as plt

class ArimaForecaster:
    def __init__(self):
        self.model = None
        self.best_params = None

    def find_optimal_params(self, train_data):
        """
        Uses auto_arima to find optimal p, d, q parameters[cite: 80].
        """
        print("Running Auto ARIMA to find optimal parameters...")
        model = pm.auto_arima(train_data, 
                              start_p=1, start_q=1,
                              max_p=5, max_q=5, # [cite: 86, 88]
                              m=1,              # Non-seasonal
                              seasonal=False,
                              d=None,           # Let model determine differencing [cite: 87]
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)
        
        self.best_params = model.order
        print(f"Best ARIMA Order: {self.best_params}")
        return self.best_params

    def train(self, train_data, order=None):
        """
        Fits the ARIMA model on training data[cite: 84].
        """
        if order is None:
            if self.best_params is None:
                self.find_optimal_params(train_data)
            order = self.best_params
            
        print(f"Training ARIMA model with order {order}...")
        self.model = SARIMAX(train_data, order=order)
        self.model_fit = self.model.fit(disp=False)
        
        # Diagnostic Check [cite: 94]
        self.model_fit.plot_diagnostics(figsize=(10, 8))
        plt.savefig(os.path.join(config.REPORTS_DIR, 'arima_diagnostics.png'))
        plt.close()

    def predict(self, steps):
        """
        Forecasts future steps[cite: 91].
        """
        forecast_result = self.model_fit.get_forecast(steps=steps)
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int() # [cite: 92]
        return predictions, conf_int

    def save_model(self, filepath):
        """Saves model using pickle/joblib [cite: 197]"""
        joblib.dump(self.model_fit, filepath)