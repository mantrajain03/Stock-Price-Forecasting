import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculates MAE, MSE, RMSE, MAPE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE calculation [cite: 35, 152]
    # Handle division by zero
    y_true_safe = np.where(y_true == 0, 1e-6, y_true) 
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Performance ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R2:   {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}