import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Data Parameters
TICKER = 'AAPL'  # [cite: 39]
START_DATE = '2018-01-01'
END_DATE = '2024-01-01'

# Feature Engineering
MOVING_AVERAGES = [20, 50, 200]  # [cite: 58]
EMA_SPANS = [12, 26]             # [cite: 59]
RSI_PERIOD = 14                  # [cite: 60]

# LSTM Hyperparameters
SEQ_LEN = 60        # Previous N days to predict next day [cite: 102]
LSTM_UNITS = 50     # [cite: 108]
DROPOUT = 0.2       # [cite: 109]
EPOCHS = 50         # [cite: 120]
BATCH_SIZE = 32     # [cite: 121]
LEARNING_RATE = 0.001 # [cite: 116]

# Splitting
TRAIN_SPLIT = 0.8   # [cite: 83, 104]