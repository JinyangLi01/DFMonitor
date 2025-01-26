import pandas as pd
import numpy as np
import joblib
from river import compose, linear_model, preprocessing, metrics, drift, ensemble, tree, optim
from collections import deque
import csv
import numba

# Load your data (replace 'your_data.csv' with your actual file)
time_period = "15-00--15-01"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
data_stream = pd.read_csv(f'../../../../data/stocks_nanosecond/{data_file_name}.csv')

len_data = len(data_stream)
len_chunk = 1
#
# ts_recv,ts_event,rtype,publisher_id,instrument_id,action,side,price,size,channel_id,order_id,flags,ts_in_delta,sequence,symbol,ts_event_datetime

# Prepare the result file for writing
result_file_name = f"prediction_result_{data_file_name}_chunk_size_{len_chunk}_v1.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')

csv_writer.writerow(["ts_recv","ts_event","rtype","publisher_id","instrument_id","action","side","price","size",
                     "channel_id","order_id","flags","ts_in_delta","sequence","symbol","sector","ts_event_datetime",
                     "next_price_direction","predicted_direction","prediction_binary_correctness"])


# Initialize the model and preprocessing steps
model = (
    preprocessing.OneHotEncoder() |  # One-hot encoding for categorical features
    preprocessing.StandardScaler() |  # Standardize numerical features
    # linear_model.LogisticRegression(optimizer=optim.SGD(), l2=0.01)  # Logistic regression model with hyperparameter tuning
tree.HoeffdingTreeClassifier()  # Hoeffding Tree classifier
)

# model = ensemble.BaggingClassifier(model=base_model, n_models=5, seed=42)  # Reduce number of models for speed

# Initialize accuracy metric for evaluation
metric = metrics.Accuracy()

# Drift detector initialization
drift_detector = drift.ADWIN()

# Buffer for batch writing
batch_results = []

# Optimized function for RSI using numba
@numba.jit(nopython=True)
def calculate_rsi(prev_closes, period=14):
    valid_closes = prev_closes[~np.isnan(prev_closes)]
    if len(valid_closes) < period:
        return 50  # Default RSI value if not enough data
    gains = np.maximum(0, np.diff(valid_closes))
    losses = np.maximum(0, -np.diff(valid_closes))
    average_gain = np.sum(gains) / period if len(gains) > 0 else 0
    average_loss = np.sum(losses) / period if len(losses) > 0 else 0
    if average_loss == 0:
        return 100
    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))

# Function to calculate Bollinger Bands
@numba.jit(nopython=True)
def calculate_bollinger_bands(prev_closes, period=20):
    valid_closes = prev_closes[~np.isnan(prev_closes)]
    if len(valid_closes) < period:
        return valid_closes[-1] if len(valid_closes) > 0 else 0, 0, 0
    sma = np.mean(valid_closes[-period:])
    std = np.std(valid_closes[-period:])
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return sma, upper_band, lower_band

# Function to calculate MACD
@numba.jit(nopython=True)
def calculate_macd(prev_closes, short_period=12, long_period=26, signal_period=9):
    valid_closes = prev_closes[~np.isnan(prev_closes)]
    if len(valid_closes) < long_period:
        return 0, 0
    ema_short = np.mean(valid_closes[-short_period:])
    ema_long = np.mean(valid_closes[-long_period:])
    macd = ema_short - ema_long
    signal_line = np.mean(valid_closes[-signal_period:])
    return macd, signal_line


# Function to preprocess a row
def preprocess_row(row):
    x = row.drop(['next_price_direction', 'ts_recv', 'ts_event', 'ts_event_datetime']).to_dict()
    return x

print(f"Total number of data points: {len_data}")

# Stream learning loop
for idx, row in data_stream.iterrows():
    x = preprocess_row(row)
    y = row['next_price_direction']

    # Predict and learn
    y_pred = model.predict_one(x)
    model.learn_one(x, y)

    # # Drift detection every 100 iterations
    # if idx % 100 == 0 and drift_detector.update(y_pred != y):
    #     print("Drift detected! Resetting the model.")
    #     model = (
    #             preprocessing.OneHotEncoder() |  # One-hot encoding for categorical features
    #             preprocessing.StandardScaler() |  # Standardize numerical features
    #             # linear_model.LogisticRegression(optimizer=optim.SGD(), l2=0.01)  # Logistic regression model with hyperparameter tuning
    #             tree.HoeffdingTreeClassifier()  # Hoeffding Tree classifier
    #     )


    batch_results.append([row['ts_recv'], row['ts_event'], row['rtype'], row['publisher_id'], row['instrument_id'],
                          row['action'], row['side'], row['price'], row['size'], row['channel_id'],
                          row['order_id'], row['flags'], row['ts_in_delta'], row['sequence'],
                          row['symbol'], row['sector'], y, y_pred, int(y == y_pred)])

    # Write results in batches of 100
    if len(batch_results) >= 100:
        csv_writer.writerows(batch_results)
        batch_results = []
        result_file.flush()

    # Update metric
    if y_pred is not None:
        metric.update(y, y_pred)

    # Print accuracy every 100 rows
    if idx % 100 == 0:
        print(f"Accuracy after {idx} observations: {metric.get()}")

# Write any remaining results in the batch buffer
if batch_results:
    csv_writer.writerows(batch_results)
    result_file.flush()

# Final accuracy
print(f"Final Accuracy: {metric.get()}")
result_file.close()
