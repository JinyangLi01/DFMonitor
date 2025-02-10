import pandas as pd
import numpy as np
import csv
from river import compose, metrics, drift, preprocessing, forest, naive_bayes, linear_model
from sklearn.preprocessing import MinMaxScaler

# Load your data (replace 'your_data.csv' with your actual file)
time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
data_stream = pd.read_csv(f'../../../../data/stocks_nanosecond/{data_file_name}.csv')

len_data = len(data_stream)
len_chunk = 1



time_start = pd.Timestamp('2024-10-15 14:00:03.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')


data_stream["ts_event"] = pd.to_datetime(data_stream["ts_event"])
# get data between time_start and time_end
data_stream = data_stream[(data_stream["ts_event"] >= time_start) & (data_stream["ts_event"] <= time_end)]






# Prepare the result file for writing
result_file_name = f"prediction_result_{data_file_name}_chunk_size_{len_chunk}_v3.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')

csv_writer.writerow(["ts_recv","ts_event","rtype","publisher_id","instrument_id","action","side","price","size",
                     "channel_id","order_id","flags","ts_in_delta","sequence","symbol","sector","ts_event_datetime",
                     "next_price_direction","predicted_direction","prediction_binary_correctness"])

# Initialize multiple models for comparison
models = {
    'ARFClassifier': (
        preprocessing.OneHotEncoder() |
        preprocessing.StandardScaler() |
        forest.ARFClassifier(n_models=10, seed=42)
    ),
    'NaiveBayes': (
        preprocessing.OneHotEncoder() |
        preprocessing.StandardScaler() |
        naive_bayes.GaussianNB()
    ),
    'LogisticRegression': (
        preprocessing.OneHotEncoder() |
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
    )
}

# Choose a model
model_name = 'LogisticRegression'  # Change this to try different models
model = models[model_name]

# Initialize accuracy metric for evaluation
metric = metrics.Accuracy()

# Drift detector initialization
drift_detector = drift.ADWIN()

# Buffer for batch writing
batch_results = []

# Function to preprocess a row
def preprocess_row(r):
    x = r.drop(['next_price_direction', 'ts_recv', 'ts_event', 'ts_event_datetime',
                  "publisher_id", "instrument_id", "channel_id", "order_id"]).to_dict()
    return x

# Feature engineering: Add price change, moving average, volatility
data_stream['delta_price'] = data_stream['price'].diff().fillna(0)
data_stream['rolling_mean_price'] = data_stream['price'].rolling(window=5, min_periods=1).mean()
data_stream['rolling_mean_size'] = data_stream['size'].rolling(window=5, min_periods=1).mean()
data_stream['price_size_interaction'] = data_stream['price'] * data_stream['rolling_mean_size']
data_stream['price_volatility'] = data_stream['price'].rolling(window=5, min_periods=1).std().fillna(0)
data_stream['price_momentum'] = data_stream['price'].diff(3).fillna(0)

# Normalize additional features
scaler = MinMaxScaler()
data_stream[['delta_price', 'rolling_mean_price', 'rolling_mean_size', 'price_volatility', 'price_momentum']] = scaler.fit_transform(
    data_stream[['delta_price', 'rolling_mean_price', 'rolling_mean_size', 'price_volatility', 'price_momentum']]
)

print(f"Total number of data points: {len_data}")

# Stream learning loop
for idx, row in data_stream.iterrows():
    x = preprocess_row(row)
    y = row['next_price_direction']

    # Skip prediction until seeing 1k data points
    if idx < 1000:
        model.learn_one(x, y)
        continue

    # Predict and learn
    y_pred = model.predict_one(x)

    # Drift detection
    if drift_detector.update(abs(y - (y_pred if y_pred is not None else 0))):
        print(f"Drift detected at index {idx}, resetting model...")
        model = models[model_name]
        drift_detector = drift.ADWIN()

    model.learn_one(x, y)

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
