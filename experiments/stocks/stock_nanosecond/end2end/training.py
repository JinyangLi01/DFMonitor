import pandas as pd
import numpy as np
import csv

from river import compose, metrics, drift, preprocessing, forest, naive_bayes, linear_model
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1) Define a "decaying" logistic regression model
# =============================================================================
class DecayingLogisticRegressionStep(linear_model.LogisticRegression):
    def __init__(self, alpha_decay=0.99, **kwargs):
        super().__init__(**kwargs)
        self.alpha_decay = alpha_decay

    def decay_weights(self, time_delta):
        """
        Decay the model's weights based on the elapsed time.
        time_delta: Time elapsed since the last update (in nanoseconds).
        """
        # Calculate the number of 1000 ns intervals in the elapsed time
        intervals = time_delta / 1000
        # Apply exponential decay for each interval
        decay_factor = self.alpha_decay ** intervals
        for feature in self.weights:
            self.weights[feature] *= decay_factor

    def learn_one(self, x, y, time_delta=1000, sample_weight=1.0):
        """
        Learn from a new sample, applying time-based decay to the weights.
        """
        # Decay the model's weights based on the elapsed time
        self.decay_weights(time_delta)
        # Now learn from the new sample, giving it full weight
        return super().learn_one(x, y, sample_weight)


# =============================================================================
# 2) Example main code with online learning, drift detection, fairness tracking
# =============================================================================

# --- Parameters ---
time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"

# Load your data (replace with your actual file path)
data_stream = pd.read_csv(f'../../../../data/stocks_nanosecond/{data_file_name}.csv')

len_data = len(data_stream)
len_chunk = 1

# Prepare the result file for writing
result_file_name = f"prediction_result_{data_file_name}_chunk_size_{len_chunk}_v3.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')

csv_writer.writerow([
    "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action",
    "side", "price", "size", "channel_id", "order_id", "flags", "ts_in_delta",
    "sequence", "symbol", "sector", "true_direction", "predicted_direction",
    "correct_prediction", "sector_representation", "fairness_index"
])

# ------------------- #
#    Initialize Models
# ------------------- #
# You can still keep a dictionary for other models, but let's highlight
# the decaying logistic regression for demonstration:
models = {
    # 'ARFClassifier': (
    #     preprocessing.OneHotEncoder() |
    #     preprocessing.StandardScaler() |
    #     forest.ARFClassifier(n_models=10, seed=42)
    # ),
    # 'NaiveBayes': (
    #     preprocessing.OneHotEncoder() |
    #     preprocessing.StandardScaler() |
    #     naive_bayes.GaussianNB()
    # ),
    'DecayingLogisticRegression': (
        preprocessing.OneHotEncoder() |
        preprocessing.StandardScaler() |
        DecayingLogisticRegression(alpha_decay=0.995)  # <== Exponential decay
    )
}

# Choose the decaying logistic regression model
model_name = 'DecayingLogisticRegression'
model = models[model_name]

# Initialize metrics
metric = metrics.Accuracy()
drift_detector = drift.ADWIN()

# Initialize sector representation and accuracy tracking
sector_weights = {sector: 0 for sector in data_stream['sector'].unique()}
sector_accuracy = {sector: metrics.Accuracy() for sector in data_stream['sector'].unique()}

# Time decay factor for "sector representation" tracking (separate from model decay!)
alpha = 0.99

def update_sector_weights(row):
    """
    Update the representation weight for the sector of the given row.
    The sector that just arrived gets a boost of (1 - alpha),
    while all previous sectors are multiplied by alpha (decay).
    """
    sector = row['sector']
    global sector_weights
    for s in sector_weights:
        sector_weights[s] *= alpha  # Decay previous weights
    sector_weights[sector] += (1 - alpha)  # Increment for current sector

def calculate_fairness():
    """
    Calculates a simple fairness index:
      1 - ((max_sector_accuracy - min_sector_accuracy) / max_sector_accuracy)
    If no sector has any samples yet, returns 1 by default.
    """
    accuracies = [
        sector_accuracy[s].get()
        for s in sector_weights.keys()
        if sector_accuracy[s].n > 0
    ]
    if len(accuracies) > 1:
        return 1 - (max(accuracies) - min(accuracies)) / max(accuracies)
    return 1.0  # Perfect fairness if all sectors are equally accurate or no data

def preprocess_row(row):
    """
    Convert row into a dict of features suitable for the model.
    Excludes non-feature columns.
    """
    x = row.drop([
        'next_price_direction', 'ts_recv', 'ts_event', 'ts_event_datetime',
        'publisher_id', 'instrument_id', 'channel_id', 'order_id'
    ]).to_dict()
    return x

# ------------------- #
#   Feature Engineering
# ------------------- #
data_stream['delta_price'] = data_stream['price'].diff().fillna(0)
data_stream['rolling_mean_price'] = data_stream['price'].rolling(window=5, min_periods=1).mean()
data_stream['rolling_mean_size'] = data_stream['size'].rolling(window=5, min_periods=1).mean()
data_stream['price_size_interaction'] = data_stream['price'] * data_stream['rolling_mean_size']
data_stream['price_volatility'] = data_stream['price'].rolling(window=5, min_periods=1).std().fillna(0)
data_stream['price_momentum'] = data_stream['price'].diff(3).fillna(0)

# Normalize engineered features
scaler = MinMaxScaler()
data_stream[['delta_price', 'rolling_mean_price', 'rolling_mean_size',
             'price_volatility', 'price_momentum']] = scaler.fit_transform(
    data_stream[['delta_price', 'rolling_mean_price', 'rolling_mean_size',
                 'price_volatility', 'price_momentum']]
)

print(f"Total number of data points: {len_data}")

# ------------------- #
#   Streaming Loop
# ------------------- #
batch_results = []
prev_ts_event = None
for idx, row in data_stream.iterrows():
    # Preprocess row and extract target
    x = preprocess_row(row)
    y = row['next_price_direction']
    sector = row['sector']
    ts_event = row['ts_event']
    # Calculate time_delta (elapsed time since the last event)
    if prev_ts_event is not None:
        time_delta = ts_event - prev_ts_event
    else:
        time_delta = 1000  # Default value for the first sample
        # Update the previous timestamp
    prev_ts_event = ts_event
    # Update sector representation weights
    update_sector_weights(row)

    # Predict (using the model state from previous samples)
    y_pred = model.predict_one(x)

    # Learn from this new sample
    # (the decay is automatically handled inside DecayingLogisticRegression)
    model.learn_one(x, y, time_delta)

    # Update sector accuracy
    if y_pred is not None:
        sector_accuracy[sector].update(y, y_pred)

    # Calculate fairness index
    fairness_index = calculate_fairness()

    # Drift detection:
    #   We feed ADWIN the absolute error for classification (0 or 1).
    #   If drift is detected, we reset the model.
    if drift_detector.update(abs(y - (y_pred if y_pred is not None else 0))):
        print(f"Drift detected at index {idx}, resetting model...")
        model = models[model_name]
        drift_detector = drift.ADWIN()

    # Collect results for batch writing
    batch_results.append([
        row['ts_recv'], row['ts_event'], row['rtype'], row['publisher_id'],
        row['instrument_id'], row['action'], row['side'], row['price'],
        row['size'], row['channel_id'], row['order_id'], row['flags'],
        row['ts_in_delta'], row['sequence'], row['symbol'], row['sector'],
        y, y_pred, int(y == y_pred), sector_weights[sector], fairness_index
    ])

    # Write results in batches of 100
    if len(batch_results) >= 100:
        csv_writer.writerows(batch_results)
        batch_results = []
        result_file.flush()

    # Update and log overall accuracy
    if y_pred is not None:
        metric.update(y, y_pred)
    if idx % 100 == 0 and idx > 0:
        print(f"Accuracy after {idx} observations: {metric.get():.4f}")
        print(f"Fairness Index after {idx} observations: {fairness_index:.4f}")

# Write any remaining results to the file
if batch_results:
    csv_writer.writerows(batch_results)
    batch_results = []
    result_file.flush()

# Final metrics
print(f"Final Accuracy: {metric.get():.4f}")
print(f"Final Fairness Index: {calculate_fairness():.4f}")
result_file.close()
