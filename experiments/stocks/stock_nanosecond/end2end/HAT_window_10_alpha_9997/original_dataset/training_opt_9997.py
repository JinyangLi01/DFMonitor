from river import linear_model
from river import preprocessing
from river import metrics
from river import datasets, drift
import math
from sklearn.preprocessing import MinMaxScaler
import csv
import pandas as pd
from river import tree

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


# Load your data (replace 'your_data.csv' with your actual file)
time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
data_stream = pd.read_csv(f'original_dataset.csv')



time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')


data_stream["ts_event"] = pd.to_datetime(data_stream["ts_event"])
# get data between time_start and time_end
data_stream = data_stream[(data_stream["ts_event"] >= time_start) & (data_stream["ts_event"] <= time_end)]



len_data = len(data_stream)
len_chunk = 1

# Function to preprocess a row
def preprocess_row(r):
    x = r.drop(['next_price_direction', 'ts_recv', date_column, 'ts_event_datetime',
                  "publisher_id", "instrument_id", "channel_id", "order_id"]).to_dict()
    return x

# Feature engineering
data_stream['delta_price'] = data_stream['price'].diff().fillna(0)
data_stream['rolling_mean_price'] = data_stream['price'].rolling(window=5, min_periods=1).mean()
data_stream['rolling_mean_size'] = data_stream['size'].rolling(window=5, min_periods=1).mean()
data_stream['price_volatility'] = data_stream['price'].rolling(window=5, min_periods=1).std().fillna(0)
data_stream['price_momentum'] = data_stream['price'].diff(3).fillna(0)
data_stream['price_change_direction'] = data_stream['delta_price'].apply(lambda x: 1 if x > 0 else 0)
data_stream['volume_weighted_price'] = data_stream['price'] * data_stream['size']



data_stream.reset_index(drop=True, inplace=True)

print(f"Total number of data points: {len_data}")


# Initialize the Hoeffding Adaptive Tree
model = tree.HoeffdingAdaptiveTreeClassifier()


use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True
window_size_nanoseconds = int(time_unit.split(' ')[0]) * int(window_size_units)


# Initialize a metric to track performance
metric = metrics.Accuracy()
drift_detector = drift.ADWIN()

# Decay rate (lambda)
decay_rate = 0.9997
date_column = 'ts_event'

time_delta = 0
batch_results = []
data_stream[date_column] = pd.to_datetime(data_stream[date_column])
previous_event_timestamp = data_stream[date_column].iloc[0]


# Prepare the result file for writing
result_file_name = f"prediction_result_end2end_HAT_window_{window_size_units}_remove_stocks_w_expo_decay.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')

csv_writer.writerow(["ts_recv","ts_event","rtype","publisher_id","instrument_id","action","side","price","size",
                     "channel_id","order_id","flags","ts_in_delta","sequence","symbol","sector",
                     "next_price_direction","predicted_direction","prediction_binary_correctness"])

# Online learning loop
for idx, row in data_stream.iterrows():
    x = preprocess_row(row)
    y = row['next_price_direction']
    y_pred = model.predict_one(x)
    model.learn_one(x, y)

    if idx < 1000:
        continue
    # Update metric and record results
    if y_pred is not None:
        metric.update(y, y_pred)
        batch_results.append(
            [row['ts_recv'], row[date_column], row['rtype'], row['publisher_id'], row['instrument_id'],
             row['action'], row['side'], row['price'], row['size'], row['channel_id'],
             row['order_id'], row['flags'], row['ts_in_delta'], row['sequence'],
             row['symbol'], row['sector'], y, y_pred, int(y == y_pred)])
    else:
        print(f"Warning: y_pred is None at index {idx}")

    # Write results in batches of 100
    if len(batch_results) >= 1000:
        csv_writer.writerows(batch_results)
        batch_results = []
        # result_file.flush()
    #
    # #
    # if drift_detector.update(abs(y - y_pred)):
    #     print("Drift detected! Resetting model...")
    #     model = tree.HoeffdingAdaptiveTreeClassifier(       )
    #
    # Print progress
    if idx % 100 == 0:
        print(f"Step {idx}, Accuracy: {metric.get():.4f}")





# Write any remaining results in the batch buffer
if batch_results:
    csv_writer.writerows(batch_results)
    result_file.flush()

# Final accuracy
print(f"Final Metric Accuracy: {metric.get():.4f}")
# Verify manual accuracy
df_results = pd.read_csv(result_file_name)
manual_accuracy = df_results['prediction_binary_correctness'].mean()
print(f"Manual Accuracy: {manual_accuracy:.4f}")


result_file.close()


