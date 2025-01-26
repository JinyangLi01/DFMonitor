import pandas as pd
from river import compose, linear_model, preprocessing, feature_extraction
from river import metrics
from river.feature_extraction import TargetAgg
import numpy as np
import csv


# Load your data (replace 'your_data.csv' with your actual file)
# The file should contain columns such as 'open', 'high', 'low', 'close', 'volume', 'vwap', etc.
time_period = "8-10_ny_time"
data_file_name = f"s&p50_all_stocks_2024-09-18_sorted_with_binary_direction_{time_period}"
data_df = pd.read_csv(f'../../../../data/stocks_online/{data_file_name}.csv')
len_data = len(data_df)
len_chunk = 100
data_stream = pd.read_csv(f'../../../../data/stocks_online/{data_file_name}.csv', iterator=True, chunksize=len_chunk)
num_chunks = len_data // len_chunk

result_file_name = f"update_feature_selection/prediction_result_{data_file_name}_chunk_size_{len_chunk}.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')

csv_writer.writerow(["timestamp","symbol","sector","open","high","low","close","volume","vwap","transactions","datetime","binary_direction","predicted_direction"])

# Initialize model with preprocessing for numerical and categorical features
model = (
preprocessing.OneHotEncoder() |  # One-hot encoding for categorical features
    preprocessing.StandardScaler() |  # Standardize numerical features
    linear_model.LogisticRegression()  # Logistic regression model
)

print(f"chunk size: {len_chunk}, num of chunks: {num_chunks}")

# Define metric to evaluate the model's performance
metric = metrics.Accuracy()
# Stream learning - Iterate over data stream
chunk_id = 0
for chunk in data_stream:
    print(f"chunk id {chunk_id}\n")
    chunk_id += 1
    for index, row in chunk.iterrows():
        # Prepare the feature set
        x = row.drop(['binary_direction', "close"]).to_dict()  # Convert row to dictionary and remove non-feature columns
        y = row['binary_direction']  # Target variable

        # # Handle missing values
        # if pd.isnull(y) or pd.isnull(x).any():
        #     continue  # Skip this iteration if there are missing values

        # Predict and update model incrementally
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        csv_writer.writerow([row["timestamp"], row["symbol"], row["sector"], row["open"], row["high"], row["low"], row["close"], row["volume"], row["vwap"], row["transactions"], row["datetime"], row["datetime_ny"], y, y_pred])

        # Evaluate
        if y_pred is not None:
            metric.update(y, y_pred)
        # print(f"Accuracy: {metric.get()}")

result_file.close()