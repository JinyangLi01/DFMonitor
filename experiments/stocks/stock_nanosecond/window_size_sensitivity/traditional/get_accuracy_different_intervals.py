from datetime import datetime

#import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anyio.lowlevel import checkpoint

from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys
import colormaps as cmaps
import math
from pandas import Timestamp, date_range, Timedelta



def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)




time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
len_chunk = 1
# Prepare the result file for writing
data_file_name = f"../../predict_results/prediction_result_{data_file_name}_chunk_size_{len_chunk}_v3.csv"
data = pd.read_csv(data_file_name)


sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services']

data = data[data["sector"].isin(sector_list)]

time_start = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')

data["ts_event"] = pd.to_datetime(data["ts_event"])
# get data between time_start and time_end
data = data[(data["ts_event"] >= time_start) & (data["ts_event"] <= time_end)]

# reset the index
data = data.reset_index(drop=True)
print("len of selected data", len(data))


#
#
# threshold = 0.4
# label_prediction = "predicted_direction"
# label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
# use_two_counters = True
# time_unit = "100000 nanosecond"
# window_size_units = "1"
# checking_interval = "200000 nanosecond"
# use_nanosecond = True



# Define the time window dynamically
time_window_str = "300ms"
time_window = pd.to_timedelta(time_window_str)  # Convert to Timedelta

# Generate time windows dynamically based on time_window and time range
time_windows = []
current_start = time_start
while current_start < time_end:
    current_end = min(current_start + time_window, time_end)
    time_windows.append((current_start, current_end))
    current_start = current_end  # Move to the next window

# Data processing to calculate accuracy for each dynamic time window
accuracy_results = []
for start, end in time_windows:
    # Filter data within each window
    window_data = data[(data["ts_event"] >= start) & (data["ts_event"] < end)]
    if not window_data.empty:
        # Calculate accuracy per sector within the window
        accuracy_per_sector = window_data.groupby("sector")[correctness_column].mean()
        # Store the result at the end of the window
        accuracy_results.extend([
            {"ts_event": end, "sector": sector, "accuracy": accuracy}
            for sector, accuracy in accuracy_per_sector.items()
        ])

print("accuracy_results", accuracy_results)

# Convert accuracy results to a DataFrame
accuracy_df = pd.DataFrame(accuracy_results)

accuracy_df.to_csv(f"traditional_accuracy_time_window_{time_window_str}.csv")

# Plot the data
fig, ax = plt.subplots()

# Plot accuracy for each sector at the end of each dynamic time window
for sector, sector_data in accuracy_df.groupby("sector"):
    ax.plot(sector_data["ts_event"], sector_data["accuracy"], marker='o', label=sector)

# Set x-axis limits and formatting
ax.set_xlim(time_start, time_end)
ax.set_xticks([time_start] + [end for _, end in time_windows])  # Dynamic x-ticks at each window end
ax.set_xticklabels([time.strftime('%H:%M:%S') for time in [time_start] + [end for _, end in time_windows]])

# Add labels and title
ax.set_xlabel("Time")
ax.set_ylabel("Accuracy")
ax.set_title(f"Accuracy per {time_window_str} Interval")
ax.legend(title="Sector")

plt.tight_layout()
plt.show()