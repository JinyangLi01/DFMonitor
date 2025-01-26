import colorsys
import csv
import math

import pandas as pd
import matplotlib.pyplot as plt

from algorithm.per_item import Accuracy_workload as fixed_workload

from algorithm.dynamic_window import Accuracy_workload as dynamic_workload

import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


method_name = "logistic_regression"
exmained_time = "14-00--14-10"
chunk_size=1
date = "20241015"
data = pd.read_csv(f"../1h_interval/prediction_result_xnas-itch-{date}_{exmained_time}_chunk_size_{chunk_size}_v3.csv")

print("len of original data", len(data))
print(data["sector"].unique())
date_column = "ts_event"
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True


# Define the New York time zone
# ny_tz = pytz.timezone('America/New_York')

time_start = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')

# get data between time_start and time_end
data = data[(data["ts_event"] >= time_start) & (data["ts_event"] <= time_end)]

# reset the index
data = data.reset_index(drop=True)
print("len of selected data", len(data))

print(data[:5])
alpha = 0.99995
threshold = 0.3


label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
use_nanosecond = True
time_unit = "10000 nanosecond"
checking_interval = "200000 nanosecond"
window_size_units = "1"
num_window_reset = ((data["ts_event"].max() - data["ts_event"].min()).total_seconds() * 1e9 /
                    int(checking_interval.split(" ")[0]))

print(f"num of window resets: {num_window_reset}")

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Financial Services'}, {"sector": 'Consumer Defensive'}, {"sector": 'Healthcare'},
                    {"sector": 'Energy'}]



print("monitored_groups: ", monitored_groups)
sectors_list = [x["sector"] for x in monitored_groups]




###############################################################################################


DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
            window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
            time_new_window_bit, time_new_window_counter, \
            time_insertion_bit, time_insertion_counter, num_accuracy_check, \
            time_query_bit, time_query_counter, num_time_window \
            = fixed_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                            date_time_format,
                                                            monitored_groups,
                                                            threshold, alpha, time_unit, eval(window_size_units),
                                                            checking_interval, label_prediction,
                                                            label_ground_truth, correctness_column, use_nanosecond,
                                                            use_two_counters)


filename = f"accuracy_fixed_window.csv"

df = pd.DataFrame(accuracy_list, columns=sectors_list)
df["check_points"] = check_points

df.to_csv(filename, index=False)

