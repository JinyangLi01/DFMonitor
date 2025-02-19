import colorsys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
from algorithm.dynamic_window import Accuracy_workload as dynamic_window_workload
from algorithm.per_item import Accuracy_workload as fixed_window_workload
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

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
data_file_name = f"../predict_results/prediction_result_{data_file_name}_chunk_size_{len_chunk}_v3.csv"
data = pd.read_csv(data_file_name)



time_start = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')


data["ts_event"] = pd.to_datetime(data["ts_event"])
# get data between time_start and time_end
data = data[(data["ts_event"] >= time_start) & (data["ts_event"] <= time_end)]

# reset the index
data = data.reset_index(drop=True)
print("len of selected data", len(data))



print(data["sector"].unique())
date_column = "ts_event"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"sector": 'Technology'}, {'sector': 'Communication Services'}, {"sector": 'Energy'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Consumer Cyclical'}, {"sector": 'Financial Services'},
                    {"sector": 'Healthcare'}, {"sector": 'Industrials'}, {"sector": "Basic Materials"}]
print(data[:5])
alpha = 0.99995


threshold = 0.3
label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "20000"
checking_interval = "200 millisecond"
use_nanosecond = True

data[correctness_column] = data[label_prediction] == data[label_ground_truth]

######################################## Fixed window ########################################


DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
            window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
            time_new_window_bit, time_new_window_counter, \
            time_insertion_bit, time_insertion_counter, num_accuracy_check, \
            time_query_bit, time_query_counter, num_time_window \
    = fixed_window_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, eval(window_size_units),
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column, use_nanosecond,
                                                    use_two_counters)
# already check the correctness of the accuracy finally got
final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]


Technology_time_decay = [x[0] for x in accuracy_list]
ConsumerCyclical_time_decay = [x[1] for x in accuracy_list]
CommunicationServices_time_decay = [x[2] for x in accuracy_list]



filename = f"fixed_window_{checking_interval}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Technology_time_decay", "ConsumerCyclical_time_decay", "CommunicationServices_time_decay",
                     "check_points"])
    for i in range(len(Technology_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1], accuracy_list[i][2], check_points[i]])




######################################## Dynamic window ########################################
T_b = 100
T_in = 30

DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
    window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
    time_new_window_bit, time_new_window_counter, \
    time_insertion_bit, time_insertion_counter, num_accuracy_check, \
    time_query_bit, time_query_counter, num_time_window \
    = dynamic_window_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, T_b, T_in,
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column,
                                                    use_two_counters, use_nanosecond)
# already check the correctness of the accuracy finally got
final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]
# print("final_accuracy", final_accuracy)

# save the result to a file
Technology_time_decay = [x[0] for x in accuracy_list]
ConsumerCyclical_time_decay = [x[1] for x in accuracy_list]
CommunicationServices_time_decay = [x[2] for x in accuracy_list]



filename = f"adaptive_window_{checking_interval}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["Technology_time_decay", "ConsumerCyclical_time_decay", "CommunicationServices_time_decay",
                     "check_points"])
    for i in range(len(Technology_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1], accuracy_list[i][2], check_points[i]])



######################################## traditional ########################################



window_size = "200ms"

value_col_name = "calculated_value"




def plot_certain_time_window(data, value_col_name, window_size, axs):
    time_window = pd.to_timedelta(window_size)  # Convert to Timedelta

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
    accuracy_df.to_csv("traditional.csv", index=False)

    # Plot accuracy for each sector at the end of each dynamic time window
    for sector, sector_data in accuracy_df.groupby("sector"):
        axs.plot(sector_data["ts_event"], sector_data["accuracy"], marker='o', label=sector)








plt.savefig("curves_smoothness_score.png", bbox_inches='tight')
plt.show()



