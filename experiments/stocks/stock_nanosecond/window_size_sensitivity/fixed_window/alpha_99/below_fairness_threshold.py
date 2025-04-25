import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys



sys.path.append("../../../../../../")
from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np

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





time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
len_chunk = 1


label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"

checking_interval = "100000 nanosecond"
use_nanosecond = True



monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
                    {"sector": 'Financial Services'}]

sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
                          'Healthcare', 'Financial Services']


alpha=0.9
fairness_threshold = 0.6

time_start_picture = 0
time_end_picture = 0


fig, axes = plt.subplots(1, 3, figsize=(7, 1.4))
plt.subplots_adjust(top=0.85, bottom=0.33, hspace=0, wspace=0.25, left=0.05, right=0.99)

time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')




window_size_units_list = [400, 1000, 3000]
curve_names = ["Technology_time_decay", "ConsumerCyclical_time_decay", "CommunicationServices_time_decay",]


def get_below_threshold_intervals(df, fairness_threshold, fairness_column="fairness"):
    below_threshold_intervals = []
    current_interval = None

    for index, row in df.iterrows():
        if row[fairness_column] < fairness_threshold:
            if current_interval is None:
                current_interval = [row["check_points"], row["check_points"]]
            else:
                current_interval[1] = row["check_points"]
        else:
            if current_interval is not None:
                below_threshold_intervals.append(current_interval)
                current_interval = None

    if current_interval is not None:
        below_threshold_intervals.append(current_interval)

    return below_threshold_intervals


below_threshold_intervals_window_size = []

for a, window_size in enumerate(window_size_units_list):
    filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
                f"_time_unit_{time_unit}*{window_size}_check_interval_{checking_interval}_start_time_"
                f"{time_start}_end_time_{time_end}.csv")

    with open(filename, 'r') as f:
        contents = f.read()
    if "]\"" in contents or "\"[" in contents:
        updated_contents = contents.replace("]\"", "")
        updated_contents = updated_contents.replace("\"[", "")

        with open(filename, 'w') as f:
            f.write(updated_contents)

    df = pd.read_csv(filename)
    df_below_threshold = []
    df["check_points"] = pd.to_datetime(df["check_points"])
    print(alpha, len(df))
    print(df.columns)
    # Remove timezone from warm_up_time
    time_start_picture = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
    time_end_picture = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
    total_duration_ns = (time_end_picture - time_start_picture).total_seconds() * 1e9

    df["check_points"] = pd.to_datetime(df["check_points"])
    # get data between time_start and time_end
    df = df[(df["check_points"] >= time_start_picture) & (df["check_points"] <= time_end_picture)]
    print("len of selected data", len(df))
    # df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
    check_points = df["check_points"].tolist()
    x_list = np.arange(0, len(df))
    for curve in curve_names:
        below_threshold_intervals = get_below_threshold_intervals(df, fairness_threshold, curve)
        df_below_threshold.append(below_threshold_intervals)
    below_threshold_intervals_window_size.append(df_below_threshold)

    print("below_threshold_intervals", below_threshold_intervals_window_size)






