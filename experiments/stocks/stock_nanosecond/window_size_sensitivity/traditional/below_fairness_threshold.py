import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys



sys.path.append("../../../../../../")
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



sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services']
#
# window_size_unit_list = ["100ms", "200ms", "500ms", "1s", "2s", "5s"]


window_size_unit_list = ["100ms", "200ms", "500ms", "1s", "2s", "5s"]



fairness_threshold = 0.6

time_start_picture = 0
time_end_picture = 0


fig, axes = plt.subplots(1, 3, figsize=(7, 1.4))
plt.subplots_adjust(top=0.85, bottom=0.33, hspace=0, wspace=0.25, left=0.05, right=0.99)

time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')





window_size_units_list = ["100ms", "200ms", "500ms", "1s", "2s", "5s"]
curve_names = ["Technology", "Consumer Cyclical", "Communication Services",]


def merge_overlapping_intervals(intervals, tolerance_ns=10000):
    # Defensive check: skip if the list is empty
    if not intervals:
        return []

    # Check if it's already a list of tuples; if not, raise a clear error
    if not isinstance(intervals[0], (list, tuple)) or len(intervals[0]) != 2:
        raise ValueError(f"Expected list of (start, end) tuples, but got: {intervals}")

    tolerance_ns = pd.Timedelta(tolerance_ns, unit="ns")
    intervals = sorted(intervals, key=lambda x: x[0])  # sort by start time
    merged = []

    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last_start, last_end = merged[-1]
            curr_start, curr_end = interval
            if curr_start <= last_end + tolerance_ns:
                merged[-1] = (last_start, max(last_end, curr_end))
            else:
                merged.append(interval)

    return merged




def get_below_threshold_intervals(df, fairness_threshold, fairness_column="fairness"):
    below_threshold_intervals = []
    current_interval = None

    for index, row in df.iterrows():
        if row[fairness_column] < fairness_threshold:
            if current_interval is None:
                current_interval = [row["ts_event"], row["ts_event"]]
            else:
                current_interval[1] = row["ts_event"]
        else:
            if current_interval is not None:
                below_threshold_intervals.append(current_interval)
                current_interval = None

    if current_interval is not None:
        below_threshold_intervals.append(current_interval)

    return below_threshold_intervals


def convert_to_numeric(intervals):
    return [(pd.Timestamp(s).value, pd.Timestamp(e).value) for s, e in intervals]

def intervals_overlap(interval1, interval2):
    return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]

def convert_back(intervals_numeric):
    return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals_numeric]

def get_all_interval_coverage(intervals_by_window, window_sizes):
    """
    Returns:
        coverage_dict: { interval: [list of window sizes that capture it] }
    """
    interval_sources = []  # List of (interval, window_size)

    for win_idx, intervals in enumerate(intervals_by_window):
        win_size = window_sizes[win_idx]
        merged_intervals = merge_overlapping_intervals(intervals)
        numeric_intervals = convert_to_numeric(merged_intervals)
        interval_sources.extend([(interval, win_size) for interval in numeric_intervals])

    # Deduplicate intervals using their numeric representation
    all_intervals = list({interval for interval, _ in interval_sources})

    # Group intervals by coverage
    coverage_dict = {}
    for interval in all_intervals:
        captured_by = []
        for win_idx, intervals in enumerate(intervals_by_window):
            win_size = window_sizes[win_idx]
            merged_intervals = merge_overlapping_intervals(intervals)
            other_numeric = convert_to_numeric(merged_intervals)
            if any(intervals_overlap(interval, other) for other in other_numeric):
                captured_by.append(win_size)
        coverage_dict[convert_back([interval])[0]] = sorted(captured_by)

    return coverage_dict

below_threshold_intervals = []
result_file_list = dict()
for window_size in window_size_unit_list:
    file_name = f"traditional_accuracy_time_window_{window_size}.csv"
    df = pd.read_csv(file_name)
    # df = df[df['sector'].isin(sorted_sector_list)]
    df = df[df['sector'].isin(sector_list)]
    result_file_list[window_size] = df

for curve in curve_names:
    num_detected = []
    below_threshold_intervals_per_window_size = []
    for a, window_size in enumerate(window_size_units_list):
        df = result_file_list[window_size]
        df = df[df["sector"] == curve]
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, format="mixed")
        df_below_threshold = []
        # df["ts_event"] = pd.to_datetime(df["ts_event"])
        time_start_picture = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
        time_end_picture = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
        total_duration_ns = (time_end_picture - time_start_picture).total_seconds() * 1e9
        # get data between time_start and time_end

        df = df[(df["ts_event"] >= time_start_picture) & (df["ts_event"] <= time_end_picture)]

        ts_event = df["ts_event"].tolist()
        x_list = np.arange(0, len(df))
        below_threshold_intervals = get_below_threshold_intervals(df, fairness_threshold, "accuracy")
        below_threshold_intervals = merge_overlapping_intervals(below_threshold_intervals)
        # print(curve, len(below_threshold_intervals), below_threshold_intervals)
        # print("window_size", window_size, "number of below threshold intervals", len(df_below_threshold))
        below_threshold_intervals_per_window_size.append(below_threshold_intervals)
        # print("below_threshold_intervals", below_threshold_intervals)
        num_detected.append(len(below_threshold_intervals)) # num of intervals detected by each window size

    print(f"\n--- Intervals for {curve} ---")
    coverage = get_all_interval_coverage(below_threshold_intervals_per_window_size, window_size_units_list)
    num_intervals_detected = [0]*len(window_size_units_list)
    for i, window_size in enumerate(window_size_units_list):
        for interval, windows in coverage.items():
            if window_size in windows:
                num_intervals_detected[i] += 1

    # for interval, windows in coverage.items():
    #     print(f"Interval {interval} captured by: {windows}")
    print("num_intervals_detected_by_smallest_window, also detected by larger windows", num_intervals_detected)
    print("percentage of intervals detected by each window size", [f"{num_intervals_detected[i]/num_intervals_detected[0]:.2f}"
                                                                   for i in range(len(num_intervals_detected))])

