
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import CR_0_20240118 as CR
from algorithm import CR_baseline_0_20240118 as CR_baseline

data = pd.read_csv('../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
# get distribution of compas_screening_date
data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
data['compas_screening_date'].hist()


# use CR to monitor tpr of different races over time
# a time window is a month

# Function to compute the time window key (e.g., year-month for 'month' windows)
def compute_time_window_key(row, window_type):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        return f"{row.year}-{row.month}"
    elif window_type == 'week':
        return f"{row.year}-{row.week}"
    elif window_type == 'day':
        return f"{row.year}-{row.month}-{row.day}"


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True



def monitorCR_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data.loc[1:, 'new_window'] = timed_data['window_key'].iloc[1:] != timed_data['window_key'].shift(1).iloc[1:]

    DFMonitor_baseline = CR_baseline.CR_baseline(monitored_groups, alpha, threshold)
    # DFMonitor.print()

    def process_each_tuple(row, DFMonitor):
        if row['new_window']:  # new window
            # print("new window, row={}, {}".format(row['id'], row['compas_screening_date']))
            # DFMonitor.print()
            DFMonitor.new_window()
        DFMonitor.insert(row)
        return

    timed_data.apply(process_each_tuple, axis=1, args=(DFMonitor_baseline,))
    return DFMonitor_baseline



def monitorCR(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data.loc[1:, 'new_window'] = timed_data['window_key'].iloc[1:] != timed_data['window_key'].shift(1).iloc[1:]

    print(timed_data['new_window'])

    DFMonitor = CR.DF_CR(monitored_groups, alpha, threshold)
    counters = [0] * len(monitored_groups)
    total_counter = 0
    first_window_processed = False

    def process_each_tuple(row, DFMonitor):
        nonlocal counters
        nonlocal total_counter
        nonlocal first_window_processed
        if row['new_window']:  # new window
            # print("new window, row={}, {}".format(row['id'], row['compas_screening_date']))
            # DFMonitor.print()
            if not first_window_processed:
                # print("initialization, row={}, {}".format(row['id'], row['compas_screening_date']))
                uf = [True if counters[i] / total_counter > threshold else False for i in range(len(counters))]
                delta = [abs(threshold * total_counter - counters[i]) * alpha for i in range(len(counters))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
                # DFMonitor.print()
            else:
                DFMonitor.new_window()
            DFMonitor.insert(row)
        elif not first_window_processed:  # initialization
            # print("initialization, row={}, {}".format(row['id'], row['compas_screening_date']))
            total_counter += 1
            for group in monitored_groups:
                if belong_to_group(row, group):
                    counters[monitored_groups.index(group)] += 1
        else:
            DFMonitor.insert(row)
        return

    timed_data.apply(process_each_tuple, axis=1, args=(DFMonitor,))
    return DFMonitor



monitored_groups = [{"race": 'African-American'}, {"race": 'Caucasian'}]
alpha = 0.5
threshold = 0.1
DFMonitor = monitorCR(data, "compas_screening_date", "1 month", monitored_groups, threshold, alpha)

DFMonitor_baseline = monitorCR_baseline(data, "compas_screening_date", "1 month", monitored_groups, threshold, alpha)


for i in range(len(DFMonitor.uf)):
    if DFMonitor.uf[i] != DFMonitor_baseline.uf[i]:
        print("i={},group={}, uf={}, uf_baseline={}".format(i, DFMonitor.groups[i], DFMonitor.uf[i], DFMonitor_baseline.uf[i]))
        # print("i={},group={}, delta={}, counter={}, counter_total={}".format(i, DFMonitor.groups[i], DFMonitor.delta[i], DFMonitor_baseline.counters[i], DFMonitor_baseline.counter_total))


