
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import CR_0_20240118 as CR
from algorithm import CR_baseline_0_20240118 as CR_baseline
from algorithm import config
import copy


# use CR to monitor tpr of different races over time
# a time window is a month

# Function to compute the time window key (e.g., year-month for 'month' windows)
def compute_time_window_key(row, window_type):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        return "{}-{}".format(row.year, row.month)
    elif window_type == 'week':
        return "{}-{}".format(row.year, row.strftime('%U'))
    elif window_type == 'day':
        return "{}-{}-{}".format(row.year, row.month, row.day)


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data["new_window"].loc[0] = False
    DFMonitor = CR.DF_CR(monitored_groups, alpha, threshold)
    DFMonitor_baseline = CR_baseline.CR_baseline(monitored_groups, alpha, threshold)
    counters = [0] * len(monitored_groups)
    total_counter = 0
    first_window_processed = False
    index = 0
    row = timed_data.iloc[index]
    for index, row in timed_data.iterrows():
        if first_window_processed:
            if DFMonitor.uf != DFMonitor_baseline.uf:
                print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
                DFMonitor.print()
                DFMonitor_baseline.print()
                return DFMonitor, DFMonitor_baseline

        if row['new_window']:
            DFMonitor_baseline.new_window()
            DFMonitor_baseline.insert(row)
            if not first_window_processed:
                uf = [True if counters[i] / total_counter <= threshold else False for i in range(len(counters))]
                delta = [abs(threshold * total_counter - counters[i]) * alpha for i in range(len(counters))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
                # DFMonitor.print()
            else:
                DFMonitor.new_window()
            DFMonitor.insert(row)
            # print(DFMonitor_baseline.uf, DFMonitor_baseline.counters)
        else:
            DFMonitor_baseline.insert(row)
            if not first_window_processed:
                total_counter += 1
                for group in monitored_groups:
                    if belong_to_group(row, group):
                        counters[monitored_groups.index(group)] += 1
            else:
                DFMonitor.insert(row)
    # last window:
    if DFMonitor.uf != DFMonitor_baseline.uf:
        print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
        DFMonitor.print()
        DFMonitor_baseline.print()
    return DFMonitor, DFMonitor_baseline


def traverse_data_DFMonitor(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    # timed_data.loc["new_window", 0] = False
    timed_data.loc[0, "new_window"] = False
    # print(timed_data[:1])
    # print(len(timed_data[timed_data['new_window'] == True]))
    DFMonitor = CR.DF_CR(monitored_groups, alpha, threshold)
    counters = [0] * len(monitored_groups)
    total_counter = 0
    first_window_processed = False
    counter_list = []

    uf_list = []
    cr_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            uf_list.append(copy.deepcopy(DFMonitor.uf))
            cr_list.append([counters[i] / total_counter for i in range(len(counters))])
            if not first_window_processed:
                uf = [True if counters[i] / total_counter <= threshold else False for i in range(len(counters))]
                delta = [abs(threshold * total_counter - counters[i]) * alpha for i in range(len(counters))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
                # DFMonitor.print()
            else:
                DFMonitor.new_window()
            DFMonitor.insert(row)
            counter_list.append(copy.deepcopy(counters))

        else:
            if not first_window_processed:
                total_counter += 1
            else:
                total_counter += 1
                DFMonitor.insert(row)
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    counters[i] += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor.uf))
    cr_list.append([counters[i] / total_counter for i in range(len(counters))])
    counter_list.append(copy.deepcopy(counters))
    return DFMonitor, uf_list, cr_list, counter_list


def traverse_data_DFMonitor_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    DFMonitor_baseline = CR_baseline.CR_baseline(monitored_groups, alpha, threshold)
    uf_list = []
    counter_list = []
    total_counter = 0
    cr_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
            counter_list.append(copy.deepcopy(DFMonitor_baseline.counters))
            cr_list.append([DFMonitor_baseline.counters[i] / total_counter for i in range(len(DFMonitor_baseline.counters))])
            total_counter *= alpha
            DFMonitor_baseline.new_window()
            DFMonitor_baseline.insert(row)
            total_counter += 1
        else:
            DFMonitor_baseline.insert(row)
            total_counter += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
    counter_list.append(copy.deepcopy(DFMonitor_baseline.counters))
    cr_list.append([DFMonitor_baseline.counters[i] / total_counter for i in range(len(DFMonitor_baseline.counters))])
    return DFMonitor_baseline, uf_list, counter_list, cr_list

#
# def monitorCR_without_class(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
#     number, unit = time_window_str.split()
#
#     timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
#     # Initialize all rows as not the start of a new window
#     timed_data['new_window'] = False
#     # Determine the start of a new window for all rows except the first
#     timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
#     timed_data.loc[0, "new_window"] = False
#     counter_map = {g: 0 for g in monitored_groups}
#     total_counter = 0
#     counter_list = []
#     for index, row in timed_data.iterrows():
#         if row['new_window'] == True:
#             counter_list.append(copy.deepcopy(counter_map.values()))
#             # all value in counter_map timed by alpha
#             for g in monitored_groups:
#                 counter_map[g] *= alpha
#             total_counter *= alpha
#         # all value in counter_map of the group that the tuple satisfies add 1
#         for g in monitored_groups:
#             if belong_to_group(row, g):
#                 counter_map[g] += 1
#         total_counter += 1
#     return counter_map, total_counter, counter_list


def CR_each_window_seperately(timed_data, date_column, time_window_str, monitored_groups, threshold):
    number, unit = time_window_str.split()

    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    counter_values = [0 for g in monitored_groups]
    total_counter_of_window = 0
    counter_list = []  # record each time window's counter_values
    cr_list = []
    uf_list = []
    for index, row in timed_data.iterrows():
        if row['new_window'] == True:
            cr = [counter_values[i] / total_counter_of_window for i in range(len(counter_values))]
            counter_list.append(copy.deepcopy(counter_values))
            cr_list.append(cr)
            uf_list.append([True if cr[i] <= threshold else False for i in range(len(cr))] )
            total_counter_of_window = 0
            counter_values = [0 for g in monitored_groups]
        # all value in counter_map of the group that the tuple satisfies add 1
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                counter_values[i] += 1
        total_counter_of_window += 1
    return counter_list, cr_list, uf_list



# data = pd.read_csv('../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
# # get distribution of compas_screening_date
# data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
# # data['compas_screening_date'].hist()

# monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}]
# alpha = 0.5
# threshold = 0.3
#
# DFMonitor, DFMonitor_baseline = process_data_one_by_one(data, "compas_screening_date", "1 month", monitored_groups, threshold, alpha)
