
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import FPR_0_20240125 as FPR
from algorithm import FPR_baseline_0_20240125 as FPR_baseline
import copy


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


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                         label_prediction = "predicted", label_ground_truth = "ground_truth"):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data["new_window"].loc[0] = False
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    counter_first_window_FP = [0 for g in monitored_groups]
    counter_first_window_TN = [0 for g in monitored_groups]
    first_window_processed = False
    index = 0
    row = timed_data.iloc[index]

    for index, row in timed_data.iterrows():
        # only modify DFMonitor when it is FP or TN
        if (row[label_prediction] == 1 and row[label_ground_truth] == 0) or (row[label_prediction] == 0 and row[label_ground_truth == 0]):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"

            if first_window_processed:
                if DFMonitor.uf != DFMonitor_baseline.uf:
                    print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
                    DFMonitor.print()
                    DFMonitor_baseline.print()
                    return DFMonitor, DFMonitor_baseline

            if row['new_window']:
                DFMonitor_baseline.new_window()
                DFMonitor_baseline.insert(row, label)

                if not first_window_processed:
                    uf = [True if counter_first_window_FP[i] / (counter_first_window_TN[i] + counter_first_window_FP[i])
                                  <= threshold else False for i in range(len(monitored_groups))]
                    delta = [abs(threshold * counter_first_window_TN[i] - (1-threshold) * counter_first_window_FP[i])
                             *alpha for i in range(len(monitored_groups))]
                    DFMonitor.initialization(uf, delta)
                    first_window_processed = True
                    # DFMonitor.print()
                else:
                    DFMonitor.new_window()
                DFMonitor.insert(row, label)
                # print(DFMonitor_baseline.uf, DFMonitor_baseline.counters)
            else:
                DFMonitor_baseline.insert(row, label)
                if not first_window_processed:
                    for group in monitored_groups:
                        if belong_to_group(row, group):
                            if label == "FP":
                                counter_first_window_FP[monitored_groups.index(group)] += 1
                            else:
                                counter_first_window_TN[monitored_groups.index(group)] += 1
                else:
                    DFMonitor.insert(row, label)
            # if first_window_processed:
            #     if DFMonitor.uf != DFMonitor_baseline.uf:
            #         print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
            #         DFMonitor.print()
            #         DFMonitor_baseline.print()
            #         return DFMonitor, DFMonitor_baseline
    # last window:
    if first_window_processed:
        if DFMonitor.uf != DFMonitor_baseline.uf:
            print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
            DFMonitor.print()
            DFMonitor_baseline.print()
            return DFMonitor, DFMonitor_baseline
    return DFMonitor, DFMonitor_baseline


def traverse_data_DFMonitor(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                            label_prediction = "predicted", label_ground_truth = "ground_truth"):
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
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    counter_list_FP = [0] * len(monitored_groups)  # record each time window's counter_values
    counter_list_TN = [0] * len(monitored_groups)
    first_window_processed = False

    uf_list = []
    fpr_list = []

    for index, row in timed_data.iterrows():
        if (row[label_prediction] == 1 and row[label_ground_truth] == 0) or (row[label_prediction] == 0 and row[label_ground_truth == 0]):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            if row['new_window']:
                uf_list.append(copy.deepcopy(DFMonitor.uf))
                fpr = [counter_list_FP[i] / (counter_list_TN[i] + counter_list_FP[i]) for i in
                       range(len(monitored_groups))]
                fpr_list.append(fpr)
                if not first_window_processed:
                    uf = [True if counter_list_FP[i] / (counter_list_TN[i] + counter_list_FP[i]) <= threshold else False
                          for i in range(len(monitored_groups))]
                    delta = [abs(threshold * counter_list_TN[i] - (1-threshold) * counter_list_FP[i]) * alpha
                                for i in range(len(monitored_groups))]
                    DFMonitor.initialization(uf, delta)
                    first_window_processed = True
                else:
                    DFMonitor.new_window()
                DFMonitor.insert(row, label)
            else:
                for group in monitored_groups:
                    if belong_to_group(row, group):
                        if label == "FP":
                            counter_list_FP[monitored_groups.index(group)] += 1
                        else:
                            counter_list_TN[monitored_groups.index(group)] += 1
                DFMonitor.insert(row, label)
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor.uf))
    fpr = [counter_list_FP[i] / (counter_list_TN[i] + counter_list_FP[i]) for i in range(len(monitored_groups))]
    fpr_list.append(fpr)
    return DFMonitor, uf_list, fpr_list, counter_list_TN, counter_list_FP


def traverse_data_DFMonitor_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                    label_prediction = "predicted", label_ground_truth = "ground_truth"):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    uf_list = []
    fpr_list = []
    counter_values_FP = [0 for g in monitored_groups]
    counter_values_TN = [0 for g in monitored_groups]
    counter_list_FP = []  # record each time window's counter_values
    counter_list_TN = []
    for index, row in timed_data.iterrows():
        if (row[label_prediction] == 1 and row[label_ground_truth] == 0) or (row[label_prediction] == 0 and row[label_ground_truth == 0]):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            if row['new_window']:
                uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
                counter_list_FP.append(copy.deepcopy(counter_values_FP))
                counter_list_TN.append(copy.deepcopy(counter_values_TN))
                fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i]) for i in range(len(monitored_groups))]
                fpr_list.append(fpr)
                DFMonitor_baseline.new_window()
                DFMonitor_baseline.insert(row, label)
            else:
                DFMonitor_baseline.insert(row, label)
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
    counter_list_FP.append(copy.deepcopy(counter_values_FP))
    counter_list_TN.append(copy.deepcopy(counter_values_TN))
    fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i]) for i in range(len(monitored_groups))]
    fpr_list.append(fpr)
    return DFMonitor_baseline, uf_list, fpr_list, counter_list_TN, counter_list_FP



def FPR_each_window_seperately(timed_data, date_column, time_window_str, monitored_groups, threshold,
                               label_prediction = "predicted", label_ground_truth = "ground_truth"):
    number, unit = time_window_str.split()

    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    counter_values_FP = [0 for g in monitored_groups]
    counter_values_TN = [0 for g in monitored_groups]
    counter_list_FP = []  # record each time window's counter_values
    counter_list_TN = []
    fpr_list = []
    uf_list = []
    for index, row in timed_data.iterrows():
        if (row[label_prediction] == 1 and row[label_ground_truth] == 0) or (row[label_prediction] == 0 and row[label_ground_truth == 0]):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            if row['new_window'] == True:
                fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i]) for i in range(len(monitored_groups))]
                counter_list_FP.append(copy.deepcopy(counter_values_FP))
                counter_list_TN.append(copy.deepcopy(counter_values_TN))
                fpr_list.append(fpr)
                uf_list.append([True if fpr[i] <= threshold else False for i in range(len(fpr))] )
                counter_values_FP = [0 for g in monitored_groups]
                counter_values_TN = [0 for g in monitored_groups]
            # all value in counter_map of the group that the tuple satisfies add 1
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FP":
                        counter_values_FP[i] += 1
                    else:
                        counter_values_TN[i] += 1
    # last window:
    fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i]) for i in range(len(monitored_groups))]
    counter_list_FP.append(copy.deepcopy(counter_values_FP))
    counter_list_TN.append(copy.deepcopy(counter_values_TN))
    fpr_list.append(fpr)
    uf_list.append([True if fpr[i] <= threshold else False for i in range(len(fpr))])
    return uf_list, fpr_list, counter_list_TN, counter_list_FP




# data = pd.read_csv('../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
# # get distribution of compas_screening_date
# data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
# # data['compas_screening_date'].hist()

# monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}]
# alpha = 0.5
# threshold = 0.3
#
# DFMonitor, DFMonitor_baseline = process_data_one_by_one(data, "compas_screening_date", "1 month", monitored_groups, threshold, alpha)
