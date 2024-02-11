import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import FPR_0_20240125 as FPR
from algorithm import FPR_baseline_0_20240125 as FPR_baseline
import copy
from algorithm import config


# use CR to monitor tpr of different races over time
# a time window is a month

# Function to compute the time window key (e.g., year-month for 'month' windows)
def compute_time_window_key(row, window_type, window_size, first_row_time):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        if window_size == 1:
            return "{}-{}".format(row.year, row.month)
        else:
            # if row.month == 2:
            #     print("row = {}, first_row_time = {}".format(row, first_row_time))
            num_year = row.year - first_row_time.year
            num_month = num_year * 12 + row.month - 1
            num_window = num_month // window_size
            last_new_window_month = first_row_time.month + num_window * window_size
            return "{}-{}".format(first_row_time.year + last_new_window_month // 12,
                                  last_new_window_month % 12 + 1)
    elif window_type == 'week':
        # Calculate the number of days since the start of the year to the current row date
        days_since_start_of_year = (row - pd.Timestamp(year=row.year, month=1, day=1)).days
        # For a 2-week window, divide by 14 (number of days in 2 weeks) to find the window index
        window_index = days_since_start_of_year // (window_size * 7)  # 7 days in a week
        # Calculate the start date of the window
        window_start_date = pd.Timestamp(year=row.year, month=1, day=1) + pd.Timedelta(
            days=window_index * window_size * 7)
        return "{}-W{}".format(window_start_date.year, window_index + 1)
    elif window_type == 'day':
        return "{}-{}-{}".format(row.year, row.month, row.day // window_size * window_size)


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                         label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    counter_first_window_FP = [0 for _ in monitored_groups]
    counter_first_window_TN = [0 for _ in monitored_groups]
    first_window_processed = False
    index = 0

    for index, row in timed_data.iterrows():
        if row['new_window']:
            DFMonitor_baseline.new_window()
            if not first_window_processed:
                uf = []
                for i in range(len(monitored_groups)):
                    if counter_first_window_FP[i] + counter_first_window_TN[i] == 0:
                        uf.append(None)
                    else:
                        if (counter_first_window_FP[i] / (counter_first_window_TN[i] + counter_first_window_FP[i])
                                > threshold):
                            uf.append(True)
                        else:
                            uf.append(False)
                delta = [0] * len(monitored_groups)

                for i in range(len(monitored_groups)):
                    delta[i] = threshold * counter_first_window_TN[i] - (1 - threshold) * counter_first_window_FP[i]
                # print(delta)
                delta = [
                    round(abs(threshold * counter_first_window_TN[i] - (1 - threshold) * counter_first_window_FP[i])
                          * alpha, config.decimal) for i in range(len(monitored_groups))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if ((row[label_prediction] == 1 and row[label_ground_truth] == 0)
                or (row[label_prediction] == 0 and row[label_ground_truth] == 0)):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            DFMonitor_baseline.insert(row, label)
            if not first_window_processed:
                for group_idx, group in enumerate(monitored_groups):
                    if belong_to_group(row, group):
                        if label == "FP":
                            counter_first_window_FP[group_idx] += 1
                        else:
                            counter_first_window_TN[group_idx] += 1
            else:
                DFMonitor.insert(row, label)
        # print("index = {}, delta = {}".format(index, DFMonitor.delta))
        if first_window_processed:
            # if index == 1161:
            if DFMonitor.uf != DFMonitor_baseline.uf:
                print("+++++++ after loop not equal **********")
                print("index === {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
                DFMonitor_baseline.print()
                DFMonitor.print()
                return DFMonitor, DFMonitor_baseline
    return DFMonitor, DFMonitor_baseline


def traverse_data_DFMonitor(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                            label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    # print(timed_data[:1])
    # print(len(timed_data[timed_data['new_window'] == True]))
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    counter_values_FP = [0 for _ in monitored_groups]
    counter_values_TN = [0 for _ in monitored_groups]
    counter_list_FP = []  # record each time window's counter_values
    counter_list_TN = []
    first_window_processed = False
    uf_list = []
    fpr_list = []

    for index, row in timed_data.iterrows():
        if row['new_window']:  # even if this is not FP or TN, we need to compute when we have a new window
            uf_list.append(copy.deepcopy(DFMonitor.uf))
            deep_copy = copy.deepcopy(counter_values_FP)
            counter_list_FP.append(deep_copy)
            deep_copy = copy.deepcopy(counter_values_TN)
            counter_list_TN.append(deep_copy)
            fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
                   if counter_values_TN[i] + counter_values_FP[i] != 0
                   else None for i in range(len(monitored_groups))]
            fpr_list.append(fpr)
            counter_values_FP = [x * alpha for x in counter_values_FP]
            counter_values_TN = [x * alpha for x in counter_values_TN]
            if not first_window_processed:
                uf = [True if counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i]) <= threshold
                      else False
                      for i in range(len(monitored_groups))]
                delta = [round(abs(threshold * counter_values_TN[i] - (1 - threshold) * counter_values_FP[i])
                               * alpha, config.decimal)
                         for i in range(len(monitored_groups))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if ((row[label_prediction] == 1 and row[label_ground_truth] == 0) or
                (row[label_prediction] == 0 and row[label_ground_truth] == 0)):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            DFMonitor.insert(row, label)
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FP":
                        counter_values_FP[i] += 1
                    else:
                        counter_values_TN[i] += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor.uf))
    counter_list_FP.append(copy.deepcopy(counter_values_FP))
    counter_list_TN.append(copy.deepcopy(counter_values_TN))
    fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
           if counter_values_TN[i] + counter_values_FP[i] != 0
           else None for i in range(len(monitored_groups))]
    fpr_list.append(fpr)
    return DFMonitor, uf_list, fpr_list, counter_list_TN, counter_list_FP


def traverse_data_DF_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                              label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    # get the date_column value of the first row
    first_row_time = timed_data.loc[0, date_column]
    # starting_time = "{}-{}".format(row.year, (row.month - 1) // window_size + 1)
    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(window_type,
                                                                                            int(window_size),
                                                                                            first_row_time))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    print("num of new_window", len(timed_data[timed_data['new_window'] == True]))
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    uf_list = []
    fpr_list = []
    counter_values_FP = [0 for _ in monitored_groups]
    counter_values_TN = [0 for _ in monitored_groups]
    counter_list_FP = []  # record each time window's counter_values
    counter_list_TN = []
    # FP_each_window = [0] * len(monitored_groups)
    # TN_each_window = [0] * len(monitored_groups)
    for index, row in timed_data.iterrows():
        if row['new_window']:
            print("new window, index = {}, date = {}, before this point FP = {}, "
                  "TN = {}".format(index, row[date_column], counter_values_FP, counter_values_TN))
            uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
            deep_copy = copy.deepcopy(counter_values_FP)
            counter_list_FP.append(deep_copy)
            deep_copy = copy.deepcopy(counter_values_TN)
            counter_list_TN.append(deep_copy)
            fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
                   if counter_values_TN[i] + counter_values_FP[i] != 0
                   else None for i in range(len(monitored_groups))]
            fpr_list.append(fpr)
            # FP_each_window = [0] * len(monitored_groups)
            # TN_each_window = [0] * len(monitored_groups)

            DFMonitor_baseline.new_window()
            counter_values_FP = [x * alpha for x in counter_values_FP]
            counter_values_TN = [x * alpha for x in counter_values_TN]
        if ((row[label_prediction] == 1 and row[label_ground_truth] == 0) or
                (row[label_prediction] == 0 and row[label_ground_truth] == 0)):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
                # if row['race'] == 'Caucasian':
                #     FP_each_window[0] += 1
                # elif row['race'] == "African-American":
                #     FP_each_window[1] += 1
            else:
                label = "TN"
                # if row['race'] == 'Caucasian':
                #     TN_each_window[0] += 1
                # elif row['race'] == "African-American":
                #     TN_each_window[1] += 1

            DFMonitor_baseline.insert(row, label)
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FP":
                        counter_values_FP[i] += 1
                    else:
                        counter_values_TN[i] += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
    counter_list_FP.append(copy.deepcopy(counter_values_FP))
    counter_list_TN.append(copy.deepcopy(counter_values_TN))
    fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
           if counter_values_TN[i] + counter_values_FP[i] != 0
           else None for i in range(len(monitored_groups))]
    fpr_list.append(fpr)
    return DFMonitor_baseline, uf_list, fpr_list, counter_list_TN, counter_list_FP


def FPR_traditional(timed_data, date_column, time_window_str, monitored_groups, threshold,
                    label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    # print(timed_data[timed_data['new_window'] == True].index)
    counter_values_FP = [0 for _ in monitored_groups]
    counter_values_TN = [0 for _ in monitored_groups]
    counter_list_FP = []  # record each time window's counter_values
    counter_list_TN = []
    fpr_list = []
    uf_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
                   if counter_values_TN[i] + counter_values_FP[i] != 0
                   else None for i in range(len(monitored_groups))]
            print("index = {}, fpr = {}, date = {}".format(index, fpr, row[date_column]))
            counter_list_FP.append(copy.deepcopy(counter_values_FP))
            counter_list_TN.append(copy.deepcopy(counter_values_TN))
            fpr_list.append(fpr)
            uf_list.append(
                [True if fpr[i] is not None and fpr[i] <= threshold else False if fpr[i] is not None else None for i
                 in range(len(fpr))])
            counter_values_FP = [0 for _ in monitored_groups]
            counter_values_TN = [0 for _ in monitored_groups]

        if ((row[label_prediction] == 1 and row[label_ground_truth] == 0) or
                (row[label_prediction] == 0 and row[label_ground_truth] == 0)):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
            # all value in counter_map of the group that the tuple satisfies add 1
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FP":
                        counter_values_FP[i] += 1
                    else:
                        counter_values_TN[i] += 1
    # last window:
    fpr = [counter_values_FP[i] / (counter_values_TN[i] + counter_values_FP[i])
           if counter_values_TN[i] + counter_values_FP[i] != 0
           else None for i in range(len(monitored_groups))]
    counter_list_FP.append(copy.deepcopy(counter_values_FP))
    counter_list_TN.append(copy.deepcopy(counter_values_TN))
    fpr_list.append(fpr)
    uf_list.append(
        [True if fpr[i] is not None and fpr[i] <= threshold else False if fpr[i] is not None else None for i in
         range(len(fpr))])
    return uf_list, fpr_list, counter_list_TN, counter_list_FP


if __name__ == "__main__":
    data = pd.read_csv('../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
    # get distribution of compas_screening_date
    data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
    # data['compas_screening_date'].hist()

    monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}]
    alpha = 0.3
    threshold = 0.3
    date_column = "compas_screening_date"
    time_window_str = "8 month"


    DFMonitor_baseline, uf_list1, fpr_list1, counter_list_TN1, counter_list_FP1 = (
        traverse_data_DF_baseline(data, date_column, time_window_str, monitored_groups, threshold, alpha))


    #
    # DFMonitor, uf_list2, fpr_list2, counter_list_TN2, counter_list_FP2 = traverse_data_DFMonitor(data, date_column,
    #                                                                                              time_window_str,
    #                                                                                              monitored_groups,
    #                                                                                              threshold, alpha)

    # uf_list, fpr_list, counter_list_TN, counter_list_FP = FPR_traditional(data, "compas_screening_date", "1 month",
    #                                                                       monitored_groups,
    #                                                                       threshold)

    # DFMonitor, DFMonitor_baseline = traverse_data_DFMonitor_and_baseline(data, date_column, time_window_str,
    #                                                                      monitored_groups, threshold, alpha)

    #
    #
    # # compare fpr1 and fpr2
    # def compare_fpr(fpr_list1, fpr_list2):
    #     for i in range(len(fpr_list1)):
    #         for j in range(len(fpr_list1[i])):
    #             if fpr_list1[i][j] != fpr_list2[i][j]:
    #                 print(
    #                     "fpr_list1[i][j] = {}, fpr_list2[i][j] = {}, i={}, j={}".format(fpr_list1[i][j],
    #                                                                                     fpr_list2[i][j], i, j))
    #                 return False
    #     return True
    #
    #
    # print(compare_fpr(fpr_list1, fpr_list2))
    # print(compare_fpr(counter_list_FP1, counter_list_FP2))
    # print(compare_fpr(counter_list_TN1, counter_list_TN2))
    # print(uf_list1 == uf_list2)
    #
    # print(fpr_list1[12:])
    # print(fpr_list2[12:])
