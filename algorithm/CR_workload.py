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
            counter_list.append(copy.deepcopy(counters))
            cr_list.append([counters[i] / total_counter for i in range(len(counters))])
            if not first_window_processed:
                uf = [True if counters[i] / total_counter <= threshold else False for i in range(len(counters))]
                delta = [abs(threshold * total_counter - counters[i]) * alpha for i in range(len(counters))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
            DFMonitor.insert(row)
            counters = [x * alpha for x in counters]
            total_counter *= alpha
        else:
            DFMonitor.insert(row)
        total_counter += 1
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
            cr_list.append(
                [DFMonitor_baseline.counters[i] / total_counter for i in range(len(DFMonitor_baseline.counters))])
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



def CR_traditional(timed_data, date_column, time_window_str, monitored_groups, threshold):
    number, unit = time_window_str.split()

    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    print(timed_data[timed_data['new_window'] == True].index)
    counter_values = [0 for g in monitored_groups]
    total_counter_of_window = 0
    counter_list = []  # record each time window's counter_values
    cr_list = []
    uf_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            cr = [counter_values[i] / total_counter_of_window for i in range(len(counter_values))]
            counter_list.append(copy.deepcopy(counter_values))
            cr_list.append(cr)
            uf_list.append([True if cr[i] <= threshold else False for i in range(len(cr))])
            total_counter_of_window = 0
            counter_values = [0 for _ in monitored_groups]
        # all value in counter_map of the group that the tuple satisfies add 1
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                counter_values[i] += 1
        total_counter_of_window += 1
    # last window:
    cr = [counter_values[i] / total_counter_of_window for i in range(len(counter_values))]
    counter_list.append(copy.deepcopy(counter_values))
    cr_list.append(cr)
    uf_list.append([True if cr[i] <= threshold else False for i in range(len(cr))])
    return counter_list, cr_list, uf_list


if __name__ == "__main__":

    data = pd.read_csv('../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
    # get distribution of compas_screening_date
    data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
    # data['compas_screening_date'].hist()

    monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}]
    alpha = 0.5
    threshold = 0.3
    date_column = "compas_screening_date"
    time_window_str = "1 month"

    DFMonitor_baseline, uf_list_baseline, counter_list_baseline, cr_list_baseline \
        = traverse_data_DFMonitor_baseline(data, date_column,
                                                    time_window_str,
                                                    monitored_groups,
                                                    threshold,
                                                    alpha)

    # use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
    DFMonitor, uf_list_DF, cr_list_DF, counter_list_DF = traverse_data_DFMonitor(data, date_column,
                                                                                          time_window_str,
                                                                                          monitored_groups,
                                                                                          threshold,
                                                                                          alpha)

    counter_list_trad, cr_list_trad, uf_list_trad = CR_traditional(data, date_column,
                                                                            time_window_str,
                                                                            monitored_groups,
                                                                            threshold)

    for i in range(0, len(cr_list_DF)):
        if cr_list_baseline[i] != cr_list_DF[i]:
            print("cr_list_baseline {} != cr_list_DF {}".format(cr_list_baseline[i], cr_list_DF[i]))



    for i in range(0, len(cr_list_DF)):
        print("cr_list_trad {}  cr_list_baseline {}  cr_list_DF {}  ".format(cr_list_trad[i],
                                                                             cr_list_baseline[i], cr_list_DF[i]))





