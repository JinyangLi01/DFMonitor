
import time

import pandas as pd
from algorithm.dynamic_window import Accuracy_bit
from algorithm.dynamic_window import Accuracy_counters
import copy
import numpy as np


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, checking_interval_units = 1,
                                         label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",
                                         T_b=50, T_in=10, use_two_counters=True):
    # Initialize DF_Accuracy_Dynamic_Window_Counter objects
    print("monitored_groups", monitored_groups)

    DFMonitor_counter = Accuracy_counters.DF_Accuracy_Dynamic_Window_Counter(monitored_groups, alpha, threshold, T_b, T_in, use_two_counters)
    DFMonitor_bit = Accuracy_bit.DF_Accuracy_Dynamic_Window_Bit(monitored_groups, alpha, threshold, T_b, T_in, use_two_counters)

    # Initialize the time for the first row
    last_clock = timed_data.iloc[0][date_column]
    last_accuracy_check = last_clock
    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    current_clock = 0
    checking_interval_seconds = 0
    dynamic_window_times = []

    # Traverse through the data
    for index, row in timed_data.iterrows():
        if int(timed_data.loc[index, correctness_column]) == 1:
            label = "correct"
        else:
            label = "incorrect"
        current_clock = row[date_column]  # Current timestamp
        time_unit_val, time_unit_str = time_unit.split(" ")
        time_unit_val = int(time_unit_val)
        time_diff_units = 0
        print(f"current_clock {current_clock}, last_clock {last_clock}, time_unit_val {time_unit_val}, time_unit_str {time_unit_str}")
        print(f"current_clock - last_clock {(current_clock - last_clock).total_seconds()}")

        if time_unit_str == "second":
            time_diff_units = ((current_clock - last_clock).total_seconds() / time_unit_val)
        elif time_unit_str == "hour":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (3600 * time_unit_val))
        elif time_unit_str == "day":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (86400 * time_unit_val))
        elif time_unit_str == "min":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (60 * time_unit_val))

        # Handle intervals where no data arrives but accuracy checks are needed
        checking_interval_seconds = checking_interval_units * time_unit_val * (
            1 if time_unit_str == "second" else
            60 if time_unit_str == "min" else
            3600 if time_unit_str == "hour" else
            86400)  # Handling different time units for checking intervals

        while (current_clock - last_accuracy_check).total_seconds() >= checking_interval_seconds:
            print("============= Performing scheduled accuracy check =============")
            print("counters", DFMonitor_counter.get_counter_correctness())
            cur_accuracy = DFMonitor_counter.get_accuracy_list()
            accuracy_list.append(cur_accuracy)
            counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
            counter_list_correct.append(counter_corr)
            counter_list_incorrect.append(counter_incorr)
            last_accuracy_check += pd.Timedelta(seconds=checking_interval_seconds)

        new_batch = DFMonitor_counter.whether_need_batch_renewal(time_diff_units)
        DFMonitor_bit.insert(row, label, time_diff_units)
        DFMonitor_counter.insert(row, label, time_diff_units)

        # Append UF list for tracking fairness/unfairness
        uf_list.append(copy.deepcopy(DFMonitor_bit.get_uf_list()))

        if new_batch:
            print(f"==================  new batch, applying exponential time decay")
            # Append the current time as the reset time
            dynamic_window_times.append(current_clock)
            # Apply time decay to counters
            counter_values_incorrect = [x * (alpha**time_diff_units) for x in counter_values_incorrect]
            counter_values_correct = [x * (alpha**time_diff_units) for x in counter_values_correct]
            # Update the last clock for the next iteration
            last_clock = current_clock


        # Update the correct/incorrect counters based on group membership
        for i, g in enumerate(monitored_groups):
            if DFMonitor_counter.belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1

    # Perform a final accuracy check after the last data point
    while (current_clock - last_accuracy_check).total_seconds() >= checking_interval_seconds:
        print("============= Performing final accuracy check =============")
        cur_accuracy = DFMonitor_counter.get_accuracy_list()
        accuracy_list.append(cur_accuracy)
        counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
        counter_list_correct.append(counter_corr)
        counter_list_incorrect.append(counter_incorr)
        last_accuracy_check += pd.Timedelta(seconds=checking_interval_seconds)

    return DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, dynamic_window_times
