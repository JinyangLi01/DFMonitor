
import time

import pandas as pd
from algorithm.per_item import Accuracy_bit
from algorithm.per_item import Accuracy_counters
import copy
import numpy as np


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


# time_unit = 10 hour
def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, window_size_units, checking_interval_units = 1, label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="", use_two_counters=True):
    # Initialize DFMonitor counter and bit objects
    DFMonitor_counter = Accuracy_counters.DF_Accuracy_Per_Item_Counter(monitored_groups, alpha, threshold, use_two_counters)
    DFMonitor_bit = Accuracy_bit.DF_Accuracy_Per_Item_Bit(monitored_groups, alpha, threshold, use_two_counters)

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
    fixed_window_times = []

    # Traverse through the data
    for index, row in timed_data.iterrows():
        if int(timed_data.loc[index, correctness_column] ) == 1:
            label = "correct"
        else:
            label = "incorrect"
        current_clock = row[date_column]  # Current timestamp
        time_unit_val, time_unit_str = time_unit.split(" ")
        time_unit_val = int(time_unit_val)

        time_diff_units = 0
        if time_unit_str == "second":
            time_diff_units = ((current_clock - last_clock).total_seconds() / time_unit_val)
        elif time_unit_str == "hour":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (3600 * time_unit_val))
        elif time_unit_str == "day":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (86400 * time_unit_val))
        elif time_unit_str == "min":
            time_diff_units = int((current_clock - last_clock).total_seconds() / (60 * time_unit_val))

        # Loop to cover intervals where no data arrives but checks need to be done
        checking_interval_seconds = checking_interval_units * time_unit_val * (
            1 if time_unit_str == "second" else
            60 if time_unit_str == "min" else
            3600 if time_unit_str == "hour" else
            86400)  # Handling different time units for checking intervals

        # Loop to cover intervals where no data arrives but checks need to be done
        while (current_clock - last_accuracy_check).total_seconds() >= checking_interval_seconds:
            fixed_window_times.append(last_accuracy_check)
            print("=============query ============== Performing scheduled accuracy check")
            print("counters", DFMonitor_counter.get_counter_correctness())
            cur_accuracy = DFMonitor_counter.get_accuracy_list()
            accuracy_list.append(cur_accuracy)
            counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
            counter_list_correct.append(counter_corr)
            counter_list_incorrect.append(counter_incorr)
            last_accuracy_check += pd.Timedelta(seconds=checking_interval_seconds)  # Move the accuracy check window forward


            are_close = np.allclose(DFMonitor_counter.get_accuracy_list(), accuracy_list[-1], rtol=1e-07,
                                    atol=1e-08)
            if not are_close:
                raise ValueError(
                    "The accuracy list from DFMonitor_counter is not the same as the one from the traversal")

        # Apply decay and reset after the window size (e.g., after 2 days or any custom window size)
        if time_diff_units >= window_size_units:
            print(f"Window size reached ({window_size_units} units), applying exponential time decay")
            fixed_window_times.append(current_clock)  # Record the time of window reset
            counter_values_incorrect = [x * (alpha ** time_diff_units) for x in counter_values_incorrect]
            counter_values_correct = [x * (alpha ** time_diff_units) for x in counter_values_correct]
            last_clock = current_clock  # Reset the last clock to the current one
            DFMonitor_bit.apply_time_decay(time_diff_units)
            DFMonitor_counter.apply_time_decay(time_diff_units)
            print("time_diff_units", time_diff_units, "counters are decayed by", alpha ** time_diff_units)

        DFMonitor_bit.insert(row, label, time_diff_units)
        DFMonitor_counter.insert(row, label, time_diff_units)
        # DFMonitor_counter.print()

        uf_list.append(copy.deepcopy(DFMonitor_bit.uf))

        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
    # Perform a final accuracy check if the last row doesn't trigger an update
    while (current_clock - last_accuracy_check).total_seconds() >= checking_interval_seconds:
        print("=============query ============== Performing final accuracy check")
        cur_accuracy = DFMonitor_counter.get_accuracy_list()
        accuracy_list.append(cur_accuracy)
        counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
        counter_list_correct.append(counter_corr)
        counter_list_incorrect.append(counter_incorr)
        last_accuracy_check += pd.Timedelta(seconds=checking_interval_seconds)

    return DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, fixed_window_times










