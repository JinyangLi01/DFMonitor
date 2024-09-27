
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
                                         threshold, alpha, time_unit, checking_interval = 24, label_prediction="predicted",
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

    # Traverse through the data
    for index, row in timed_data.iterrows():
        if int(timed_data.loc[index, correctness_column] ) == 1:
            label = "correct"
        else:
            label = "incorrect"
        current_clock = row[date_column]  # Current timestamp
        time_unit_val, time_unit_str = time_unit.split(" ")
        time_unit_val = int(time_unit_val)
        time_diff = 0
        if time_unit_str == "second":
            time_diff = ((current_clock - last_clock).total_seconds() / time_unit_val)
        elif time_unit_str == "hour":
            time_diff = int((current_clock - last_clock).total_seconds() / (3600 * time_unit_val))
        elif time_unit_str == "day":
            time_diff = int((current_clock - last_clock).total_seconds() / (86400 * time_unit_val))
        elif time_unit_str == "min":
            time_diff = int((current_clock - last_clock).total_seconds() / (60 * time_unit_val))

        # Loop to cover intervals where no data arrives but checks need to be done
        while (current_clock - last_accuracy_check).total_seconds() >= checking_interval * 3600:
            print("=============query ============== Performing scheduled accuracy check")
            print("counters", DFMonitor_counter.get_counter_correctness())
            cur_accuracy = DFMonitor_counter.get_accuracy_list()
            accuracy_list.append(cur_accuracy)
            counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
            counter_list_correct.append(counter_corr)
            counter_list_incorrect.append(counter_incorr)
            last_accuracy_check += pd.Timedelta(hours=checking_interval)  # Move the accuracy check window forward

            are_close = np.allclose(DFMonitor_counter.get_accuracy_list(), accuracy_list[-1], rtol=1e-07,
                                    atol=1e-08)
            if not are_close:
                raise ValueError(
                    "The accuracy list from DFMonitor_counter is not the same as the one from the traversal")

        if time_diff > 0:
            print("time_diff: ", time_diff, "current_clock: ", current_clock, "last_clock: ", last_clock)
            last_clock = current_clock
        # print(index, row, label, current_clock, time_diff)
        # DFMonitor_counter.print()
        # Update DFMonitor_bit and DFMonitor_counter with the new item
        DFMonitor_bit.insert(row, label, time_diff)
        DFMonitor_counter.insert(row, label, time_diff)
        # DFMonitor_counter.print()

        uf_list.append(copy.deepcopy(DFMonitor_bit.uf))
        counter_values_incorrect = [x * (alpha**time_diff) for x in counter_values_incorrect]
        counter_values_correct = [x * (alpha**time_diff) for x in counter_values_correct]
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
    # Perform a final accuracy check if the last row doesn't trigger an update
    while (current_clock - last_accuracy_check).total_seconds() >= checking_interval * 3600:
        print("=============query ============== Performing final accuracy check")
        cur_accuracy = DFMonitor_counter.get_accuracy_list()
        accuracy_list.append(cur_accuracy)
        counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
        counter_list_correct.append(counter_corr)
        counter_list_incorrect.append(counter_incorr)
        last_accuracy_check += pd.Timedelta(hours=checking_interval)

    return DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect










