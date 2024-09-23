
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
                                         threshold, alpha, time_unit, label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",
                                         T_b=100, T_in=10, use_two_counters=True):
    # Initialize DF_Accuracy_Dynamic_Window_Counter objects
    DFMonitor_counter = Accuracy_counters.DF_Accuracy_Dynamic_Window_Counter(monitored_groups, alpha, threshold, T_b, T_in, use_two_counters)
    DFMonitor_bit = Accuracy_bit.DF_Accuracy_Dynamic_Window_Bit(monitored_groups, alpha, threshold, T_b, T_in, use_two_counters)

    # Initialize the time for the first row
    last_clock = timed_data.iloc[0][date_column]

    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []

    # Traverse through the data
    for index, row in timed_data.iterrows():
        if int(timed_data.loc[index, correctness_column]) == 1:
            label = "correct"
        else:
            label = "incorrect"
        current_clock = row[date_column]  # Current timestamp

        if time_unit == "second":
            time_diff = (current_clock - last_clock).total_seconds()
        elif time_unit == "hour":
            time_diff = (current_clock - last_clock).total_seconds() / 3600

        # Update DFMonitor_bit and DFMonitor_counter with the new item
        new_batch = DFMonitor_bit.insert(row, label, time_diff)
        DFMonitor_counter.insert(row, label, time_diff)

        # Append UF list for tracking fairness/unfairness
        uf_list.append(copy.deepcopy(DFMonitor_bit.get_uf_list()))

        if new_batch:
            # Apply time decay to counters
            counter_values_incorrect = [x * (alpha**time_diff) for x in counter_values_incorrect]
            counter_values_correct = [x * (alpha**time_diff) for x in counter_values_correct]

        # Update the correct/incorrect counters based on group membership
        for i, g in enumerate(monitored_groups):
            if DFMonitor_counter.belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1

        # Append accuracy calculations if time has passed
        if new_batch:
            # Calculate accuracy for each group
            accuracy_list.append([counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                                  if counter_values_correct[j] + counter_values_incorrect[j] > 0 else 0
                                  for j in range(len(counter_values_correct))])

            # Compare with DFMonitor_counter accuracy list
            are_close = np.allclose(DFMonitor_counter.get_uf_list(), accuracy_list[-1], rtol=1e-07, atol=1e-08)
            if not are_close:
                raise ValueError("The accuracy list from DFMonitor_counter is not the same as the one from the traversal")

        # Update the last clock for the next iteration
        last_clock = current_clock

    return DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect
