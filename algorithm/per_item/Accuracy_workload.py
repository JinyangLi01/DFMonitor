
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


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="", use_two_counters=True):
    # Initialize DFMonitor counter and bit objects
    DFMonitor_counter = Accuracy_counters.DF_Accuracy_Per_Item_Counter(monitored_groups, alpha, threshold, use_two_counters)
    DFMonitor_bit = Accuracy_bit.DF_Accuracy_Per_Item_Bit(monitored_groups, alpha, threshold, use_two_counters)

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
        if int(timed_data.loc[index, correctness_column] ) == 1:
            label = "correct"
        else:
            label = "incorrect"
        current_clock = row[date_column]  # Current timestamp

        # Calculate time difference in seconds
        time_diff_seconds = (current_clock - last_clock).total_seconds()
        # print(index, row, label, current_clock, time_diff_seconds)
        # DFMonitor_counter.print()
        # Update DFMonitor_bit and DFMonitor_counter with the new item
        DFMonitor_bit.insert(row, label, time_diff_seconds)
        DFMonitor_counter.insert(row, label, time_diff_seconds)
        # DFMonitor_counter.print()

        uf_list.append(copy.deepcopy(DFMonitor_bit.uf))
        counter_values_incorrect = [x * (alpha**time_diff_seconds) for x in counter_values_incorrect]
        counter_values_correct = [x * (alpha**time_diff_seconds) for x in counter_values_correct]
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
        counter_list_correct.append(copy.deepcopy(counter_values_correct))
        counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
        # print(counter_values_correct, counter_values_incorrect)
        accuracy_list.append([counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j]) if counter_values_correct[j] + counter_values_incorrect[j] > 0 else 0
                              for j in range(len(counter_values_correct)) ])
        # Update the last clock to the current one for the next iteration
        last_clock = current_clock
        # print("==================== stream finished. compare the accuracy list with those in DFMonitor_counter =======================\n")
        # print(DFMonitor_counter.get_accuracy_list())
        # print(accuracy_list[-1])
        # compare
        # Use np.allclose() to compare with a default or custom tolerance
        are_close = np.allclose(DFMonitor_counter.get_accuracy_list(), accuracy_list[-1], rtol=1e-07, atol=1e-08)
        if not are_close:
            raise ValueError("The accuracy list from DFMonitor_counter is not the same as the one from the traversal")
    return DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect










