import time

import pandas as pd
from tensorflow.python.tools.optimize_for_inference_lib import flags

from algorithm.per_item import Accuracy_bit
from algorithm.per_item import Accuracy_counters
import copy
import numpy as np
from line_profiler_pycharm import profile

def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


# time_unit = 10 hour
@profile
def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, window_size_units, checking_interval="1 hour",
                                         label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",use_nanosecond=False,
                                         use_two_counters=True):
    print("time unit", time_unit, "window_size_units", window_size_units, "checking_interval", checking_interval)
    # Initialize DFMonitor counter and bit objects
    DFMonitor_counter = Accuracy_counters.DF_Accuracy_Per_Item_Counter(monitored_groups, alpha, threshold,
                                                                       use_two_counters)
    DFMonitor_bit = Accuracy_bit.DF_Accuracy_Per_Item_Bit(monitored_groups, alpha, threshold, use_two_counters)

    check_points = []

    # Initialize the time for the first row
    last_clock = timed_data.iloc[0][date_column]
    last_accuracy_check = last_clock
    window_reset_time = last_clock

    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    current_clock = 0
    window_reset_time_list = []

    elapsed_time_DFMonitor_counter = 0
    elapsed_time_DFMonitor_bit = 0
    total_time_new_window_bit = 0
    total_time_new_window_counter = 0
    total_time_insertion_bit = 0
    total_time_insertion_counter = 0

    total_num_time_window = 0
    total_num_accuracy_check = 0
    total_time_query_bit = 0
    total_time_query_counter = 0
    # Define durations in nanoseconds or seconds depending on use_nanosecond flag

    # Determine time unit and checking interval duration
    time_unit_val, time_unit_str = time_unit.split(" ")
    time_unit_val = int(time_unit_val)
    checking_interval_units_num, checking_interval_units_str = checking_interval.split(" ")
    checking_interval_units_num = int(checking_interval_units_num)

    if use_nanosecond:
        time_unit_duration = (
                    time_unit_val * (1 if time_unit_str == "nanosecond" else 1000 if time_unit_str == "microsecond"
            else 1000000 if time_unit_str == "millisecond" else 1000000000 if time_unit_str == "second"
            else 60000000000 if time_unit_str == "min" else 3600000000000 if time_unit_str == "hour"
            else 86400000000000 if time_unit_str == "day" else 0))
        checking_interval_duration = (checking_interval_units_num * (1 if checking_interval_units_str == "nanosecond"
                                                                     else 1000 if checking_interval_units_str == "microsecond"
        else 1000000 if checking_interval_units_str == "millisecond"
        else 1000000000 if checking_interval_units_str == "second" else 60000000000 if checking_interval_units_str == "min"
        else 3600000000000 if checking_interval_units_str == "hour" else 86400000000000 if checking_interval_units_str == "day" else 0))
    else:
        time_unit_duration = (time_unit_val * (1 if time_unit_str == "second" else 60 if time_unit_str == "min"
        else 3600 if time_unit_str == "hour" else 86400 if time_unit_str == "day" else 0))
        checking_interval_duration = (checking_interval_units_num * (
            1 if checking_interval_units_str == "second" else 60 if checking_interval_units_str == "min"
            else 3600 if checking_interval_units_str == "hour" else 86400 if checking_interval_units_str == "day" else 0))


    # Traverse through the data
    for index, row in timed_data.iterrows():
        if index % 100 == 0:
            print("index", index, "datetime", row[date_column])
        current_clock = row[date_column]  # Current timestamp
        flags = False
        for g in monitored_groups:
            if belong_to_group(row, g):
                flags = True
                break
        if not flags:
            continue
        time_decay_moment = False
        # Apply time decay at every elapsed time window
        while (current_clock - window_reset_time).total_seconds() * (
        1e9 if use_nanosecond else 1) >= window_size_units * time_unit_duration:
            time_decay_moment = True
            # print("before time decay: ", counter_values_correct, counter_values_incorrect)
            # Apply exponential decay
            for i in range(len(monitored_groups)):
                counter_values_incorrect[i] *= alpha ** window_size_units
                counter_values_correct[i] *= alpha ** window_size_units
            window_reset_time_list.append(window_reset_time)
            # Update window reset time
            window_reset_time += pd.Timedelta(
                nanoseconds=(window_size_units * time_unit_duration)) if use_nanosecond else pd.Timedelta(
                seconds=(window_size_units * time_unit_duration))

            # Apply decay in DFMonitor objects
            time1_DFMonitor_counter = time.time()
            DFMonitor_counter.apply_time_decay(1)
            time2_DFMonitor_counter = time.time()
            elapsed_time_DFMonitor_counter += time2_DFMonitor_counter - time1_DFMonitor_counter
            total_time_new_window_counter += time2_DFMonitor_counter - time1_DFMonitor_counter

            time1_DFMonitor_bit = time.time()
            DFMonitor_bit.apply_time_decay(1)
            time2_DFMonitor_bit = time.time()
            elapsed_time_DFMonitor_bit += time2_DFMonitor_bit - time1_DFMonitor_bit
            total_time_new_window_bit += time2_DFMonitor_bit - time1_DFMonitor_bit

            total_num_time_window += 1
        # Scheduled accuracy checks
        # print(current_clock, (current_clock - last_accuracy_check).total_seconds())
        while (current_clock - last_accuracy_check).total_seconds() * (
        1e9 if use_nanosecond else 1) >= checking_interval_duration:
            check_points.append(last_accuracy_check)
            time1 = time.time()
            cur_uf = DFMonitor_bit.get_uf_list()
            time2 = time.time()
            total_time_query_counter += time2 - time1

            time1 = time.time()
            cur_uf = DFMonitor_bit.get_uf_list()
            time2 = time.time()
            total_time_query_bit += time2 - time1

            cur_accuracy = DFMonitor_counter.get_accuracy_list()
            accuracy_list.append(cur_accuracy)
            counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
            counter_list_correct.append(copy.deepcopy(counter_corr))
            counter_list_incorrect.append(copy.deepcopy(counter_incorr))
            last_accuracy_check += pd.Timedelta(
                nanoseconds=checking_interval_duration) if use_nanosecond else pd.Timedelta(
                seconds=checking_interval_duration)
            total_num_accuracy_check += 1
            uf_list.append(copy.deepcopy(DFMonitor_bit.uf))


        # Insert the current row into DFMonitor
        label = "correct" if int(timed_data.loc[index, correctness_column]) == 1 else "incorrect"
        # Insert the current row into DFMonitor
        time1_DFMonitor_bit = time.time()
        DFMonitor_bit.insert(row, label, 0)  # time_diff_units is set to 0 as decay is handled separately
        time2_DFMonitor_bit = time.time()
        elapsed_time_DFMonitor_bit += time2_DFMonitor_bit - time1_DFMonitor_bit
        total_time_insertion_bit += time2_DFMonitor_bit - time1_DFMonitor_bit

        time1_DFMonitor_counter = time.time()
        DFMonitor_counter.insert(row, label, 0)
        time2_DFMonitor_counter = time.time()
        elapsed_time_DFMonitor_counter += time2_DFMonitor_counter - time1_DFMonitor_counter
        total_time_insertion_counter += time2_DFMonitor_counter - time1_DFMonitor_counter

        # Update counters for correct/incorrect values for each group
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == "correct":
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1

    # Final accuracy check if the last row did not trigger one
    while (current_clock - last_accuracy_check).total_seconds() * (
    1e9 if use_nanosecond else 1) >= checking_interval_duration:
        time1 = time.time()
        cur_uf = DFMonitor_counter.get_uf_list()
        time2 = time.time()
        total_time_query_counter += time2 - time1

        time1 = time.time()
        cur_uf = DFMonitor_bit.get_uf_list()
        time2 = time.time()
        total_time_query_bit += time2 - time1

        cur_accuracy = DFMonitor_counter.get_accuracy_list()
        accuracy_list.append(cur_accuracy)
        counter_corr, counter_incorr = DFMonitor_counter.get_counter_correctness()
        counter_list_correct.append(copy.deepcopy(counter_corr))
        counter_list_incorrect.append(copy.deepcopy(counter_incorr))
        last_accuracy_check += pd.Timedelta(
            nanoseconds=checking_interval_duration) if use_nanosecond else pd.Timedelta(
            seconds=checking_interval_duration)
        total_num_accuracy_check += 1

    return (DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect,
            window_reset_time_list, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter,
            total_time_new_window_bit, total_time_new_window_counter,
            total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,
            total_time_query_bit, total_time_query_counter, total_num_time_window)