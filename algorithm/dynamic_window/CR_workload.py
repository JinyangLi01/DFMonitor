
import time
import pandas as pd
from algorithm.dynamic_window import CR_bit
from algorithm.dynamic_window import CR_counters
import copy
import numpy as np
from line_profiler_pycharm import profile

def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True

class DynamicWindowManager:
    def __init__(self, T_in, T_b):
        self.T_in = T_in  # Max time between arrivals
        self.T_b = T_b  # Max batch length
        self.current_batch_duration = 0

    def should_renew_window(self, current_time, last_event_time, time_diff_units):
        """
        Determines if a new batch window is needed based on time thresholds.
        """
        if time_diff_units >= self.T_in or self.current_batch_duration + time_diff_units >= self.T_b:
            self.current_batch_duration = 0  # Reset batch duration
            return True
        self.current_batch_duration += time_diff_units
        return False

@profile
def traverse_data_DFMonitor_and_baseline_CR(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, T_b, T_in,
                                         checking_interval="1 hour",
                                         label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",
                                         use_two_counters=True, use_nanosecond=False):
    # Initialize DF_CoverageRatio_Dynamic_Window_Counter objects
    print("monitored_groups", monitored_groups)

    DFMonitor_counter = CR_counters.DF_CoverageRatio_Dynamic_Window_Counter(
        monitored_groups, alpha, threshold, T_b, T_in
    )
    DFMonitor_bit = CR_bit.DF_CoverageRatio_Dynamic_Window_Bit(
        monitored_groups, alpha, threshold, T_b, T_in
    )

    window_manager = DynamicWindowManager(T_in, T_b)
    last_event_time = None

    check_points = []
    last_clock = timed_data.iloc[0][date_column]
    last_CR_check = last_clock
    coverage_ratios_list = []
    counter_list_group = []
    counter_list_total = []
    uf_list = []

    window_reset_times = []
    checking_interval_seconds = 0
    checking_interval_nanosecond = 0
    current_clock = 0

    # Convert time unit and checking interval to appropriate scale
    time_unit_val, time_unit_str = time_unit.split(" ")
    time_unit_val = int(time_unit_val)
    time_unit_seconds = 0
    time_unit_nanosecond = 0
    elapsed_time_DFMonitor_counter = 0
    elapsed_time_DFMonitor_bit = 0
    total_time_new_window_bit = 0
    total_time_new_window_counter = 0
    total_time_insertion_bit = 0
    total_time_insertion_counter = 0
    total_num_CR_check = 0
    total_time_query_bit = 0
    total_time_query_counter = 0
    total_num_time_window = 0

    if use_nanosecond:
        time_unit_nanosecond = time_unit_val * (
            1 if time_unit_str == "nanosecond" else
            1000 if time_unit_str == "microsecond" else
            1_000_000 if time_unit_str == "millisecond" else
            1_000_000_000 if time_unit_str == "second" else
            60 * 1_000_000_000 if time_unit_str == "min" else
            3600 * 1_000_000_000 if time_unit_str == "hour" else
            86400 * 1_000_000_000 if time_unit_str == "day" else 0
        )
    else:
        time_unit_seconds = time_unit_val * (
            1 if time_unit_str == "second" else
            60 if time_unit_str == "min" else
            3600 if time_unit_str == "hour" else
            86400 if time_unit_str == "day" else 0
        )

    checking_interval_units_num = int(checking_interval.split(" ")[0])
    checking_interval_units_str = checking_interval.split(" ")[1]

    if use_nanosecond:
        checking_interval_nanosecond = checking_interval_units_num * (
            1 if checking_interval_units_str == "nanosecond" else
            1000 if checking_interval_units_str == "microsecond" else
            1_000_000 if checking_interval_units_str == "millisecond" else
            1_000_000_000 if checking_interval_units_str == "second" else
            60 * 1_000_000_000 if checking_interval_units_str == "min" else
            3600 * 1_000_000_000 if checking_interval_units_str == "hour" else
            86400 * 1_000_000_000 if checking_interval_units_str == "day" else 0
        )
    else:
        checking_interval_seconds = checking_interval_units_num * (
            1 if checking_interval_units_str == "second" else
            60 if checking_interval_units_str == "min" else
            3600 if checking_interval_units_str == "hour" else
            86400 if checking_interval_units_str == "day" else 0
        )

    for index, row in timed_data.iterrows():
        if index % 100 == 0:
            print(f"Processing row {index} out of {len(timed_data)}")

        current_clock = row[date_column]

        if use_nanosecond:
            time_diff_units = (current_clock - last_clock).total_seconds() * 1e9 / time_unit_nanosecond
        else:
            time_diff_units = (current_clock - last_clock).total_seconds() / time_unit_seconds

        while (current_clock - last_CR_check).total_seconds() * (1e9 if use_nanosecond else 1) >= (
                checking_interval_nanosecond if use_nanosecond else checking_interval_seconds):
            check_points.append(last_CR_check)
            time1 = time.time()
            cur_uf_bit = DFMonitor_counter.get_uf_list()
            time2 = time.time()
            total_time_query_counter += time2 - time1

            time1 = time.time()
            cur_uf_bit = DFMonitor_bit.get_uf_list()
            time2 = time.time()
            total_time_query_bit += time2 - time1
            uf_list.append(copy.deepcopy(cur_uf_bit))
            cur_cr = DFMonitor_counter.get_coverage_ratio_list()
            coverage_ratios_list.append(copy.deepcopy(cur_cr))
            cur_cr = DFMonitor_counter.get_coverage_ratio_list()

            last_CR_check += pd.Timedelta(
                nanoseconds=checking_interval_nanosecond) if use_nanosecond else pd.Timedelta(
                seconds=checking_interval_seconds)
            total_num_CR_check += 1

        last_clock = current_clock
        if last_event_time and window_manager.should_renew_window(current_clock, last_event_time, time_diff_units):
            time1 = time.time()
            DFMonitor_counter.batch_update()
            time2 = time.time()
            DFMonitor_bit.batch_update()
            time3 = time.time()
            elapsed_time_DFMonitor_counter += time2 - time1
            total_time_new_window_counter += time2 - time1
            elapsed_time_DFMonitor_bit += time3 - time2
            total_time_new_window_bit += time3 - time2

            window_reset_times.append(current_clock)
            total_num_time_window += 1
        last_event_time = current_clock

        time1 = time.time()
        DFMonitor_bit.insert(row, time_diff_units)
        time2 = time.time()
        DFMonitor_counter.insert(row, time_diff_units)
        time3 = time.time()
        elapsed_time_DFMonitor_bit += time2 - time1
        total_time_insertion_bit += time2 - time1
        total_time_insertion_counter += time3 - time2
        elapsed_time_DFMonitor_counter += time3 - time2

    print("elapsed_time_DFMonitor_bit", elapsed_time_DFMonitor_bit)
    print("elapsed_time_DFMonitor_counter", elapsed_time_DFMonitor_counter)
    print("total_time_new_window_bit", total_time_new_window_bit)
    print("total_time_new_window_counter", total_time_new_window_counter)
    print("total_time_insertion_bit", total_time_insertion_bit)
    print("total_time_insertion_counter", total_time_insertion_counter)

    return (
        DFMonitor_bit, DFMonitor_counter, uf_list, coverage_ratios_list, counter_list_group, counter_list_total,
        window_reset_times, check_points,
        elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, total_time_new_window_bit,
        total_time_new_window_counter, total_time_insertion_bit, total_time_insertion_counter,
        total_num_CR_check, total_time_query_bit, total_time_query_counter, total_num_time_window
    )
