import time
import pandas as pd
from algorithm.dynamic_window import FPR_bit
from algorithm.dynamic_window import FPR_counters
import copy
import numpy as np
from line_profiler_pycharm import profile



def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True

class DynamicWindowManager:
    def __init__(self, T_in=None, T_b=None, time_unit=1, use_nanosecond=False):
        """
        Initializes the DynamicWindowManager.

        Parameters:
        - T_in: Maximum allowed time difference between events, in units.
        - T_b: Maximum batch length, in units.
        - time_unit: Base unit of time (e.g., seconds, nanoseconds).
        - use_nanosecond: Whether to use nanoseconds for time calculations.
        """
        self.T_in = T_in
        self.T_b = T_b
        self.time_unit = time_unit
        self.use_nanosecond = use_nanosecond
        self.current_batch_duration = 0

    def should_renew_window(self, time_diff_units):
        if time_diff_units >= self.T_in or self.current_batch_duration + time_diff_units >= self.T_b:
            self.current_batch_duration = 0  # Reset batch duration
            return True
        self.current_batch_duration += time_diff_units
        return False


@profile
def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, T_b, T_in,
                                         checking_interval="1 hour",
                                         label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",
                                         use_two_counters=False, use_nanosecond=False, start_date=None, end_date=None):
    # Initialize DF_Accuracy_Dynamic_Window_Counter objects
    print("monitored_groups", monitored_groups)
    # Convert time_unit and checking_interval to appropriate scale
    time_unit_val, time_unit_str = time_unit.split(" ")
    time_unit_val = int(time_unit_val)

    if use_nanosecond:
        base_conversion = {
            "nanosecond": 1,
            "microsecond": 1_000,
            "millisecond": 1_000_000,
            "second": 1_000_000_000,
            "min": 60 * 1_000_000_000,
            "hour": 3600 * 1_000_000_000,
            "day": 86400 * 1_000_000_000,
        }
    else:
        base_conversion = {
            "second": 1,
            "min": 60,
            "hour": 3600,
            "day": 86400,
        }

    time_unit_scale = base_conversion.get(time_unit_str, 1) * time_unit_val
    checking_interval_val, checking_interval_str = checking_interval.split(" ")
    checking_interval_val = int(checking_interval_val)
    checking_interval_scale = base_conversion.get(checking_interval_str, 1) * checking_interval_val

    window_manager = DynamicWindowManager(
        T_in=T_in, T_b=T_b, time_unit=time_unit_scale, use_nanosecond=use_nanosecond
    )

    DFMonitor_counter = FPR_counters.DF_FPR_Dynamic_Window_Counter(
        monitored_groups, alpha, threshold, T_b, T_in, use_two_counters
    )
    DFMonitor_bit = FPR_bit.DF_FPR_Dynamic_Window_Bit(
        monitored_groups, alpha, threshold, T_b, T_in, use_two_counters
    )
    # Generate monthly checkpoints: 1st day of each month
    if start_date and end_date:
        monthly_checkpoints = pd.date_range(start=start_date, end=end_date, freq="MS").tolist()
    else:
        raise ValueError("start_date and end_date are required for monthly accuracy checks.")


    check_points = []
    last_clock = timed_data.iloc[0][date_column]
    last_accuracy_check = last_clock
    window_reset_times = []
    last_event_time = last_clock

    counter_values_FP = [0 for _ in monitored_groups]
    counter_values_TN = [0 for _ in monitored_groups]
    counter_values_TP = [0 for _ in monitored_groups]
    counter_values_FN = [0 for _ in monitored_groups]
    counter_list_fp = []  # record each time window's counter_values
    counter_list_tn = []
    counter_list_tp = []
    counter_list_fn = []
    uf_list = []
    fpr_list = []
    checking_interval_nanosecond = 0
    current_clock = 0
    current_time = 0

    time_unit_seconds = 0
    time_unit_nanosecond = 0
    elapsed_time_DFMonitor_counter = 0
    elapsed_time_DFMonitor_bit = 0
    total_time_new_window_bit = 0
    total_time_new_window_counter = 0
    total_time_insertion_bit = 0
    total_time_insertion_counter = 0
    total_num_accuracy_check = 0
    total_time_query_bit = 0
    total_time_query_counter = 0
    total_num_time_window = 0


    for index, row in timed_data.iterrows():
        if index > 6210:
            print("stop\n")
        if index % 100 == 0:
            print(f"Processing row {index} out of {len(timed_data)}")
        label = "FP"
        if ((row[label_prediction] == 1 and row[label_ground_truth] == 0) or
                (row[label_prediction] == 0 and row[label_ground_truth] == 0)):
            if row[label_prediction] == 1 and row[label_ground_truth] == 0:
                label = "FP"
            else:
                label = "TN"
        else:
            continue
        current_time = row[date_column]
        # Perform accuracy checks at regular intervals
        # Iterate through predefined monthly checkpoints
        for checkpoint in monthly_checkpoints:
            while current_time >= checkpoint and checkpoint > last_accuracy_check:
                print("checkpoint= ", checkpoint)
                check_points.append(checkpoint)
                time1 = time.time()
                cur_uf_counter = DFMonitor_counter.get_uf_list()
                time2 = time.time()
                total_time_query_counter += time2 - time1

                time1 = time.time()
                cur_uf_bit = DFMonitor_bit.get_uf_list()
                time2 = time.time()
                total_time_query_bit += time2 - time1
                fpr_list.append(DFMonitor_counter.get_FPR_list())
                uf_list.append(copy.deepcopy(cur_uf_bit))
                counter_fp, counter_fn, counter_tp, counter_tn = DFMonitor_counter.get_counter_fp_fn_tp_tn()
                counter_list_fp.append(copy.deepcopy(counter_fp))
                counter_list_fn.append(copy.deepcopy(counter_fn))
                counter_list_tp.append(copy.deepcopy(counter_tp))
                counter_list_tn.append(copy.deepcopy(counter_tn))
                # Move to the next checkpoint
                last_accuracy_check = checkpoint

        # Calculate time difference in units
        time_diff = (current_time - last_event_time).total_seconds() if not use_nanosecond else (
                current_time - last_event_time).nanoseconds
        time_diff_units = time_diff / time_unit_scale
        if time_diff > 0:
            print("stop\n")

        # Determine if a new window is needed
        if window_manager.should_renew_window(time_diff_units):
            DFMonitor_counter.batch_update()
            DFMonitor_bit.batch_update()
            window_reset_times.append(current_time)
            total_num_time_window += 1

        last_event_time = current_time
        last_clock = current_time
        # Insert data into monitors
        time1 = time.time()
        DFMonitor_bit.insert(row, label, time_diff_units)
        time2 = time.time()
        DFMonitor_counter.insert(row, label, time_diff_units)
        time3 = time.time()

        elapsed_time_DFMonitor_bit += time2 - time1
        total_time_insertion_bit += time2 - time1
        total_time_insertion_counter += time3 - time2
        elapsed_time_DFMonitor_counter += time3 - time2

        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == "FP":
                    counter_values_FP[i] += 1
                elif label == "FN":
                    counter_values_FN[i] += 1
                elif label == "TP":
                    counter_values_TP[i] += 1
                elif label == "TN":
                    counter_values_TN[i] += 1

    # Accuracy checks at predefined monthly checkpoints
    for checkpoint in monthly_checkpoints:
        while current_time >= checkpoint > last_accuracy_check:
            check_points.append(checkpoint)
            last_accuracy_check = checkpoint

            # Perform accuracy check
            time1 = time.time()
            cur_uf_counter = DFMonitor_counter.get_uf_list()
            time2 = time.time()
            elapsed_time_DFMonitor_counter += time2 - time1

            time1 = time.time()
            cur_uf_bit = DFMonitor_bit.get_uf_list()
            time2 = time.time()
            elapsed_time_DFMonitor_bit += time2 - time1

            print(f"Accuracy checked at {checkpoint}")


    print("elapsed_time_DFMonitor_bit", elapsed_time_DFMonitor_bit)
    print("elapsed_time_DFMonitor_counter", elapsed_time_DFMonitor_counter)
    print("total_time_new_window_bit", total_time_new_window_bit)
    print("total_time_new_window_counter", total_time_new_window_counter)
    print("total_time_insertion_bit", total_time_insertion_bit)
    print("total_time_insertion_counter", total_time_insertion_counter)
    print("new window + insertion bit", total_time_new_window_bit + total_time_insertion_bit)
    print("new window + insertion counter", total_time_new_window_counter + total_time_insertion_counter)

    return (
        DFMonitor_bit, DFMonitor_counter, uf_list, fpr_list, counter_list_fp, counter_list_fn,
        counter_list_tp, counter_list_tn,
        window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter,
        total_time_new_window_bit, total_time_new_window_counter,
        total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,
        total_time_query_bit, total_time_query_counter, total_num_time_window)
