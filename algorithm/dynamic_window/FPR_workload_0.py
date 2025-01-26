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
    def __init__(self, T_in=None, T_b=None, start_date=None, end_date=None, mechanism="fixed"):
        """
        Initializes the DynamicWindowManager.

        Parameters:
        - T_in: Maximum allowed time difference between events (for fixed duration).
        - T_b: Maximum batch length (for fixed duration).
        - start_date: Start date of the dataset (for monthly mechanism).
        - end_date: End date of the dataset (for monthly mechanism).
        - mechanism: "fixed" for T_in/T_b-based windows, "monthly" for calendar-based windows.
        """
        self.mechanism = mechanism

        if mechanism == "fixed":
            self.T_in = T_in
            self.T_b = T_b
            self.current_batch_duration = 0
        elif mechanism == "monthly":
            if not start_date or not end_date:
                raise ValueError("Start and end dates must be provided for the 'monthly' mechanism.")
            self.monthly_boundaries = pd.date_range(start=start_date, end=end_date, freq="MS").tolist()
            self.current_month_index = 0
            print("Monthly boundaries:", self.monthly_boundaries)

    def should_renew_window(self, current_time, last_event_time=None, time_diff_units=None):
        """
        Determines if a new batch window is needed.

        Parameters:
        - current_time: Current timestamp.
        - last_event_time: Timestamp of the last event (for fixed mechanism).
        - time_diff_units: Difference in time units between events (for fixed mechanism).

        Returns:
        - True if a new window is needed; False otherwise.
        """
        if self.mechanism == "fixed":
            if time_diff_units >= self.T_in or self.current_batch_duration + time_diff_units >= self.T_b:
                self.current_batch_duration = 0  # Reset batch duration
                return True
            self.current_batch_duration += time_diff_units
            return False

        elif self.mechanism == "monthly":
            if self.current_month_index < len(self.monthly_boundaries) - 1:
                next_boundary = self.monthly_boundaries[self.current_month_index + 1]
                if current_time >= next_boundary:
                    self.current_month_index += 1
                    return True
            return False


@profile
def traverse_data_DFMonitor_and_baseline(timed_data, date_column, date_time_format, monitored_groups,
                                         threshold, alpha, time_unit, T_b, T_in,
                                         checking_interval="1 hour", mechanism="fixed", start_date=None, end_date=None,
                                         label_prediction="predicted",
                                         label_ground_truth="ground_truth", correctness_column="",
                                         use_two_counters=False, use_nanosecond=False):
    # Initialize DF_Accuracy_Dynamic_Window_Counter objects
    print("monitored_groups", monitored_groups)
    if mechanism == "fixed":
        window_manager = DynamicWindowManager(T_in=T_in, T_b=T_b, mechanism="fixed")
    elif mechanism == "monthly":
        window_manager = DynamicWindowManager(start_date=start_date, end_date=end_date, mechanism="monthly")
    else:
        raise ValueError("Invalid mechanism type. Use 'fixed' or 'monthly'.")


    DFMonitor_counter = FPR_counters.DF_FPR_Dynamic_Window_Counter(
        monitored_groups, alpha, threshold, T_b, T_in, use_two_counters
    )
    DFMonitor_bit = FPR_bit.DF_FPR_Dynamic_Window_Bit(
        monitored_groups, alpha, threshold, T_b, T_in, use_two_counters
    )
    # Generate monthly checkpoints: 1st day of each month
    monthly_checkpoints = pd.date_range(start="2013-01-01", end="2014-12-31", freq="MS").tolist()

    last_event_time = None
    check_points = []
    last_clock = timed_data.iloc[0][date_column]
    last_accuracy_check = last_clock
    window_reset_times = []

    checking_interval_units_num = int(checking_interval.split(" ")[0])
    checking_interval_units_str = checking_interval.split(" ")[1]
    checking_interval_seconds = checking_interval_units_num * {
        "second": 1,
        "min": 60,
        "hour": 3600,
        "day": 86400
    }.get(checking_interval_units_str, 0)

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
    total_num_accuracy_check = 0
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


    for index, row in timed_data.iterrows():
        # print(f"{row}")
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

        # Determine if the window should be renewed
        time_diff_units = (current_time - last_clock).total_seconds()
        if window_manager.should_renew_window(current_time, last_event_time, time_diff_units):
            DFMonitor_counter.batch_update()
            DFMonitor_bit.batch_update()
            window_reset_times.append(current_time)

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
    current_time = last_clock
    # Final accuracy check
    for checkpoint in monthly_checkpoints:
        if checkpoint > last_accuracy_check:
            check_points.append(checkpoint)
            time1 = time.time()
            cur_uf_counter = DFMonitor_counter.get_uf_list()
            time2 = time.time()
            total_time_query_counter += time2 - time1

            time1 = time.time()
            cur_uf_bit = DFMonitor_bit.get_uf_list()
            time2 = time.time()
            total_time_query_bit += time2 - time1
            cur_fpr = DFMonitor_counter.get_FPR_list()
            fpr_list.append(cur_fpr)
            uf_list.append(copy.deepcopy(cur_uf_bit))
            counter_fp, counter_fn, counter_tp, counter_tn = DFMonitor_counter.get_counter_fp_fn_tp_tn()
            counter_list_fp.append(copy.deepcopy(counter_fp))
            counter_list_fn.append(copy.deepcopy(counter_fn))
            counter_list_tp.append(copy.deepcopy(counter_tp))
            counter_list_tn.append(copy.deepcopy(counter_tn))
            last_accuracy_check = checkpoint


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
