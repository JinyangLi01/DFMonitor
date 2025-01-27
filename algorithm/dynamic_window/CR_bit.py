import sys
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile


class DF_CoverageRatio_Dynamic_Window_Bit:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.coverage_ratios = np.zeros(len(monitored_groups), dtype=float)  # Initialize coverage ratios
        self.total_count = 0  # Total number of items
        self.alpha = alpha  # Decay factor
        self.threshold = int(threshold * 10000)  # Scale threshold by 10000
        self.T_b = T_b  # Maximum batch size (time duration limit)
        self.T_in = T_in  # Maximum time interval between items

        num_groups = len(monitored_groups)
        self.uf = np.zeros(num_groups, dtype=bool)
        self.delta = np.array([0] * len(monitored_groups), dtype=int)

        # Initialize timers
        self.Delta_a = 0
        self.Delta_b = 0
        self.Delta_in = 0
        self.last_item_time = 0
        self.current_time = 0
        self.current_batch_size = 0

    # def initialization(self, group_counters=None):
    #     self.coverage_ratios = np.array(coverage_ratios, dtype=float)
    #     self.total_count = total_count
    #     self.counter_group = np.array([coverage_ratios], dtype=int)


    def get_uf_of_group(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]


    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(
            deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.coverage_ratios) + self.coverage_ratios.nbytes
        size += sys.getsizeof(self.total_count)
        size += sys.getsizeof(self.alpha)
        return size



    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def whether_need_batch_renewal(self, time_interval):
        self.Delta_in = time_interval
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()
            return True
        return False


    def insert(self, tuple_, time_interval, new_batch=False):
        self.Delta_in = time_interval
        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]
        try:
            series = self.groups[col_name]
            group_idx = series[series == col_value].index[0]
        except KeyError:
            return
        # Perform operations for the matched group
        if not self.uf[group_idx]:
            self.delta[group_idx] += 1 - self.threshold
        else:
            if self.delta[group_idx] >= 1 - self.threshold:
                self.delta[group_idx] -= 1 - self.threshold
            else:
                self.delta[group_idx] = 1 - self.threshold - self.delta[group_idx]
                self.uf[group_idx] = False

        if not new_batch:
            self.Delta_b += self.Delta_in
            self.Delta_in = 0
            self.current_batch_size += 1


    def batch_update(self):
        # Compute time decay factor
        decay_exponent = self.Delta_a + (self.Delta_b / 2)
        time_decay_factor = (self.alpha) ** decay_exponent

        # Apply decay
        self.coverage_ratios *= time_decay_factor
        self.total_count *= time_decay_factor

        # Reset timers for the new batch
        self.Delta_a = self.Delta_b / 2 + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0



    def get_uf_list(self):
        return self.uf.tolist()