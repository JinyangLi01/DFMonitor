import sys
import numpy as np
import pandas as pd
from line_profiler_pycharm import profile

class DF_CoverageRatio_Dynamic_Window_Counter:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in):
        self.groups = pd.DataFrame(monitored_groups).astype('category')
        self.alpha = alpha
        self.threshold = int(threshold * 10000)  # Scale threshold by 10000
        self.T_b = T_b  # Maximum batch size (time duration limit)
        self.T_in = T_in  # Maximum time interval between items

        num_groups = len(monitored_groups)
        self.counters = np.array([0] * len(monitored_groups), dtype=np.float64)
        self.counter_total = 0

        self.uf = np.zeros(num_groups, dtype=bool)  # Fair/unfair flag
        # Initialize the three timers
        self.Delta_a = 0  # Tracks time between the midpoint of the last batch and the start of the latest batch
        self.Delta_b = 0  # Tracks the time duration of the current batch
        self.Delta_in = 0  # Tracks the time interval since the arrival of the last item
        self.last_item_time = 0  # Track the time of the last inserted item
        self.current_time = 0  # Current system time, incremented per item
        self.current_batch_size = 0  # Tracks the size of the current batch

    def get_coverage_ratio_list(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            coverage_ratios = np.divide(self.counters, self.counter_total, where=self.counter_total > 0)
        return coverage_ratios.tolist()

    def initialization(self, uf, counters=None, counter_total=None):
        self.uf = np.array(uf, dtype=bool)
        self.counters = np.array(counters, dtype=int)
        self.counter_total = np.sum(self.counters)

    def get_uf_of_group(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = (sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() +
                + sys.getsizeof(self.uf) + self.uf.nbytes + sys.getsizeof(self.threshold) + sys.getsizeof(self.alpha) +
                sys.getsizeof(self.counters) + self.counters.nbytes +
                sys.getsizeof(self.counter_total) +
                sys.getsizeof(self.T_b) + sys.getsizeof(self.T_in) +
                sys.getsizeof(self.Delta_a) + sys.getsizeof(self.Delta_b) +
                sys.getsizeof(self.Delta_in) + sys.getsizeof(self.last_item_time) +
                sys.getsizeof(self.current_time) + sys.getsizeof(self.current_batch_size))
        return size

    @profile
    def whether_need_batch_renewal(self, time_interval):
        self.Delta_in = time_interval
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()
            return True
        return False

    @profile
    def insert(self, tuple_, time_interval, new_batch=False):
        self.Delta_in = time_interval
        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]
        try:
            series = self.groups[col_name]
            index = series[series == col_value].index[0]
        except KeyError:
            return  # If group not found, exit

        self.counters[index] += 10000
        self.counter_total += 10000
        cr = self.counters[index] / self.counter_total if self.counter_total > 0 else 0
        self.uf[index] = cr >= self.threshold

        if not new_batch:
            self.Delta_b += self.Delta_in
            self.Delta_in = 0
            self.current_batch_size += 1
            self.last_item_time = self.current_time

    def np_list_scale(self, list, time_decay_factor):
        list = np.array(list, dtype=np.float64)
        list *= time_decay_factor
        list = np.rint(list).astype(np.int32)  # Use rounding instead of truncating
        return list

    @profile
    def batch_update(self):
        time_decay_factor = (self.alpha) ** (self.Delta_a + (self.Delta_b / 2))
        self.counters = self.np_list_scale(self.counters, time_decay_factor)
        self.counter_total = int(np.sum(self.counters))

        self.Delta_a = (self.Delta_b / 2) + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0

    @profile
    def get_counters(self):
        return self.counters.tolist()

    @profile
    def get_counter_total(self):
        return self.counter_total

    def get_uf_list(self):
        return self.uf.tolist()
