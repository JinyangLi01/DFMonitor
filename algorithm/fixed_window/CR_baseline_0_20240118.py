import sys

import pandas as pd
import numpy as np


class CR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False]*len(monitored_groups), dtype=bool)
        self.counters = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.counter_total = 0
        self.threshold = threshold
        self.alpha = alpha


    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("counters", self.counters)
        print("counter_total", self.counter_total)


    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.counters) + self.counters.nbytes
        size += sys.getsizeof(self.counter_total)
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size


    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def insert(self, tuple_):
        self.counter_total += 1
        for group_idx in self.groups.index:
            row = self.groups.loc[group_idx]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                self.counters[group_idx] += 1
            if self.counters[group_idx] / self.counter_total <= self.threshold:
                self.uf[group_idx] = True
            else:
                self.uf[group_idx] = False


    def new_window(self):
        self.counter_total *= self.alpha
        self.counters = self.counters * self.alpha

