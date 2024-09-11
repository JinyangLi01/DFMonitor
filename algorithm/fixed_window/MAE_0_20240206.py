import sys
import pandas as pd
import numpy as np

class DF_MAE:
    def __init__(self, monitored_groups, alpha, threshold):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.count = np.zeros(len(monitored_groups), dtype=int)
        self.cumulative_error = np.zeros(len(monitored_groups), dtype=float)
        self.uf = np.array([False] * len(monitored_groups), dtype=bool)  # unfair flag array
        self.alpha = alpha
        self.threshold = threshold

    def update_group_metrics(self, group_idx, actual, predicted):
        error = abs(actual - predicted)
        self.cumulative_error[group_idx] += error
        self.count[group_idx] += 1
        # Update unfair flag based on current MAE and threshold
        current_mae = self.cumulative_error[group_idx] / self.count[group_idx]
        self.uf[group_idx] = current_mae > self.threshold

    def find_group_index(self, tuple_):
        for index, row in self.groups.iterrows():
            if all(row[key] == tuple_.get(key, None) for key in row.index):
                return index
        return None

    def insert(self, tuple_, actual, predicted):
        group_idx = self.find_group_index(tuple_)
        if group_idx is not None:
            self.update_group_metrics(group_idx, actual, predicted)

    def new_window(self):
        self.cumulative_error *= self.alpha
        self.count = np.round(self.count * self.alpha).astype(int)

    def print_mae(self):
        mae = np.where(self.count > 0, self.cumulative_error / self.count, 0)
        print("MAE for each group:", mae)
        print("Unfair groups (MAE > threshold):", self.uf)

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.count) + self.count.nbytes
        size += sys.getsizeof(self.cumulative_error) + self.cumulative_error.nbytes
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size
