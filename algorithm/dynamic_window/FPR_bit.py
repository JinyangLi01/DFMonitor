import sys

import numpy as np
import pandas as pd


class DF_FPR_Dynamic_Window_Bit:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in, use_two_counters=True):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        self.use_two_counters = use_two_counters
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False] * len(monitored_groups), dtype=bool)
        self.delta = np.array([0] * len(monitored_groups), dtype=int)
        self.alpha = alpha
        self.threshold = int(threshold * 10000)
        self.T_b = T_b
        self.T_in = T_in

        self.Delta_a = 0
        self.Delta_b = 0
        self.Delta_in = 0
        self.last_item_time = 0
        self.current_time = 0
        self.current_batch_size = 0

    def initialization(self, uf, delta):
        self.uf = np.array(uf, dtype=bool)
        self.delta = np.array([int(d * 10000) for d in delta], dtype=int)

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(
            deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.delta) + self.delta.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

    def print(self):
        print("uf", self.uf)
        # Convert `delta` back to floating-point for printing
        print("delta", self.delta / 10000)

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

    def insert(self, tuple_, label, time_interval, new_batch=False):
        self.Delta_in = time_interval
        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]
        try:
            # Narrow down to the relevant row index
            series = self.groups[col_name]
            group_idx = series[series == col_value].index[0]

            # Define the logic for updating delta and uf
            if label == "TN":
                if not self.uf[group_idx]:
                    self.delta[group_idx] += self.threshold
                else:
                    if self.delta[group_idx] >= self.threshold:
                        self.delta[group_idx] -= self.threshold
                    else:
                        self.delta[group_idx] = self.threshold - self.delta[group_idx]
                        self.uf[group_idx] = False
            else:  # For non-TN labels
                if not self.uf[group_idx]:
                    if self.delta[group_idx] > 1 - self.threshold:
                        self.delta[group_idx] -= 1 - self.threshold
                    else:  # Original unfair
                        self.delta[group_idx] = 1 - self.threshold - self.delta[group_idx]
                        self.uf[group_idx] = True
                else:
                    self.delta[group_idx] += 1 - self.threshold
        except KeyError:
            # Handle the case where the value is not found (e.g., do nothing or log a message)
            return
        if not new_batch:
            self.Delta_b += self.Delta_in
            self.Delta_in = 0
            self.current_batch_size += 1


    def _update_delta(self, idx, condition, delta_if_true, delta_if_false):
        if condition:
            self.delta[idx] += delta_if_true
        else:
            self.delta[idx] = max(0, self.delta[idx] + delta_if_false)
            self.uf[idx] = not self.uf[idx]

    def _update_one_counter(self, idx, label):
        if label in ['tp', 'tn']:
            self._update_delta(idx, not self.uf[idx], 10000 - self.threshold, -(10000 - self.threshold))
        elif label in ['fp', 'fn']:
            self._update_delta(idx, not self.uf[idx], -self.threshold, self.threshold)


    def batch_update(self):
        # Compute time decay factor as a float (scaled by 10000)
        decay_exponent = self.Delta_a + (self.Delta_b / 2)
        scaled_decay_factor = (self.alpha) ** decay_exponent
        # Apply decay to `delta`
        self.delta = (self.delta * scaled_decay_factor).astype(int)

        # Reset timers for the new batch
        self.Delta_a = self.Delta_b / 2 + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0


    def get_uf_list(self):
        return self.uf.tolist()
