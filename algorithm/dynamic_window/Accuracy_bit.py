import sys
import pandas as pd
import numpy as np


class DF_Accuracy_Dynamic_Window_Bit:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in,  use_two_counters=True):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False] * len(monitored_groups), dtype=bool)
        self.delta = np.array([0] * len(monitored_groups), dtype=np.float64)
        self.alpha = alpha
        self.threshold = threshold
        self.T_b = T_b  # Maximum batch size (time duration limit)
        self.T_in = T_in  # Maximum time interval between items

        # Initialize the three timers
        self.Delta_a = 0  # Tracks time between the midpoint of the last batch and the start of the latest batch
        self.Delta_b = 0  # Tracks the time duration of the current batch
        self.Delta_in = 0  # Tracks the time interval since the arrival of the last item
        self.last_item_time = 0  # Track the time of the last inserted item
        self.current_time = 0  # Current system time, incremented per item
        self.current_batch_size = 0  # Tracks the size of the current batch

    def initialization(self, uf, delta):
        self.uf = np.array(uf, dtype=bool)
        self.delta = np.array(delta, dtype=np.float64)

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

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
        print("delta", self.delta)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def insert(self, tuple_, label, time_interval):
        # Update Delta_in to track the time since the last item
        self.Delta_in = time_interval
        # Check if batch updating is required based on the conditions
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()

        # Insert the new tuple and update relevant fields
        for index in self.groups.index:
            row = self.groups.loc[index]
            correct = (label == 'correct')
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                if not self.uf[index]:
                    if correct:
                        self.delta[index] += 1 - self.threshold
                    else:
                        if self.delta[index] >= self.threshold:
                            self.delta[index] -= self.threshold
                        else:
                            self.delta[index] = self.threshold - self.delta[index]
                            self.uf[index] = True
                else:
                    if correct:
                        if self.delta[index] >= 1 - self.threshold:
                            self.delta[index] -= 1 - self.threshold
                        else:
                            self.delta[index] = 1 - self.threshold - self.delta[index]
                            self.uf[index] = False
                    else:
                        self.delta[index] += self.threshold

        # Update the timers
        self.Delta_b += self.Delta_in
        self.Delta_in = 0  # Reset Delta_in after each insertion
        self.current_batch_size += 1
        self.last_item_time = self.current_time

    def batch_update(self):
        # Apply time decay to the current batch
        time_decay_factor = self.alpha ** (self.Delta_a + (self.Delta_b / 2))
        self.delta *= time_decay_factor

        # Start a new batch and update the timers
        self.Delta_a = (self.Delta_b / 2) + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0

    def get_uf_list(self):
        return self.uf.tolist()