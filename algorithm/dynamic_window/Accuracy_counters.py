import sys
import numpy as np
import pandas as pd


class DF_Accuracy_Dynamic_Window_Counter:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df

        # Initialize counters for FP, FN, TP, TN
        self.fp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # False Positives
        self.fn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # False Negatives
        self.tp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # True Positives
        self.tn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # True Negatives

        self.uf = np.array([False] * len(monitored_groups), dtype=bool)  # Fair/unfair flag
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
        size += sys.getsizeof(self.fp_counters) + self.fp_counters.nbytes
        size += sys.getsizeof(self.fn_counters) + self.fn_counters.nbytes
        size += sys.getsizeof(self.tp_counters) + self.tp_counters.nbytes
        size += sys.getsizeof(self.tn_counters) + self.tn_counters.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

    def print(self):
        print("uf", self.uf)
        print("FP Counters", self.fp_counters)
        print("FN Counters", self.fn_counters)
        print("TP Counters", self.tp_counters)
        print("TN Counters", self.tn_counters)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    label = one of 'fp', 'fn', 'tp', 'tn'
    """

    def insert(self, tuple_, label, current_time):
        self.current_time = current_time
        time_interval = self.current_time - self.last_item_time

        # Update Delta_in to track the time since the last item
        self.Delta_in = time_interval

        # Check if batch updating is required based on the conditions
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()

        # Insert the new tuple and update relevant fields
        for index in self.groups.index:
            row = self.groups.loc[index]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                # Update the respective counter based on the label
                if label == 'fp':
                    self.fp_counters[index] += 1
                elif label == 'fn':
                    self.fn_counters[index] += 1
                elif label == 'tp':
                    self.tp_counters[index] += 1
                elif label == 'tn':
                    self.tn_counters[index] += 1

                # Calculate the accuracy for the group
                total = self.tp_counters[index] + self.fp_counters[index]
                if total > 0:  # Avoid division by zero
                    accuracy = self.tp_counters[index] / total
                else:
                    accuracy = 0

                # Update the unfair/fair flag based on the threshold
                self.uf[index] = accuracy <= self.threshold

        # Update the timers
        self.Delta_b += self.Delta_in
        self.Delta_in = 0  # Reset Delta_in after each insertion
        self.current_batch_size += 1
        self.last_item_time = self.current_time

    def batch_update(self):
        # Apply time decay to the current batch
        time_decay_factor = self.alpha ** (self.Delta_a + (self.Delta_b / 2))
        self.fp_counters *= time_decay_factor
        self.fn_counters *= time_decay_factor
        self.tp_counters *= time_decay_factor
        self.tn_counters *= time_decay_factor

        # Start a new batch and update the timers
        self.Delta_a = (self.Delta_b / 2) + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0
