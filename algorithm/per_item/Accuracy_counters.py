import sys
import pandas as pd
import numpy as np


class DF_Accuracy_Per_Item_Counter:
    def __init__(self, monitored_groups, alpha, threshold):
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
        self.alpha = alpha  # Time decay factor
        self.threshold = threshold  # Fairness threshold

    def initialization(self, uf, fp_counters, fn_counters, tp_counters, tn_counters):
        self.uf = np.array(uf, dtype=bool)
        self.fp_counters = np.array(fp_counters, dtype=np.float64)
        self.fn_counters = np.array(fn_counters, dtype=np.float64)
        self.tp_counters = np.array(tp_counters, dtype=np.float64)
        self.tn_counters = np.array(tn_counters, dtype=np.float64)

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

    def insert(self, tuple_, label):
        # Apply time decay to all counters before processing the new item
        self.apply_time_decay()

        # Update counters and fairness flag based on the new item
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

    def apply_time_decay(self):
        # Apply the time decay factor alpha to all counters
        self.fp_counters *= self.alpha
        self.fn_counters *= self.alpha
        self.tp_counters *= self.alpha
        self.tn_counters *= self.alpha
