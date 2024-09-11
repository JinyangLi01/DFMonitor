import sys
import numpy as np
import pandas as pd

class FNR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False]*len(monitored_groups), dtype=bool)
        self.counters_FN = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.counters_TP = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.threshold = threshold
        self.alpha = alpha

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.counters_FN) + self.counters_FN.nbytes
        size += sys.getsizeof(self.counters_TP) + self.counters_TP.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

    def print(self):
        print("FNR_baseline, groups: ", self.groups)
        print("uf", self.uf)
        print("counters_FN", self.counters_FN)
        print("counters_TP", self.counters_TP)
        print("FNR of groups: ", [self.counters_FN[i] / (self.counters_FN[i] + self.counters_TP[i])
                                  if (self.counters_FN[i] + self.counters_TP[i]) > 0 else 0
                                  for i in range(len(self.counters_FN))] )
        print("\n")

    """
    label = "FN" or "TP"
    """
    def insert(self, tuple_, label):
        def belong(index):
            row = self.groups.loc[index]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                if label == "TP":
                    self.counters_TP[index] += 1
                else:  # FN
                    self.counters_FN[index] += 1

                total = self.counters_FN[index] + self.counters_TP[index]
                self.uf[index] = self.counters_FN[index] / total > self.threshold

        for index in self.groups.index:
            belong(index)

    def new_window(self):
        self.counters_FN = self.alpha * self.counters_FN
        self.counters_TP = self.alpha * self.counters_TP
