import sys
import numpy as np
import pandas as pd


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size

class FPR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False]*len(monitored_groups), dtype=bool)
        self.counters_TN = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.counters_FP = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.threshold = threshold
        self.alpha = alpha

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.counters_TN) + self.counters_TN.nbytes
        size += sys.getsizeof(self.counters_FP) + self.counters_FP.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

    def print(self):
        print("FPR_baseline, groups: ", self.groups)
        print("uf", self.uf)
        print("counters_TN", self.counters_TN)
        print("counters_FP", self.counters_FP)
        print("FPR of groups: ", [self.counters_FP[i] / (self.counters_TN[i] + self.counters_FP[i])
                                  for i in range(len(self.counters_FP))] )
        print("\n")


    """
    label = "FP" or "TN"
    """

    def insert(self, tuple_, label):
        def belong(index):
            row = self.groups.loc[index]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                if label == "TN":
                    self.counters_TN[index] += 1
                else:
                    self.counters_FP[index] += 1

                total = self.counters_TN[index] + self.counters_FP[index]
                self.uf[index] = self.counters_FP[index] / total > self.threshold

        for index in self.groups.index:
            belong(index)


    def new_window(self):
        self.counters_FP = self.alpha * self.counters_FP
        self.counters_TN = self.alpha * self.counters_TN


