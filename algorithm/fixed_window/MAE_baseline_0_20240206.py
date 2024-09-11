import sys

import numpy as np
import pandas as pd
class Accuracy_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df
        self.uf = np.array([False]*len(monitored_groups), dtype=bool)
        self.counters_correct = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.counters_incorrect = np.array([0]*len(monitored_groups), dtype=np.float64)
        self.alpha = alpha
        self.threshold = threshold

    def find(self, group):
        # idx = self.group2idx[str(group)]
        idx = self.groups.index(group)
        return self.uf[idx]


    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.counters_correct) + self.counters_correct.nbytes
        size += sys.getsizeof(self.counters_incorrect) + self.counters_incorrect.nbytes
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size


    def print(self):
        print("uf", self.uf)
        print("counters_correct", self.counters_correct)
        print("counters_incorrect", self.counters_incorrect)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    label = correct or incorrect
    """

    def insert(self, tuple_, label):
        for index in self.groups.index:
            row = self.groups.loc[index]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                if label == 'correct':
                    self.counters_correct[index] += 1
                else:
                    self.counters_incorrect[index] += 1
                self.uf[index] = (self.counters_correct[index] /
                                  (self.counters_correct[index] + self.counters_incorrect[index])
                                  <= self.threshold)




    def new_window(self):
        self.counters_correct = self.alpha * self.counters_correct
        self.counters_incorrect = self.alpha * self.counters_incorrect
