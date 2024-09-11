import sys

import numpy as np
import pandas as pd


class DF_CR:
    def __init__(self, monitored_groups, alpha, threshold):
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

    def initialization(self, uf, delta):
        self.uf = np.array(uf, dtype=bool)
        self.delta = np.array(delta, dtype=np.float64)

    def find(self, group):
        # idx = self.group2idx[str(group)]
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("delta", self.delta)


    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
        size += sys.getsizeof(self.uf) + self.uf.nbytes
        size += sys.getsizeof(self.delta) + self.delta.nbytes
        size += sys.getsizeof(self.alpha)
        size += sys.getsizeof(self.threshold)
        return size

    """
    here we assume that the tuple can satisfy multiple groups
    """
    def insert(self, tuple_):
        for group_idx in self.groups.index:
            row = self.groups.loc[group_idx]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                if not self.uf[group_idx]:
                    self.delta[group_idx] += 1 - self.threshold
                else:
                    if self.delta[group_idx] >= 1 - self.threshold:
                        self.delta[group_idx] -= 1 - self.threshold
                    else:
                        self.delta[group_idx] = 1 - self.threshold - self.delta[group_idx]
                        self.uf[group_idx] = False
            else:
                if self.uf[group_idx]:
                    self.delta[group_idx] += self.threshold
                else:
                    if self.delta[group_idx] >= self.threshold:
                        self.delta[group_idx] -= self.threshold
                    else:
                        self.delta[group_idx] = self.threshold - self.delta[group_idx]
                        self.uf[group_idx] = True



    def new_window(self):
        self.delta = self.delta * self.alpha


