import sys
import pandas as pd
import numpy as np


class DF_Accuracy_Per_Item:
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
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() - self.groups.memory_usage().sum()
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

    def insert(self, tuple_, label):
        # This version updates delta and uf after every item, without any window mechanism
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


    def new_window(self):
        # No window mechanism, so this method is not used in per-item accuracy
        pass
