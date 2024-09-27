import sys
import pandas as pd
import numpy as np


class DF_Accuracy_Per_Item_Bit:
    def __init__(self, monitored_groups, alpha, threshold, use_two_counters=True):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df

        # Initialize delta values (representing relative distance from flipping fairness bit)
        self.delta = np.array([0] * len(monitored_groups), dtype=np.float64)
        self.uf = np.array([False] * len(monitored_groups), dtype=bool)  # Fair/unfair flag
        self.alpha = alpha  # Time decay factor
        self.threshold = threshold  # Fairness threshold
        self.use_two_counters = use_two_counters  # Use two counters or four counters

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

    """
    label = one of 'correct', 'incorrect' or 'fp', 'fn', 'tp', 'tn'
    """

    def insert(self, tuple_, label, time_diff):
        # Apply time decay to all delta values before processing the new item
        self.apply_time_decay(time_diff)

        # Update delta and fairness flag based on the new item
        for index in self.groups.index:
            row = self.groups.loc[index]
            if all(row.get(key, None) == value for key, value in tuple_.items()):
                # Handle two-counters scenario ('correct', 'incorrect')
                if self.use_two_counters:
                    correct = (label == 'correct')

                    if not self.uf[index]:  # Currently fair
                        if correct:
                            self.delta[index] += 1 - self.threshold
                        else:
                            if self.delta[index] >= self.threshold:
                                self.delta[index] -= self.threshold
                            else:
                                self.delta[index] = self.threshold - self.delta[index]
                                self.uf[index] = True  # Flip to unfair
                    else:  # Currently unfair
                        if correct:
                            if self.delta[index] >= 1 - self.threshold:
                                self.delta[index] -= 1 - self.threshold
                            else:
                                self.delta[index] = 1 - self.threshold - self.delta[index]
                                self.uf[index] = False  # Flip to fair
                        else:
                            self.delta[index] += self.threshold

                # Handle four-counters scenario ('fp', 'fn', 'tp', 'tn')
                else:
                    if label in ['tp', 'tn']:  # Correct predictions
                        if not self.uf[index]:  # Currently fair
                            self.delta[index] += 1 - self.threshold
                        else:  # Currently unfair
                            if self.delta[index] >= 1 - self.threshold:
                                self.delta[index] -= 1 - self.threshold
                            else:
                                self.delta[index] = 1 - self.threshold - self.delta[index]
                                self.uf[index] = False  # Flip to fair
                    elif label in ['fp', 'fn']:  # Incorrect predictions
                        if not self.uf[index]:  # Currently fair
                            if self.delta[index] >= self.threshold:
                                self.delta[index] -= self.threshold
                            else:
                                self.delta[index] = self.threshold - self.delta[index]
                                self.uf[index] = True  # Flip to unfair
                        else:  # Currently unfair
                            self.delta[index] += self.threshold

    def apply_time_decay(self, time_diff_seconds):
        if time_diff_seconds > 0:
            # Apply the time decay factor alpha to all delta values
            self.delta = self.delta * (self.alpha ** time_diff_seconds)
