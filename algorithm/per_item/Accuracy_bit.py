import sys
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
from typing import List, Dict, Union, Optional

class DF_Accuracy_Per_Item_Bit:
    SCALE_FACTOR = 10000
    def __init__(self, monitored_groups, alpha, threshold, use_two_counters=True):
        # Optimize monitored_groups processing
        df = pd.DataFrame(monitored_groups).astype('category')
        self.groups = df
        # Initialize delta values (representing relative distance from flipping fairness bit)
        self.delta = np.zeros(len(monitored_groups), dtype=np.int64)
        self.uf = np.zeros(len(monitored_groups), dtype=bool)  # Fair/unfair flag
        self.alpha = alpha
        self.threshold = int(threshold * self.SCALE_FACTOR)  # Fairness threshold, scaled
        self.use_two_counters = use_two_counters  # Use two counters or four counters

    def initialization(self, uf, delta):
        self.uf = np.array(uf, dtype=bool)
        self.delta = np.array([int(d * self.SCALE_FACTOR) for d in delta], dtype=np.int64)

    def find(self, group: Dict[str, Union[str, int]]) -> bool:
        try:
            idx = self.groups.loc[self.groups[list(group)] == pd.Series(group)].index[0]
            return self.uf[idx]
        except KeyError:
            return False

    def get_size(self):
        size = sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum()
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


    @profile
    def insert(self, tuple_, label, window_num):
        # Apply time decay to all delta values before processing the new item
        self.apply_time_decay(window_num)

        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]  # 22.8ns
        try:
            series = self.groups[col_name]
            group_idx = series[series == col_value].index[0]
        except KeyError:
            return  # If group not found, exit

        delta_index = self.delta[group_idx]
        uf_index = self.uf[group_idx]
        scaled_threshold = self.threshold
        scaled_complement = self.SCALE_FACTOR - scaled_threshold

        if self.use_two_counters:
            if not uf_index:  # Currently fair
                if label == 'correct':
                    delta_index += scaled_complement
                else:
                    if delta_index >= scaled_threshold:
                        delta_index -= scaled_threshold
                    else:
                        delta_index = scaled_threshold - delta_index
                        self.uf[group_idx] = True  # Flip to unfair
            else:  # Currently unfair
                if label == 'correct':
                    if delta_index >= scaled_complement:
                        delta_index -= scaled_complement
                    else:
                        delta_index = scaled_complement - delta_index
                        self.uf[group_idx] = False  # Flip to fair
                else:
                    delta_index += scaled_threshold
        else:  # Four counters (tp, tn, fp, fn)
            if label in ['tp', 'tn']:  # Correct predictions
                if not uf_index:  # Currently fair
                    delta_index += scaled_complement
                else:
                    if delta_index >= scaled_complement:
                        delta_index -= scaled_complement
                    else:
                        delta_index = scaled_complement - delta_index
                        self.uf[group_idx] = False  # Flip to fair
            elif label in ['fp', 'fn']:  # Incorrect predictions
                if not uf_index:  # Currently fair
                    if delta_index >= scaled_threshold:
                        delta_index -= scaled_threshold
                    else:
                        delta_index = scaled_threshold - delta_index
                        self.uf[group_idx] = True  # Flip to unfair
                else:  # Currently unfair
                    delta_index += scaled_threshold

        # Update delta at the index
        self.delta[group_idx] = delta_index


    @profile
    def apply_time_decay(self, window_num):
        if window_num > 0:
            decay_factor = self.alpha ** window_num
            self.delta = (self.delta * decay_factor).astype(int)

    def get_uf_list(self):
        return self.uf.tolist()

