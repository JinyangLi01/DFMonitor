import sys
import pandas as pd
import numpy as np
from line_profiler_pycharm import profile
from typing import List, Dict, Union, Optional

class DF_Accuracy_Per_Item_Counter:
    def __init__(self, monitored_groups, alpha, threshold, use_two_counters=True):
        self.groups = pd.DataFrame(monitored_groups).astype("category")
        self.use_two_counters = use_two_counters

        num_groups = len(monitored_groups)
        if use_two_counters:
            self.correct_prediction_counters = np.zeros(num_groups, dtype=np.float64)
            self.incorrect_prediction_counters = np.zeros(num_groups, dtype=np.float64)
        else:
            self.fp_counters = np.zeros(num_groups, dtype=np.float64)
            self.fn_counters = np.zeros(num_groups, dtype=np.float64)
            self.tp_counters = np.zeros(num_groups, dtype=np.float64)
            self.tn_counters = np.zeros(num_groups, dtype=np.float64)

        self.uf = np.zeros(num_groups, dtype=bool)  # Fair/unfair flag
        self.alpha = alpha  # Time decay factor
        self.threshold = threshold  # Fairness threshold


    def initialization(self, uf, correct_prediction_counters=None, incorrect_prediction_counters=None,
                        fp_counters=None, fn_counters=None, tp_counters=None, tn_counters=None):
        self.uf = np.array(uf, dtype=bool)
        if self.use_two_counters:
            if correct_prediction_counters and incorrect_prediction_counters:
                self.correct_prediction_counters = np.array(correct_prediction_counters, dtype=np.float64)
                self.incorrect_prediction_counters = np.array(incorrect_prediction_counters, dtype=np.float64)
        else:
            if fp_counters and fn_counters and tp_counters and tn_counters:
                self.fp_counters = np.array(fp_counters, dtype=np.float64)
                self.fn_counters = np.array(fn_counters, dtype=np.float64)
                self.tp_counters = np.array(tp_counters, dtype=np.float64)
                self.tn_counters = np.array(tn_counters, dtype=np.float64)


    def find(self, group: Dict[str, Union[str, int]]) -> bool:
        return self.groups.loc[self.groups[list(group)] == pd.Series(group)].index[0] in self.uf


    def get_size(self) -> int:
        counters = [
            self.correct_prediction_counters if self.use_two_counters else self.fp_counters,
            self.incorrect_prediction_counters if self.use_two_counters else self.fn_counters,
            None if self.use_two_counters else self.tp_counters,
            None if self.use_two_counters else self.tn_counters,
        ]
        counters = [c for c in counters if c is not None]
        return (sum(sys.getsizeof(c) + c.nbytes for c in counters) + sys.getsizeof(self.uf)
                + sys.getsizeof(self.groups)) + self.groups.memory_usage(deep=True).sum()


    def print(self):
        print("uf", self.uf)

        if self.use_two_counters:
            print("Correct Prediction Counters", self.correct_prediction_counters)
            print("Incorrect Prediction Counters", self.incorrect_prediction_counters)
        else:
            print("FP Counters", self.fp_counters)
            print("FN Counters", self.fn_counters)
            print("TP Counters", self.tp_counters)
            print("TN Counters", self.tn_counters)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def update_group(self, group_idx: int, label: str):
        if self.use_two_counters:
            if label == "correct":
                self.correct_prediction_counters[group_idx] += 1
            elif label == "incorrect":
                self.incorrect_prediction_counters[group_idx] += 1
        else:
            getattr(self, f"{label}_counters")[group_idx] += 1


    @profile
    def insert(self, tuple_, label, window_num):
        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]  # 22.8ns
        try:
            series = self.groups[col_name]
            group_idx = series[series == col_value].index[0]
        except KeyError:
            return  # If group not found, exit
        # # Convert input tuple to a DataFrame row for vectorized comparison
        # group_values = self.groups[self.groups.columns[0]].values
        # # Directly access the value to match
        # col_value = tuple_[self.groups.columns[0]]
        # # Find matches using numpy
        # match = group_values == col_value
        # matched_indices = np.where(match)[0]
        # if matched_indices.size > 0:
        # group_idx = int(matched_indices[0])
        self.update_group(group_idx, label)
        counters = (
            self.correct_prediction_counters[group_idx],
            self.incorrect_prediction_counters[group_idx],
        ) if self.use_two_counters else (
            self.tp_counters[group_idx] + self.fn_counters[group_idx],
            self.fp_counters[group_idx] + self.tn_counters[group_idx],
        )
        total = sum(counters)
        accuracy = counters[0] / total if total > 0 else 0
        self.uf[group_idx] = accuracy <= self.threshold



    @profile
    def apply_time_decay(self, window_num):
        if window_num > 0:
            decay_factor = self.alpha ** window_num
            if self.use_two_counters:
                self.correct_prediction_counters *= decay_factor
                self.incorrect_prediction_counters *= decay_factor
            else:
                self.fp_counters *= decay_factor
                self.fn_counters *= decay_factor
                self.tp_counters *= decay_factor
                self.tn_counters *= decay_factor

    def get_accuracy_list(self):
        if self.use_two_counters:
            total_predictions = self.correct_prediction_counters + self.incorrect_prediction_counters
            with np.errstate(divide='ignore', invalid='ignore'):
                accuracy = np.divide(self.correct_prediction_counters, total_predictions, where=total_predictions > 0)
            return accuracy.tolist()
        else:
            total_predictions = self.tp_counters + self.fp_counters + self.fn_counters + self.tn_counters
            with np.errstate(divide='ignore', invalid='ignore'):
                accuracy = np.divide(self.tp_counters + self.fn_counters, total_predictions,
                                     where=total_predictions > 0)
            return accuracy.tolist()

    def get_accuracy_group(self, group: Dict[str, Union[str, int]]) -> float:
        # Convert the group dictionary to a NumPy array for vectorized comparison
        group_values = np.array([self.groups[col].cat.codes for col in self.groups.columns]).T
        group_codes = np.array([self.groups[col].cat.categories.get_loc(value)
                                if value in self.groups[col].cat.categories else -1
                                for col, value in group.items()])

        # Find the matching row index
        match = np.all(group_values == group_codes, axis=1)
        matched_indices = np.where(match)[0]

        if matched_indices.size > 0:
            idx = matched_indices[0]  # Take the first match
            if self.use_two_counters:
                correct_val = self.correct_prediction_counters[idx]
                incorrect_val = self.incorrect_prediction_counters[idx]
                total = correct_val + incorrect_val
                return correct_val / total if total > 0 else 0
            else:
                tp_val = self.tp_counters[idx]
                fn_val = self.fn_counters[idx]
                total = tp_val + fn_val + self.fp_counters[idx] + self.tn_counters[idx]
                return (tp_val + fn_val) / total if total > 0 else 0
        return 0.0


    def get_uf_list(self):
        return self.uf.tolist()

    def get_counter_correctness(self):
        return self.correct_prediction_counters, self.incorrect_prediction_counters

    def get_counter_fp_fn_tp_tn(self):
        return self.fp_counters, self.fn_counters, self.tp_counters, self.tn_counters
