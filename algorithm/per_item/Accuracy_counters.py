import sys
import pandas as pd
import numpy as np


class DF_Accuracy_Per_Item_Counter:
    def __init__(self, monitored_groups, alpha, threshold, use_two_counters=True):
        df = pd.DataFrame(monitored_groups)
        unique_keys = set()
        for item in monitored_groups:
            unique_keys.update(item.keys())
        for key in unique_keys:
            df[key] = df[key].astype("category")
        self.groups = df

        self.use_two_counters = use_two_counters

        if use_two_counters:
            # Initialize counters for correct and incorrect predictions
            self.correct_prediction_counters = np.array([0] * len(monitored_groups), dtype=np.float64)
            self.incorrect_prediction_counters = np.array([0] * len(monitored_groups), dtype=np.float64)
        else:
            # Initialize counters for FP, FN, TP, TN
            self.fp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)
            self.fn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)
            self.tp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)
            self.tn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)

        self.uf = np.array([False] * len(monitored_groups), dtype=bool)  # Fair/unfair flag
        self.alpha = alpha  # Time decay factor
        self.threshold = threshold  # Fairness threshold

    def initialization(self, uf, correct_prediction_counters=None, incorrect_prediction_counters=None,
                        fp_counters=None, fn_counters=None, tp_counters=None, tn_counters=None):
        self.uf = np.array(uf, dtype=bool)

        if self.use_two_counters:
            self.correct_prediction_counters = np.array(correct_prediction_counters, dtype=np.float64)
            self.incorrect_prediction_counters = np.array(incorrect_prediction_counters, dtype=np.float64)
        else:
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

        if self.use_two_counters:
            size += sys.getsizeof(self.correct_prediction_counters) + self.correct_prediction_counters.nbytes
            size += sys.getsizeof(self.incorrect_prediction_counters) + self.incorrect_prediction_counters.nbytes
        else:
            size += sys.getsizeof(self.fp_counters) + self.fp_counters.nbytes
            size += sys.getsizeof(self.fn_counters) + self.fn_counters.nbytes
            size += sys.getsizeof(self.tp_counters) + self.tp_counters.nbytes
            size += sys.getsizeof(self.tn_counters) + self.tn_counters.nbytes

        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

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

    def insert(self, tuple_, label, time_diff_seconds):
        # Apply time decay to all counters before processing the new item
        self.apply_time_decay(time_diff_seconds)

        # Update counters and fairness flag based on the new item
        for index in self.groups.index:
            row = self.groups.loc[index]
            if all(tuple_.get(key, None) == value for key, value in row.items()):
                if self.use_two_counters:
                    # Update correct/incorrect counters
                    if label == 'correct':
                        self.correct_prediction_counters[index] += 1
                    elif label == 'incorrect':
                        self.incorrect_prediction_counters[index] += 1

                    # Calculate accuracy
                    total = self.correct_prediction_counters[index] + self.incorrect_prediction_counters[index]
                    accuracy = self.correct_prediction_counters[index] / total if total > 0 else 0

                else:
                    # Update the respective counter based on the label
                    if label == 'fp':
                        self.fp_counters[index] += 1
                    elif label == 'fn':
                        self.fn_counters[index] += 1
                    elif label == 'tp':
                        self.tp_counters[index] += 1
                    elif label == 'tn':
                        self.tn_counters[index] += 1

                    # Calculate accuracy (TP / (TP + FP))
                    total = self.tp_counters[index] + self.fp_counters[index]
                    accuracy = self.tp_counters[index] / total if total > 0 else 0

                # Update the unfair/fair flag based on the threshold
                self.uf[index] = accuracy <= self.threshold

    def apply_time_decay(self, time_diff_seconds):
        if time_diff_seconds > 0:
            if self.use_two_counters:
                # Apply the time decay factor alpha to correct and incorrect counters
                self.correct_prediction_counters = self.correct_prediction_counters * (self.alpha ** time_diff_seconds)
                self.incorrect_prediction_counters = self.incorrect_prediction_counters * (self.alpha ** time_diff_seconds)
            else:
                # Apply the time decay factor alpha to all four counters
                self.fp_counters = self.fp_counters * (self.alpha ** time_diff_seconds)
                self.fn_counters = self.fn_counters * (self.alpha ** time_diff_seconds)
                self.tp_counters = self.tp_counters * (self.alpha ** time_diff_seconds)
                self.tn_counters = self.tn_counters * (self.alpha ** time_diff_seconds)

    def get_accuracy_list(self):
        if self.use_two_counters:
            # Avoid division by zero for the two-counter scenario
            total_predictions = self.correct_prediction_counters + self.incorrect_prediction_counters
            return [correct / (correct + incorrect) if correct + incorrect > 0 else 0 for correct, incorrect in zip(self.correct_prediction_counters, self.incorrect_prediction_counters)]
        else:
            # Avoid division by zero for the four-counter scenario (tp and fp)
            total_predictions = self.tp_counters + self.fp_counters + self.fn_counters + self.tn_counters
            return [(tp + fn) / total if total > 0 else 0 for tp, fn, total in zip(self.tp_counters, self.fn_counters, total_predictions)]

    def get_accuracy_group(self, g):
        correct_val = 0
        incorrect_val = 0
        tp_val = 0
        tn_val = 0
        fp_val = 0
        fn_val = 0
        for index in self.groups.index:
            row = self.groups.loc[index]
            if all(g.get(key, None) == value for key, value in row.items()):
                if self.use_two_counters:
                    correct_val += self.correct_prediction_counters[index]
                    incorrect_val += self.incorrect_prediction_counters[index]
                else:
                    tp_val += self.tp_counters[index]
                    tn_val += self.tn_counters[index]
                    fp_val += self.fp_counters[index]
                    fn_val += self.fn_counters[index]
        if self.use_two_counters:
            total = correct_val + incorrect_val
            return correct_val / total if total > 0 else 0
        else:
            total = tp_val + fp_val + fn_val + tn_val
            return (tp_val + fn_val) / total if total > 0 else 0
