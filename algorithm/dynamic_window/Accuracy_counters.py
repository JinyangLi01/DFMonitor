import sys
import numpy as np
import pandas as pd


class DF_Accuracy_Dynamic_Window_Counter:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in, use_two_counters=True):
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
            self.fp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # False Positives
            self.fn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # False Negatives
            self.tp_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # True Positives
            self.tn_counters = np.array([0] * len(monitored_groups), dtype=np.float64)  # True Negatives

        self.uf = np.array([False] * len(monitored_groups), dtype=bool)  # Fair/unfair flag
        self.alpha = alpha
        self.threshold = threshold
        self.T_b = T_b  # Maximum batch size (time duration limit)
        self.T_in = T_in  # Maximum time interval between items

        # Initialize the three timers
        self.Delta_a = 0  # Tracks time between the midpoint of the last batch and the start of the latest batch
        self.Delta_b = 0  # Tracks the time duration of the current batch
        self.Delta_in = 0  # Tracks the time interval since the arrival of the last item
        self.last_item_time = 0  # Track the time of the last inserted item
        self.current_time = 0  # Current system time, incremented per item
        self.current_batch_size = 0  # Tracks the size of the current batch

    def get_counter_correctness(self):
        return self.correct_prediction_counters, self.incorrect_prediction_counters

    def get_counter_fp_fn_tp_tn(self):
        return self.fp_counters, self.fn_counters, self.tp_counters, self.tn_counters

    def get_accuracy_list(self):
        if self.use_two_counters:
            # Avoid division by zero for the two-counter scenario
            total_predictions = self.correct_prediction_counters + self.incorrect_prediction_counters
            s = [correct / (correct + incorrect) if correct + incorrect > 0 else 0 for correct, incorrect in
                    zip(self.correct_prediction_counters, self.incorrect_prediction_counters)]
            return s
        else:
            # Avoid division by zero for the four-counter scenario (tp and fp)
            total_predictions = self.tp_counters + self.fp_counters + self.fn_counters + self.tn_counters
            s = [(tp + fn) / total if total > 0 else 0 for tp, fn, total in
                    zip(self.tp_counters, self.fn_counters, total_predictions)]
            return s

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
        size += sys.getsizeof(self.T_b)
        size += sys.getsizeof(self.T_in)
        size += sys.getsizeof(self.Delta_a)
        size += sys.getsizeof(self.Delta_b)
        size += sys.getsizeof(self.Delta_in)
        size += sys.getsizeof(self.last_item_time)
        size += sys.getsizeof(self.current_time)
        size += sys.getsizeof(self.current_batch_size)
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

    """
    label = one of 'fp', 'fn', 'tp', 'tn'
    """

    def whether_need_batch_renewal(self, time_interval):
        self.Delta_in = time_interval
        print(
            f"func whether_need_batch_renewal, Delta_in = {self.Delta_in}, T_in = {self.T_in}, Delta_b = {self.Delta_b}, T_b = {self.T_b}, time_interval = {time_interval}")
        return self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b

    def insert(self, tuple_, label, time_interval):
        self.Delta_in = time_interval
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()
        # Insert the new tuple and update relevant fields
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

                    # Calculate the accuracy for the group
                    total = (self.tp_counters[index] + self.fp_counters[index]
                             + self.fn_counters[index] + self.tn_counters[index])
                    if total > 0:  # Avoid division by zero
                        accuracy = (self.tp_counters[index] + self.fn_counters[index]) / total
                    else:
                        accuracy = 0

                # Update the unfair/fair flag based on the threshold
                self.uf[index] = accuracy <= self.threshold

        # Update the timers
        self.Delta_b += self.Delta_in
        self.Delta_in = 0  # Reset Delta_in after each insertion
        self.current_batch_size += 1
        self.last_item_time = self.current_time
        return


    def batch_update(self):
        # Apply time decay to the current batch
        time_decay_factor = self.alpha ** (self.Delta_a + (self.Delta_b / 2))
        if self.use_two_counters:
            self.correct_prediction_counters *= time_decay_factor
            self.incorrect_prediction_counters *= time_decay_factor
        else:
            self.fp_counters *= time_decay_factor
            self.fn_counters *= time_decay_factor
            self.tp_counters *= time_decay_factor
            self.tn_counters *= time_decay_factor

        # Start a new batch and update the timers
        self.Delta_a = (self.Delta_b / 2) + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0

    def get_uf_list(self):
        return self.uf.tolist()

