import sys
import numpy as np
import pandas as pd
from line_profiler_pycharm import profile

class DF_Accuracy_Dynamic_Window_Counter:
    def __init__(self, monitored_groups, alpha, threshold, T_b, T_in, use_two_counters=True):
        self.groups = pd.DataFrame(monitored_groups).astype('category')
        self.use_two_counters = use_two_counters
        self.alpha = alpha
        self.threshold = int(threshold * 10000)  # Scale threshold by 10000
        self.T_b = T_b  # Maximum batch size (time duration limit)
        self.T_in = T_in  # Maximum time interval between items

        num_groups = len(monitored_groups)
        if use_two_counters:
            # Initialize counters for correct and incorrect predictions
            self.correct_prediction_counters = np.zeros(num_groups, dtype=int)
            self.incorrect_prediction_counters = np.zeros(num_groups, dtype=int)
        else:
            # Initialize counters for FP, FN, TP, TN
            self.fp_counters = np.zeros(num_groups, dtype=int)
            self.fn_counters = np.zeros(num_groups, dtype=int)
            self.tp_counters = np.zeros(num_groups, dtype=int)
            self.tn_counters = np.zeros(num_groups, dtype=int)

        self.uf = np.zeros(num_groups, dtype=bool)  # Fair/unfair flag

        # Initialize the three timers
        self.Delta_a = 0  # Tracks time between the midpoint of the last batch and the start of the latest batch
        self.Delta_b = 0  # Tracks the time duration of the current batch
        self.Delta_in = 0  # Tracks the time interval since the arrival of the last item
        self.last_item_time = 0  # Track the time of the last inserted item
        self.current_time = 0  # Current system time, incremented per item
        self.current_batch_size = 0  # Tracks the size of the current batch


    def get_counter_correctness(self):
        return self.correct_prediction_counters/10000, self.incorrect_prediction_counters/10000

    def get_counter_fp_fn_tp_tn(self):
        return self.fp_counters/10000, self.fn_counters/10000, self.tp_counters/10000, self.tn_counters/10000

    def get_accuracy_list(self):
        if self.use_two_counters:
            total_predictions = self.correct_prediction_counters + self.incorrect_prediction_counters
            with np.errstate(divide='ignore', invalid='ignore'):
                accuracy = np.divide(self.correct_prediction_counters, total_predictions, where=total_predictions > 0)
            return accuracy.tolist()
        else:
            total_predictions = self.tp_counters + self.fp_counters + self.fn_counters + self.tn_counters
            with np.errstate(divide='ignore', invalid='ignore'):
                accuracy = np.divide(self.tp_counters + self.fn_counters, total_predictions, where=total_predictions > 0)
            return accuracy.tolist()


    def initialization(self, uf, correct_prediction_counters=None, incorrect_prediction_counters=None,
                        fp_counters=None, fn_counters=None, tp_counters=None, tn_counters=None):
        self.uf = np.array(uf, dtype=bool)
        if self.use_two_counters:
            self.correct_prediction_counters = np.array(correct_prediction_counters, dtype=int)
            self.incorrect_prediction_counters = np.array(incorrect_prediction_counters, dtype=int)
        else:
            self.fp_counters = np.array(fp_counters, dtype=int)
            self.fn_counters = np.array(fn_counters, dtype=int)
            self.tp_counters = np.array(tp_counters, dtype=int)
            self.tn_counters = np.array(tn_counters, dtype=int)

    def get_uf_of_group(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = (sys.getsizeof(self.groups) + self.groups.memory_usage(deep=True).sum() + sys.getsizeof(self.uf) +
                self.uf.nbytes + sys.getsizeof(self.threshold) + sys.getsizeof(self.alpha) + sys.getsizeof(self.T_b) +
                sys.getsizeof(self.T_in) + sys.getsizeof(self.Delta_a) + sys.getsizeof(self.Delta_b) +
                sys.getsizeof(self.Delta_in) + sys.getsizeof(self.last_item_time) + sys.getsizeof(self.current_time) +
                sys.getsizeof(self.current_batch_size))
        if self.use_two_counters:
            size += (sys.getsizeof(self.correct_prediction_counters) + self.correct_prediction_counters.nbytes +
                     sys.getsizeof(self.incorrect_prediction_counters) + self.incorrect_prediction_counters.nbytes)
        else:
            size += (sys.getsizeof(self.fp_counters) + self.fp_counters.nbytes + sys.getsizeof(self.fn_counters) +
                     self.fn_counters.nbytes + sys.getsizeof(self.tp_counters) + self.tp_counters.nbytes +
                     sys.getsizeof(self.tn_counters) + self.tn_counters.nbytes)
        return size

    def print(self):
        print("uf", self.uf)
        if self.use_two_counters:
            print("Correct Prediction Counters", self.correct_prediction_counters/10000)
            print("Incorrect Prediction Counters", self.incorrect_prediction_counters/10000)
        else:
            print("FP Counters", self.fp_counters/10000)
            print("FN Counters", self.fn_counters/10000)
            print("TP Counters", self.tp_counters/10000)
            print("TN Counters", self.tn_counters/10000)


    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    label = one of 'fp', 'fn', 'tp', 'tn'
    """
    @profile
    def whether_need_batch_renewal(self, time_interval):
        self.Delta_in = time_interval
        if self.Delta_in >= self.T_in or self.Delta_b + self.Delta_in >= self.T_b:
            self.batch_update()
            return True
        return False

    @profile
    def insert(self, tuple_, label, time_interval, new_batch=False):
        self.Delta_in = time_interval
        col_name = self.groups.columns[0]
        col_value = tuple_[col_name]
        try:
            series = self.groups[col_name]
            index = series[series == col_value].index[0]
        except KeyError:
            return  # If group not found, exit

        if self.use_two_counters:
            if label == 'correct':
                self.correct_prediction_counters[index] += 10000
            elif label == 'incorrect':
                self.incorrect_prediction_counters[index] += 10000
            total = self.correct_prediction_counters[index] + self.incorrect_prediction_counters[index]
            accuracy = self.correct_prediction_counters[index] / total if total > 0 else 0
            self.uf[index] = accuracy <= self.threshold
        else:
            if label == 'fp':
                self.fp_counters[index] += 10000
            elif label == 'fn':
                self.fn_counters[index] += 10000
            elif label == 'tp':
                self.tp_counters[index] += 10000
            elif label == 'tn':
                self.tn_counters[index] += 10000
            total = (self.tp_counters[index] + self.fp_counters[index] +
                     self.fn_counters[index] + self.tn_counters[index])
            accuracy = (self.tp_counters[index] + self.fn_counters[index]) / total if total > 0 else 0
            self.uf[index] = accuracy <= self.threshold
        if not new_batch:
            self.Delta_b += self.Delta_in
            self.Delta_in = 0
            self.current_batch_size += 1
            self.last_item_time = self.current_time


    def np_list_scale(self, list, time_decay_factor):
        list = np.array(list, dtype=np.float64)
        list *= time_decay_factor
        list = list.astype(np.int32)
        return list

    @profile
    def batch_update(self):
        time_decay_factor = (self.alpha) ** (self.Delta_a + (self.Delta_b / 2))
        if self.use_two_counters:
            self.correct_prediction_counters = np.array(self.correct_prediction_counters, dtype=np.float64)
            self.correct_prediction_counters *= time_decay_factor
            self.correct_prediction_counters = self.correct_prediction_counters.astype(np.int32)
            self.incorrect_prediction_counters = np.array(self.incorrect_prediction_counters, dtype=np.float64)
            self.incorrect_prediction_counters *= time_decay_factor
            self.incorrect_prediction_counters = self.incorrect_prediction_counters.astype(np.int32)
        else:
            self.fp_counters = self.np_list_scale(self.fp_counters, time_decay_factor)
            self.fn_counters = self.np_list_scale(self.fn_counters, time_decay_factor)
            self.tp_counters = self.np_list_scale(self.tp_counters, time_decay_factor)
            self.tn_counters = self.np_list_scale(self.tn_counters, time_decay_factor)

        self.Delta_a = (self.Delta_b / 2) + self.Delta_in
        self.Delta_b = 0
        self.Delta_in = 0
        self.current_batch_size = 0

    @profile
    def get_uf_list(self):
        return self.uf.tolist()

