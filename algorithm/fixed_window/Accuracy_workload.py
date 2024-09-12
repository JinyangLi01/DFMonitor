import time

import pandas as pd
from algorithm.fixed_window import Accuracy_baseline_0_20240206 as Accuracy_baseline, Accuracy_0_20240206 as CR, config
import copy


# use CR to monitor tpr of different races over time
# a time window is a month

# Function to compute the time window key (e.g., year-month for 'month' windows)
def compute_time_window_key(row, window_type, window_size, first_row_time):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        if window_size == 1:
            return "{}-{}".format(row.year, row.month)
        else:
            # if row.month == 2:
            #     print("row = {}, first_row_time = {}".format(row, first_row_time))
            num_year = row.year - first_row_time.year
            num_month = num_year * 12 + row.month - 1
            num_window = num_month // window_size
            last_new_window_month = first_row_time.month + num_window * window_size
            return "{}-{}".format(first_row_time.year + last_new_window_month // 12,
                                  last_new_window_month % 12 + 1)
    elif window_type == 'week':
        # Calculate the number of days since the start of the year to the current row date
        days_since_start_of_year = (row - pd.Timestamp(year=row.year, month=1, day=1)).days
        # For a 2-week window, divide by 14 (number of days in 2 weeks) to find the window index
        window_index = days_since_start_of_year // (window_size * 7)  # 7 days in a week
        # Calculate the start date of the window
        window_start_date = pd.Timestamp(year=row.year, month=1, day=1) + pd.Timedelta(
            days=window_index * window_size * 7)
        return "{}-W{}".format(window_start_date.year, window_index + 1)
    elif window_type == 'day':
        return "{}-{}-{}".format(row.year, row.month, row.day // window_size * window_size)


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def traverse_data_DFMonitor_and_baseline(timed_data, date_column, time_window_str, date_time_format, monitored_groups,
                                         threshold, alpha, label_prediction="predicted",
                                         label_ground_truth="ground_truth"):
    if date_time_format:
        window_size, window_type = time_window_str.split()
        # Apply the function to compute the window key for each row
        first_row_time = timed_data.loc[0, date_column]
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                                 args=(window_type, int(window_size), first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data["new_window"].loc[0] = False
    DFMonitor = CR.DF_Accuracy_Fixed_Window_Bit(monitored_groups, alpha, threshold)
    DFMonitor_baseline = Accuracy_baseline.DF_Accuracy_Fixed_Window_Counter(monitored_groups, alpha, threshold)
    counter_first_window_correct = [0] * len(monitored_groups)
    counter_first_window_incorrect = [0] * len(monitored_groups)
    first_window_processed = False
    index = 0

    for index, row in timed_data.iterrows():
        if first_window_processed:
            if DFMonitor.uf != DFMonitor_baseline.uf:
                print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
                DFMonitor.print()
                DFMonitor_baseline.print()
                return DFMonitor, DFMonitor_baseline

        if row['new_window']:
            DFMonitor_baseline.new_window()
            if not first_window_processed:
                uf = []
                for k in range(len(monitored_groups)):
                    if counter_first_window_correct[k] + counter_first_window_incorrect[k] == 0:
                        uf.append(None)
                    else:
                        if (counter_first_window_correct[k] /
                                (counter_first_window_correct[k] + counter_first_window_incorrect[k]) <= threshold):
                            uf.append(True)
                        else:
                            uf.append(False)

                delta = [abs(threshold * (counter_first_window_correct[j] + counter_first_window_incorrect[j])
                             - counter_first_window_correct[j]) * alpha
                         for j in range(len(counter_first_window_incorrect))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'
            DFMonitor_baseline.insert(row, label)
        if not first_window_processed:
            for group_idx, group in enumerate(monitored_groups):
                if belong_to_group(row, group):
                    if label == 'correct':
                        counter_first_window_correct[group_idx] += 1
                    else:
                        counter_first_window_incorrect[group_idx] += 1
        else:
            DFMonitor_baseline.insert(row, label)

    # last window:
    if DFMonitor.uf != DFMonitor_baseline.uf:
        # print("index = {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
        DFMonitor.print()
        DFMonitor_baseline.print()
    return DFMonitor, DFMonitor_baseline


def traverse_data_DFMonitor_only(timed_data, date_column, time_window_str, date_time_format, monitored_groups,
                                 threshold,
                                 alpha, label_prediction="predicted", label_ground_truth="ground_truth"):
    global time1
    if date_time_format:
        window_size, window_type = time_window_str.split()
        # Apply the function to compute the window key for each row
        first_row_time = timed_data.loc[0, date_column]
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                                 args=(window_type, int(window_size), first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    # timed_data.loc["new_window", 0] = False
    timed_data.loc[0, "new_window"] = False
    # print(timed_data[:1])
    # print(len(timed_data[timed_data['new_window'] == True]))
    DFMonitor = CR.DF_Accuracy_Fixed_Window_Bit(monitored_groups, alpha, threshold)
    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    first_window_processed = False
    first_row_id = timed_data[timed_data['new_window'] == True].index[0]
    num_items_after_first_window = len(timed_data) - first_row_id
    total_time_new_window = 0
    num_new_windows = len(timed_data[timed_data['new_window'] == True])

    for index, row in timed_data.iterrows():
        if row['new_window']:
            if not first_window_processed:
                uf = [True if counter_values_correct[j] /
                              (counter_values_correct[j] + counter_values_incorrect[j]) <= threshold
                      else False for j in range(len(counter_values_correct))]
                delta = [round(abs(threshold * (counter_values_correct[j] + counter_values_incorrect[j]) -
                                   counter_values_correct[j]) * alpha, config.decimal)
                         for j in range(len(counter_values_incorrect))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
                time1 = time.time()
            else:
                timea = time.time()
                DFMonitor.new_window()
                timeb = time.time()
                total_time_new_window += timeb - timea
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'
        if first_window_processed:
            DFMonitor.insert(row, label)
        else:
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == 'correct':
                        counter_values_correct[i] += 1
                    else:
                        counter_values_incorrect[i] += 1
    time2 = time.time()
    elapsed_time = time2 - time1
    return DFMonitor, elapsed_time, total_time_new_window, num_items_after_first_window, num_new_windows


def traverse_data_DFMonitor(timed_data, date_column, time_window_str, date_time_format, monitored_groups, threshold,
                            alpha, label_prediction="predicted", label_ground_truth="ground_truth"):
    if date_time_format:
        window_size, window_type = time_window_str.split()
        first_row_time = timed_data.loc[0, date_column]
        # Apply the function to compute the window key for each row
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(window_type,
                                                                                                int(window_size),
                                                                                                first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    # timed_data.loc["new_window", 0] = False
    timed_data.loc[0, "new_window"] = False
    # print(timed_data[:1])
    # print(len(timed_data[timed_data['new_window'] == True]))
    DFMonitor = CR.DF_Accuracy_Fixed_Window_Bit(monitored_groups, alpha, threshold)
    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    first_window_processed = False

    for index, row in timed_data.iterrows():
        if row['new_window']:
            # print("new window, index = {}".format(index))
            uf_list.append(copy.deepcopy(DFMonitor.uf))
            counter_list_correct.append(copy.deepcopy(counter_values_correct))
            counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
            accuracy_list.append([counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                                  for j in range(len(counter_values_correct))])
            counter_values_incorrect = [x * alpha for x in counter_values_incorrect]
            counter_values_correct = [x * alpha for x in counter_values_correct]
            if not first_window_processed:
                uf = [True if counter_values_correct[j] /
                              (counter_values_correct[j] + counter_values_incorrect[j]) <= threshold
                      else False for j in range(len(counter_values_correct))]
                delta = [round(abs(threshold * (counter_values_correct[j] + counter_values_incorrect[j]) -
                                   counter_values_correct[j]) * alpha, config.decimal)
                         for j in range(len(counter_values_incorrect))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'
        if first_window_processed:
            DFMonitor.insert(row, label)
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == 'correct':
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor.uf))
    counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
    counter_list_correct.append(copy.deepcopy(counter_values_correct))
    accuracy = [counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                if counter_values_correct[j] + counter_values_incorrect[j] != 0
                else None for j in range(len(counter_values_correct))]
    accuracy_list.append(accuracy)
    return DFMonitor, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect


def traverse_data_DF_baseline_only(timed_data, date_column, time_window_str, date_time_format, monitored_groups,
                                     threshold, alpha, label_prediction="predicted", label_ground_truth="ground_truth"):
    global time1
    if date_time_format:
        window_size, window_type = time_window_str.split()
        # Apply the function to compute the window key for each row
        first_row_time = timed_data.loc[0, date_column]
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                                 args=(window_type, int(window_size), first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    DFMonitor_baseline = Accuracy_baseline.DF_Accuracy_Fixed_Window_Counter(monitored_groups, alpha, threshold)
    first_window_processed = False
    first_row_id = timed_data[timed_data['new_window'] == True].index[0]
    num_items_after_first_window = len(timed_data) - first_row_id
    total_time_new_window = 0
    # number of new windows
    num_new_windows = len(timed_data[timed_data['new_window'] == True])
    for index, row in timed_data.iterrows():
        if row['new_window']:
            timea = time.time()
            DFMonitor_baseline.new_window()
            timeb = time.time()
            total_time_new_window += timeb - timea
            if not first_window_processed:
                first_window_processed = True
                time1 = time.time()
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'
        DFMonitor_baseline.insert(row, label)
    time2 = time.time()
    elapsed_time = time2 - time1
    return DFMonitor_baseline, elapsed_time, total_time_new_window, num_items_after_first_window, num_new_windows




def traverse_data_DFMonitor_baseline(timed_data, date_column, time_window_str, date_time_format, monitored_groups,
                                     threshold, alpha, label_prediction="predicted", label_ground_truth="ground_truth"):
    if date_time_format:
        window_size, window_type = time_window_str.split()
        # Apply the function to compute the window key for each row
        first_row_time = timed_data.loc[0, date_column]
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                                 args=(window_type, int(window_size), first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    DFMonitor_baseline = Accuracy_baseline.DF_Accuracy_Fixed_Window_Counter(monitored_groups, alpha, threshold)
    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            # print("new window, index = {}".format(index))
            uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
            counter_list_correct.append(copy.deepcopy(counter_values_correct))
            counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
            accuracy_list.append([counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                                  for j in range(len(counter_values_correct))])
            counter_values_incorrect = [x * alpha for x in counter_values_incorrect]
            counter_values_correct = [x * alpha for x in counter_values_correct]
            DFMonitor_baseline.new_window()
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'
        DFMonitor_baseline.insert(row, label)
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == 'correct':
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
    # last window:
    uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
    counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
    counter_list_correct.append(copy.deepcopy(counter_values_correct))
    accuracy = [counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                if counter_values_correct[j] + counter_values_incorrect[j] != 0
                else None for j in range(len(counter_values_correct))]
    accuracy_list.append(accuracy)
    return DFMonitor_baseline, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect


def Accuracy_traditional(timed_data, date_column, time_window_str, date_time_format, monitored_groups, threshold,
                   label_prediction="predicted", label_ground_truth="ground_truth"):
    if date_time_format:
        window_size, window_type = time_window_str.split()
        # Apply the function to compute the window key for each row
        first_row_time = timed_data.loc[0, date_column]
        timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                                 args=(window_type, int(window_size), first_row_time))
    else:
        timed_data['window_key'] = timed_data[date_column]
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.loc[0, "new_window"] = False
    counter_values_correct = [0] * len(monitored_groups)
    counter_values_incorrect = [0] * len(monitored_groups)
    counter_list_correct = []
    counter_list_incorrect = []
    uf_list = []
    accuracy_list = []
    for index, row in timed_data.iterrows():
        if row['new_window']:
            # print("new window, index = {}".format(index))
            accuracy = [counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                        if counter_values_correct[j] + counter_values_incorrect[j] != 0
                        else None for j in range(len(counter_values_correct))]
            counter_list_correct.append(copy.deepcopy(counter_values_correct))
            counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
            accuracy_list.append(accuracy)
            uf_list.append([True if accuracy[i] <= threshold else False for i in range(len(accuracy))])
            counter_values_correct = [0] * len(monitored_groups)
            counter_values_incorrect = [0] * len(monitored_groups)
        if row[label_prediction] == row[label_ground_truth]:
            label = 'correct'
        else:
            label = 'incorrect'

        # all value in counter_map of the group that the tuple satisfies add 1
        for i, g in enumerate(monitored_groups):
            if belong_to_group(row, g):
                if label == 'correct':
                    counter_values_correct[i] += 1
                else:
                    counter_values_incorrect[i] += 1
    # last window:
    accuracy = [counter_values_correct[j] / (counter_values_correct[j] + counter_values_incorrect[j])
                if counter_values_correct[j] + counter_values_incorrect[j] != 0
                else None for j in range(len(counter_values_correct))]
    accuracy_list.append(accuracy)
    counter_list_correct.append(copy.deepcopy(counter_values_correct))
    counter_list_incorrect.append(copy.deepcopy(counter_values_incorrect))
    uf_list.append([True if accuracy[k] <= threshold else False for k in range(len(accuracy))])
    return uf_list, accuracy_list, counter_list_correct, counter_list_incorrect


if __name__ == "__main__":

    data = pd.read_csv('../../data/name_gender/baby_names_1880_2020_predicted.csv')
    print(data["sex"].unique())
    date_column = "year"
    date_time_format = False
    time_window_str = "10 year"
    monitored_groups = [{"sex": 'male'}, {"sex": 'female'}]
    print(data[:5])
    alpha = 0.5
    threshold = 0.8
    label_prediction = "sex"
    label_ground_truth = "predicted_gender"

    (DFMonitor_baseline, uf_list_baseline, accuracy_list_baseline, counter_list_correct_baseline,
     counter_list_incorrect_baseline) = traverse_data_DFMonitor_baseline(data, date_column,
                                                                         time_window_str,
                                                                         date_time_format,
                                                                         monitored_groups,
                                                                         threshold,
                                                                         alpha, label_prediction, label_ground_truth)

    # use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
    DFMonitor, uf_list_DF, accuracy_list_DF, counter_list_correct_DF, counter_list_incorrect_DF \
        = traverse_data_DFMonitor(data, date_column,
                                  time_window_str,
                                  date_time_format,
                                  monitored_groups,
                                  threshold,
                                  alpha, label_prediction, label_ground_truth)

    # uf_list_tra, accuracy_list_trad, counter_list_correct_trad, counter_list_incorrect_trad \
    #     = CR_traditional(data, date_column,
    #                      time_window_str, date_time_format,
    #                      monitored_groups,
    #                      threshold, label_prediction, label_ground_truth)

    for i in range(0, len(accuracy_list_DF)):
        print(accuracy_list_DF[i], accuracy_list_baseline[i])
        if accuracy_list_baseline[i] != accuracy_list_DF[i]:
            print("accuracy_list_baseline {} != accuracy_list_DF {}".format(accuracy_list_baseline[i],
                                                                            accuracy_list_DF[i]))

    # for i in range(0, len(accuracy_list_trad)):
    #     print("accuracy_list_trad {}  accuracy_list_baseline {}  "
    #           "accuracy_list_DF {}".format(accuracy_list_trad[i], accuracy_list_baseline[i], accuracy_list_DF[i]))
