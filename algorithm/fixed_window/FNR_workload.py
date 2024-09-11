import pandas as pd
from algorithm.fixed_window import FPR_0_20240125 as FPR, config, FPR_baseline_0_20240125 as FPR_baseline
import copy
import time


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



def traverse_data_DFMonitor_and_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                         label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    counter_first_window_FN = [0 for _ in monitored_groups]
    counter_first_window_TP = [0 for _ in monitored_groups]
    first_window_processed = False

    for index, row in timed_data.iterrows():
        if row['new_window']:
            DFMonitor_baseline.new_window()
            if not first_window_processed:
                uf = []
                for i in range(len(monitored_groups)):
                    if counter_first_window_FN[i] + counter_first_window_TP[i] == 0:
                        uf.append(None)
                    else:
                        if (counter_first_window_FN[i] / (counter_first_window_TP[i] + counter_first_window_FN[i])
                                > threshold):
                            uf.append(True)
                        else:
                            uf.append(False)
                delta = [0] * len(monitored_groups)

                for i in range(len(monitored_groups)):
                    delta[i] = threshold * counter_first_window_TP[i] - (1 - threshold) * counter_first_window_FN[i]
                delta = [
                    round(abs(threshold * counter_first_window_TP[i] - (1 - threshold) * counter_first_window_FN[i])
                          * alpha, config.decimal) for i in range(len(monitored_groups))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1)
                or (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"
            DFMonitor_baseline.insert(row, label)
            if not first_window_processed:
                for group_idx, group in enumerate(monitored_groups):
                    if belong_to_group(row, group):
                        if label == "FN":
                            counter_first_window_FN[group_idx] += 1
                        else:
                            counter_first_window_TP[group_idx] += 1
            else:
                DFMonitor.insert(row, label)
        if first_window_processed:
            if DFMonitor.uf != DFMonitor_baseline.uf:
                print("+++++++ after loop not equal **********")
                print("index === {}, row={}, {}".format(index, row['id'], row['compas_screening_date']))
                DFMonitor_baseline.print()
                DFMonitor.print()
                return DFMonitor, DFMonitor_baseline
    return DFMonitor, DFMonitor_baseline


def traverse_data_DFMonitor_only(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                 label_prediction="predicted", label_ground_truth="ground_truth"):
    global time1
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    counter_values_FN = [0 for _ in monitored_groups]
    counter_values_TP = [0 for _ in monitored_groups]
    first_window_processed = False
    first_row_id = timed_data[timed_data['new_window'] == True].index[0]
    num_items_after_first_window = len(timed_data) - first_row_id
    total_time_new_window = 0
    num_new_windows = len(timed_data[timed_data['new_window'] == True])

    for index, row in timed_data.iterrows():
        if row['new_window']:
            if not first_window_processed:
                uf = [True if counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i]) <= threshold
                      else False
                      for i in range(len(monitored_groups))]
                delta = [round(abs(threshold * counter_values_TP[i] - (1 - threshold) * counter_values_FN[i])
                               * alpha, config.decimal)
                         for i in range(len(monitored_groups))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
                time1 = time.time()
            else:
                timea = time.time()
                DFMonitor.new_window()
                timeb = time.time()
                total_time_new_window += timeb - timea
        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1) or
                (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"
            if first_window_processed:
                DFMonitor.insert(row, label)
            else:
                for i, g in enumerate(monitored_groups):
                    if belong_to_group(row, g):
                        if label == "FN":
                            counter_values_FN[i] += 1
                        else:
                            counter_values_TP[i] += 1
    time2 = time.time()
    elapsed_time = time2 - time1
    return DFMonitor, elapsed_time, total_time_new_window, num_items_after_first_window, num_new_windows

def traverse_data_DFMonitor(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                            label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor = FPR.DF_FPR(monitored_groups, alpha, threshold)
    counter_values_FN = [0 for _ in monitored_groups]
    counter_values_TP = [0 for _ in monitored_groups]
    counter_list_FN = []
    counter_list_TP = []
    first_window_processed = False
    uf_list = []
    fnr_list = []

    for index, row in timed_data.iterrows():
        if row['new_window']:
            uf_list.append(copy.deepcopy(DFMonitor.uf))
            deep_copy = copy.deepcopy(counter_values_FN)
            counter_list_FN.append(deep_copy)
            deep_copy = copy.deepcopy(counter_values_TP)
            counter_list_TP.append(deep_copy)
            fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
                   if counter_values_TP[i] + counter_values_FN[i] != 0
                   else None for i in range(len(monitored_groups))]
            fnr_list.append(fnr)
            counter_values_FN = [x * alpha for x in counter_values_FN]
            counter_values_TP = [x * alpha for x in counter_values_TP]
            if not first_window_processed:
                uf = [True if counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i]) <= threshold
                      else False
                      for i in range(len(monitored_groups))]
                delta = [round(abs(threshold * counter_values_TP[i] - (1 - threshold) * counter_values_FN[i])
                               * alpha, config.decimal)
                         for i in range(len(monitored_groups))]
                DFMonitor.initialization(uf, delta)
                first_window_processed = True
            else:
                DFMonitor.new_window()
        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1) or
                (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"
            DFMonitor.insert(row, label)
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FN":
                        counter_values_FN[i] += 1
                    else:
                        counter_values_TP[i] += 1
    uf_list.append(copy.deepcopy(DFMonitor.uf))
    counter_list_FN.append(copy.deepcopy(counter_values_FN))
    counter_list_TP.append(copy.deepcopy(counter_values_TP))
    fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
           if counter_values_TP[i] + counter_values_FN[i] != 0
           else None for i in range(len(monitored_groups))]
    fnr_list.append(fnr)
    return DFMonitor, uf_list, fnr_list, counter_list_TP, counter_list_FN


def traverse_data_DF_baseline_only(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                                   label_prediction="predicted", label_ground_truth="ground_truth"):
    global time1
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(window_type,
                                                                                            int(window_size),
                                                                                            first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    first_window_processed = False
    first_row_id = timed_data[timed_data['new_window'] == True].index[0]
    num_items_after_first_window = len(timed_data) - first_row_id
    total_time_new_window = 0
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
        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1) or
                (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"
            DFMonitor_baseline.insert(row, label)
    time2 = time.time()
    elapsed_time = time2 - time1
    return DFMonitor_baseline, elapsed_time, total_time_new_window, num_items_after_first_window, num_new_windows


def traverse_data_DF_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha,
                              label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(window_type,
                                                                                            int(window_size),
                                                                                            first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    DFMonitor_baseline = FPR_baseline.FPR_baseline(monitored_groups, alpha, threshold)
    uf_list = []
    fnr_list = []
    counter_values_FN = [0 for _ in monitored_groups]
    counter_values_TP = [0 for _ in monitored_groups]
    counter_list_FN = []
    counter_list_TP = []

    for index, row in timed_data.iterrows():
        if row['new_window']:
            print("new window, index = {}, date = {}, before this point FN = {}, "
                  "TP = {}".format(index, row[date_column], counter_values_FN, counter_values_TP))
            uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
            deep_copy = copy.deepcopy(counter_values_FN)
            counter_list_FN.append(deep_copy)
            deep_copy = copy.deepcopy(counter_values_TP)
            counter_list_TP.append(deep_copy)
            fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
                   if counter_values_TP[i] + counter_values_FN[i] != 0
                   else None for i in range(len(monitored_groups))]
            fnr_list.append(fnr)

            DFMonitor_baseline.new_window()
            counter_values_FN = [x * alpha for x in counter_values_FN]
            counter_values_TP = [x * alpha for x in counter_values_TP]
        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1) or
                (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"

            DFMonitor_baseline.insert(row, label)
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FN":
                        counter_values_FN[i] += 1
                    else:
                        counter_values_TP[i] += 1
    uf_list.append(copy.deepcopy(DFMonitor_baseline.uf))
    counter_list_FN.append(copy.deepcopy(counter_values_FN))
    counter_list_TP.append(copy.deepcopy(counter_values_TP))
    fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
           if counter_values_TP[i] + counter_values_FN[i] != 0
           else None for i in range(len(monitored_groups))]
    fnr_list.append(fnr)
    return DFMonitor_baseline, uf_list, fnr_list, counter_list_TP, counter_list_FN


def FNR_traditional(timed_data, date_column, time_window_str, monitored_groups, threshold,
                    label_prediction="predicted", label_ground_truth="ground_truth"):
    window_size, window_type = time_window_str.split()
    first_row_time = timed_data.loc[0, date_column]
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key,
                                                             args=(window_type, int(window_size), first_row_time))
    timed_data['new_window'] = False
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data.at[0, "new_window"] = False
    counter_values_FN = [0 for _ in monitored_groups]
    counter_values_TP = [0 for _ in monitored_groups]
    counter_list_FN = []
    counter_list_TP = []
    fnr_list = []
    uf_list = []

    for index, row in timed_data.iterrows():
        if row['new_window']:
            fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
                   if counter_values_TP[i] + counter_values_FN[i] != 0
                   else None for i in range(len(monitored_groups))]
            print("index = {}, fnr = {}, date = {}".format(index, fnr, row[date_column]))
            counter_list_FN.append(copy.deepcopy(counter_values_FN))
            counter_list_TP.append(copy.deepcopy(counter_values_TP))
            fnr_list.append(fnr)
            uf_list.append(
                [True if fnr[i] is not None and fnr[i] <= threshold else False if fnr[i] is not None else None for i
                 in range(len(fnr))])
            counter_values_FN = [0 for _ in monitored_groups]
            counter_values_TP = [0 for _ in monitored_groups]

        if ((row[label_prediction] == 0 and row[label_ground_truth] == 1) or
                (row[label_prediction] == 1 and row[label_ground_truth] == 1)):
            if row[label_prediction] == 0 and row[label_ground_truth] == 1:
                label = "FN"
            else:
                label = "TP"
            for i, g in enumerate(monitored_groups):
                if belong_to_group(row, g):
                    if label == "FN":
                        counter_values_FN[i] += 1
                    else:
                        counter_values_TP[i] += 1
    fnr = [counter_values_FN[i] / (counter_values_TP[i] + counter_values_FN[i])
           if counter_values_TP[i] + counter_values_FN[i] != 0
           else None for i in range(len(monitored_groups))]
    counter_list_FN.append(copy.deepcopy(counter_values_FN))
    counter_list_TP.append(copy.deepcopy(counter_values_TP))
    fnr_list.append(fnr)
    uf_list.append(
        [True if fnr[i] is not None and fnr[i] <= threshold else False if fnr[i] is not None else None for i in
         range(len(fnr))])
    return uf_list, fnr_list, counter_list_TP, counter_list_FN


if __name__ == "__main__":
    data = pd.read_csv('../../experiments/movielens/river100K/result_hoeffding_adaptive_classifier.csv')
    print(data["gender"].unique())
    date_column = "datetime"
    # get distribution of compas_screening_date
    data[date_column] = pd.to_datetime(data[date_column])
    date_time_format = True
    time_window_str = "1 month"
    monitored_groups = [{"gender": "M"}, {"gender": "F"}]

    alpha = 0.5
    threshold = 0.3
    label_prediction = "prediction"
    label_ground_truth = "rating"

    DFMonitor, uf_list_DF, fnr_list_DF, counter_list_TP_DF, counter_list_FN_DF \
        = traverse_data_DFMonitor(data,
                                           date_column,
                                           time_window_str,
                                           monitored_groups,
                                           threshold,
                                           alpha, label_prediction, label_ground_truth)
    print(fnr_list_DF)
    #
    # DFMonitor, uf_list2, fpr_list2, counter_list_TN2, counter_list_FP2 = traverse_data_DFMonitor(data, date_column,
    #                                                                                              time_window_str,
    #                                                                                              monitored_groups,
    #                                                                                              threshold, alpha)

    # uf_list, fpr_list, counter_list_TN, counter_list_FP = FPR_traditional(data, "compas_screening_date", "1 month",
    #                                                                       monitored_groups,
    #                                                                       threshold)

    # DFMonitor, DFMonitor_baseline = traverse_data_DFMonitor_and_baseline(data, date_column, time_window_str,
    #                                                                      monitored_groups, threshold, alpha)

    #
    #
    # # compare fpr1 and fpr2
    # def compare_fpr(fpr_list1, fpr_list2):
    #     for i in range(len(fpr_list1)):
    #         for j in range(len(fpr_list1[i])):
    #             if fpr_list1[i][j] != fpr_list2[i][j]:
    #                 print(
    #                     "fpr_list1[i][j] = {}, fpr_list2[i][j] = {}, i={}, j={}".format(fpr_list1[i][j],
    #                                                                                     fpr_list2[i][j], i, j))
    #                 return False
    #     return True
    #
    #
    # print(compare_fpr(fpr_list1, fpr_list2))
    # print(compare_fpr(counter_list_FP1, counter_list_FP2))
    # print(compare_fpr(counter_list_TN1, counter_list_TN2))
    # print(uf_list1 == uf_list2)
    #
    # print(fpr_list1[12:])
    # print(fpr_list2[12:])
