"""
this file is used for sanity check of the CR algorithm
In thie script, I traverse all rows of data frame and compute the time-decaying value of CR
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import CR_0_20240118 as CR
from algorithm import CR_baseline_0_20240118 as CR_baseline


timed_data = pd.read_csv('../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
# get distribution of compas_screening_date
timed_data['compas_screening_date'] = pd.to_datetime(timed_data['compas_screening_date'])


monitored_groups = [{"race": 'African-American'}, {"race": 'Caucasian'}]
alpha = 0.5
threshold = 0.3
time_window_str = '1 month'
date_column = 'compas_screening_date'
number, unit = time_window_str.split()


# Apply the function to compute the window key for each row
def compute_time_window_key(row, window_type):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        return f"{row.year}-{row.month}"
    elif window_type == 'week':
        return f"{row.year}-{row.week}"
    elif window_type == 'day':
        return f"{row.year}-{row.month}-{row.day}"


timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
# Initialize all rows as not the start of a new window
timed_data['new_window'] = False
# Determine the start of a new window for all rows except the first
timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
timed_data["new_window"].loc[0] = False
counter_Caucasian = 0
counter_African_American = 0
total_counter = 0

for index, row in timed_data.iterrows():
    # if index==600 or index == 601:
    #     print(counter_Caucasian/total_counter <= threshold, counter_African_American/total_counter <= threshold, counter_Caucasian, counter_African_American, total_counter, counter_Caucasian/total_counter, counter_African_American/total_counter)

    if row['new_window'] == True:
        counter_Caucasian *= alpha
        counter_African_American *= alpha
        total_counter *= alpha
    if row['race'] == 'Caucasian':
        counter_Caucasian += 1
    elif row['race'] == 'African-American':
        counter_African_American += 1
    total_counter += 1



def monitorCR_vanilla(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    def belong_to_group(tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    # Apply the function to compute the window key for each row
    def compute_time_window_key(row, window_type):
        if window_type == 'year':
            return row.year
        elif window_type == 'month':
            return f"{row.year}-{row.month}"
        elif window_type == 'week':
            return f"{row.year}-{row.week}"
        elif window_type == 'day':
            return f"{row.year}-{row.month}-{row.day}"

    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Initialize all rows as not the start of a new window
    timed_data['new_window'] = False
    # Determine the start of a new window for all rows except the first
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data["new_window"].loc[0] = False
    counter_map = {g: 0 for g in monitored_groups}
    total_counter = 0

    for index, row in timed_data.iterrows():
        if row['new_window'] == True:
            # all value in counter_map timed by alpha
            for g in monitored_groups:
                counter_map[g] *= alpha
            total_counter *= alpha
        # all value in counter_map of the group that the tuple satisfies add 1
        for g in monitored_groups:
            if belong_to_group(row, g):
                counter_map[g] += 1
        total_counter += 1
