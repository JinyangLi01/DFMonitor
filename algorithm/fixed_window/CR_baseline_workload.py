import pandas as pd
from algorithm.fixed_window import CR_baseline_0_20240118 as CR_baseline

data = pd.read_csv('../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
# get distribution of compas_screening_date
data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
data['compas_screening_date'].hist()


# use CR to monitor tpr of different races over time
# a time window is a month

# Function to compute the time window key (e.g., year-month for 'month' windows)
def compute_time_window_key(row, window_type):
    if window_type == 'year':
        return row.year
    elif window_type == 'month':
        return "{}-{}".format(row.year, row.month)
    elif window_type == 'week':
        return "{}-{}".format(row.year, row.strftime('%U'))
    elif window_type == 'day':
        return "{}-{}-{}".format(row.year, row.month, row.day)


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def monitorCR_baseline(timed_data, date_column, time_window_str, monitored_groups, threshold, alpha):
    number, unit = time_window_str.split()

    # Apply the function to compute the window key for each row
    timed_data['window_key'] = timed_data[date_column].apply(compute_time_window_key, args=(unit,))
    # Determine the start of a new window
    timed_data['new_window'] = timed_data['window_key'] != timed_data['window_key'].shift(1)
    timed_data["new_window"].loc[0] = False
    DFMonitor = CR_baseline.CR_baseline(monitored_groups, alpha, threshold)
    DFMonitor.print()

    def process_each_tuple(row, DFMonitor):
        if row['new_window']:  # new window
            print("new window, row={}, {}".format(row['id'], row['compas_screening_date']))
            DFMonitor.print()
            DFMonitor.new_window()
        DFMonitor.insert(row)
        return

    timed_data.apply(process_each_tuple, axis=1, args=(DFMonitor,))


# monitored_groups = [{"race": 'African-American'}, {"race": 'Caucasian'}]
# alpha = 0.5
# threshold = 0.1
# monitorCR_baseline(data, "compas_screening_date", "1 month", monitored_groups, threshold, alpha)
#
