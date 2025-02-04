from river import linear_model
from river import preprocessing
from river import metrics
from river import datasets
import math

from scipy.fft import ifft2
from sklearn.preprocessing import MinMaxScaler
import csv
import pandas as pd
from algorithm.fixed_window import CR_0_20240118 as CR, CR_baseline_0_20240118 as CR_baseline



def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


#
#
# # Load your data (replace 'your_data.csv' with your actual file)
# decay_function = "exponential_decay"
# decay_rate = 5
# data_stream = pd.read_csv(f'../{decay_function}_filtered_data_decay_rate_{decay_rate}.csv')
#
#
time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
data_stream = pd.read_csv(f'../../../../../../data/stocks_nanosecond/{data_file_name}.csv')


use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True
window_size_nanoseconds = 100000
checking_interval_nanoseconds = 100000



date_column = 'ts_event'

time_delta = 0
batch_results = []
data_stream[date_column] = pd.to_datetime(data_stream[date_column])
previous_event_timestamp = data_stream[date_column].iloc[0]
previous_check_timestamp = data_stream[date_column].iloc[0]

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Financial Services'}, {"sector": 'Consumer Defensive'}, {"sector": 'Healthcare'},
                    {"sector": 'Energy'}]
sectors_list = [x['sector'] for x in monitored_groups]
print(data_stream[:5])
alpha = 0.9997
threshold = 0.3


# Prepare the result file for writing
# result_file_name = f"dynamic_CR_original_alpha_{str(get_integer(alpha))}.csv"
result_file_name = f"dynamic_CR_{decay_function}_rate_{decay_rate}_alpha_{get_integer(alpha)}.csv"
result_file = open(result_file_name, mode='w', newline='')
csv_writer = csv.writer(result_file, delimiter=',')
csv_writer.writerow(["timestamp", "Technology", "Consumer Cyclical", "Communication Services",
                     "Financial Services", "Consumer Defensive", "Healthcare", "Energy"])


DFMonitor_baseline = CR_baseline.CR_baseline(monitored_groups, alpha, threshold)
counters_all = 0
counters_groups = [0] * len(monitored_groups)
first_window_processed = False
checking_record_cr = []
checking_record_counters = []
checking_record_all = []




time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')


data_stream[date_column] = pd.to_datetime(data_stream[date_column])
# get data between time_start and time_end
data_stream = data_stream[(data_stream[date_column] >= time_start) & (data_stream[date_column] <= time_end)]




pre_timestamp = data_stream[date_column].iloc[0]
cur_timestamp = data_stream[date_column].iloc[0]

for idx, row in data_stream.iterrows():
    sector = row['sector']
    if sector not in sectors_list:
        continue
    cur_timestamp = row[date_column]
    time_delta_window = (cur_timestamp - previous_event_timestamp).total_seconds() * (1e9 if use_two_counters else 1)
    time_delta_check = (cur_timestamp - previous_check_timestamp).total_seconds() * (1e9 if use_two_counters else 1)
    if time_delta_check > checking_interval_nanoseconds:
        while (previous_check_timestamp + pd.Timedelta(
                nanoseconds=checking_interval_nanoseconds)) < cur_timestamp:
            previous_check_timestamp = previous_check_timestamp + pd.Timedelta(
                nanoseconds=checking_interval_nanoseconds)
            checking_record_all.append(counters_all)
            checking_record_counters.append(counters_groups)
            all_cur_cr = [x / counters_all if counters_all != 0 else 0 for x in counters_groups]
            checking_record_cr.append(all_cur_cr)
            csv_writer.writerow([previous_check_timestamp] + all_cur_cr)

    if time_delta_window > window_size_nanoseconds:
        num_window = 0
        while (previous_event_timestamp + pd.Timedelta(
                nanoseconds=window_size_nanoseconds)) < cur_timestamp:
            previous_event_timestamp = previous_event_timestamp + + pd.Timedelta(
                nanoseconds=window_size_nanoseconds)
            num_window += 1
            DFMonitor_baseline.new_window()
        # print("num_window", num_window)
        counters_all = counters_all * math.pow(alpha, num_window)
        counters_groups = [x * math.pow(alpha, num_window) for x in counters_groups]
    DFMonitor_baseline.insert(row)
    # insert to counters
    counters_all += 1
    counters_groups[sectors_list.index(sector)] += 1

print("counters_all", counters_all)
print("counters_groups", counters_groups)
print("counters_groups sum", sum(counters_groups))




