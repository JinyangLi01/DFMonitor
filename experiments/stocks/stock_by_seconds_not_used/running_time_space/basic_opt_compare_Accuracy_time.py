import csv
# import colorsys
import math
import pandas as pd
# import matplotlib.pyplot as plt
import pytz
import datetime

# from algorithm.fixed_window import Accuracy_workload as workload

from algorithm.per_item import Accuracy_workload as workload

import seaborn as sns


# def scale_lightness(rgb, scale_l):
#     # convert rgb to hls
#     h, l, s = colorsys.rgb_to_hls(*rgb)
#     # manipulate h, l, s values and return as rgb
#     return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

# plt.figure(figsize=(6, 3.5))
# plt.rcParams['font.size'] = 20


method_name = "logistic_regression"
exmained_time = "10-0--12-0_ny_time"
chunk_size=1
date = "2024-10-15"
data = pd.read_csv(f'../stock_2024-10-15/2h_interval/prediction_result_s&p50_all_stocks_{date}_'
                   f'sorted_with_binary_direction_{exmained_time}_chunk_size_{chunk_size}.csv')
print("len of original data", len(data))
print(data["sector"].unique())
date_column = "datetime_ny"
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True


# Define the New York time zone
ny_tz = pytz.timezone('America/New_York')

time_start = ny_tz.localize(datetime.datetime.now().replace(year=2024, month=10, day=15, hour=11, minute=0, second=0, microsecond=0))
# Get the current date and set the time to 14:00 (2:00 PM)
time_end = ny_tz.localize(datetime.datetime.now().replace(year=2024, month=10, day=15, hour=11, minute=2, second=0, microsecond=0))

# get data between time_start and time_end
data = data[(data["datetime_ny"] >= time_start) & (data["datetime_ny"] <= time_end)]

# reset the index
data = data.reset_index(drop=True)
print("len of selected data", len(data))

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Financial Services'}, {"sector": 'Consumer Defensive'}, {"sector": 'Healthcare'}]
print(data[:5])
alpha = 0.997


threshold = 0.3
label_prediction = "predicted_direction"
label_ground_truth = "binary_direction"
correctness_column = "binary_correctness"
use_two_counters = True
time_unit = "1 second"
window_size_units = "1"
checking_interval = "1 second"


window_size_str_list = ["2 year", "5 year", "10 year", "20 year", "40 year", "60 year", "80 year", "100 year"]
window_size_str_list_brev = ["2", "5", "10", "20", "40", "60", "80", "100"]



########################################## start running time ##########################################

def one_run(monitored_group, num_repeat=1):

    total_elapsed_time_bit = 0
    total_elapsed_time_counter = 0
    total_elasped_time_new_window_bit = 0
    total_elasped_time_new_window_counter = 0
    total_elapsed_time_insertion_bit = 0
    total_elapsed_time_insertion_counter = 0
    total_num_time_window = 0

    for i in range(num_repeat):
        DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
            window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
            total_time_insertion_bit, total_time_insertion_counter, \
            total_time_new_window_bit, total_time_new_window_counter, num_time_window \
            = workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                            date_time_format,
                                                            monitored_groups,
                                                            threshold, alpha, time_unit, eval(window_size_units),
                                                            checking_interval, label_prediction,
                                                            label_ground_truth, correctness_column, use_two_counters)



        total_elapsed_time_bit += elapsed_time_DFMonitor_bit
        total_elapsed_time_counter += elapsed_time_DFMonitor_counter
        total_elasped_time_new_window_bit += total_time_new_window_bit
        total_elasped_time_new_window_counter += total_time_new_window_counter
        total_elapsed_time_insertion_bit += total_time_insertion_bit
        total_elapsed_time_insertion_counter += total_time_insertion_counter
        total_num_time_window += num_time_window
    print("monitored groups: ", monitored_groups)
    # print("avg elapsed_time_base out of {} executions: {}".format(num_repeat, total_elapsed_time_base / num_repeat))
    # print("avg elapsed_time_opt out of {} executions: {}".format(num_repeat, total_elapsed_time_opt / num_repeat))
    # print("num_items_after_first_window_base: ", num_items_after_first_window_base)
    # print("avg time of an operation base: ", total_elapsed_time_base / (num_items_after_first_window_base * num_repeat))
    # print("avg time of an operation opt: ", total_elapsed_time_opt / (num_items_after_first_window_base * num_repeat))
    num_item = len(data)
    avg_elapsed_time_base_per_item = total_elapsed_time_bit / num_repeat / num_item
    avg_elapsed_time_counter_per_item = total_elapsed_time_counter / num_repeat / num_item
    avg_new_window_bit = total_elasped_time_new_window_bit / total_num_time_window
    avg_new_window_counter = total_elasped_time_new_window_counter / total_num_time_window
    avg_insertion_bit_per_item = total_elapsed_time_insertion_bit / num_repeat / num_item
    avg_insertion_counter_per_item = total_elapsed_time_insertion_counter / num_repeat / num_item

    return (avg_elapsed_time_base_per_item, avg_elapsed_time_counter_per_item, avg_new_window_bit,
            avg_new_window_counter, avg_insertion_bit_per_item, avg_insertion_counter_per_item)





monitored_groups_set = [[{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}],
    [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
     {"sector": 'Financial Services'}, {"sector": 'Consumer Defensive'}, {"sector": 'Healthcare'}]]



avg_elapsed_time_counter_each_group_set, avg_elapsed_time_bit_each_group_set = 0, 0
avg_insertion_bit_each_group_set, avg_insertion_counter_each_group_set = 0, 0
avg_new_window_bit_each_group_set, avg_new_window_counter_each_group_set = 0, 0

result_insertion_bit = []
result_insertion_counter = []
result_new_window_bit = []
result_new_window_counter = []
result_elapsed_bit = []
result_elapsed_counter = []

repeat = 1
for i, group in enumerate(monitored_groups_set):
    print("group: ", group, "i: ", i)
    (avg_elapsed_time_bit_per_item, avg_elapsed_time_counter_per_item, avg_new_window_bit,
     avg_new_window_counter, avg_insertion_bit_per_item, avg_insertion_counter_per_item) = (
        one_run(monitored_groups, 1))
    avg_elapsed_time_counter_each_group_set += avg_elapsed_time_counter_per_item
    avg_elapsed_time_bit_each_group_set += avg_elapsed_time_bit_per_item
    avg_insertion_bit_each_group_set += avg_insertion_bit_per_item
    avg_insertion_counter_each_group_set += avg_insertion_counter_per_item
    avg_new_window_counter_each_group_set += avg_new_window_counter
    avg_new_window_bit_each_group_set += avg_new_window_bit

    result_insertion_bit.append(avg_insertion_bit_per_item)
    result_insertion_counter.append(avg_insertion_counter_per_item)
    result_new_window_bit.append(avg_new_window_bit)
    result_new_window_counter.append(avg_new_window_counter)
    result_elapsed_bit.append(avg_elapsed_time_bit_per_item)
    result_elapsed_counter.append(avg_elapsed_time_counter_per_item)


with open("running_time_Accuracy.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["avg_elapsed_time_counter", "avg_elapsed_time_bit", "avg_new_window_counter", "avg_new_window_bit",
                     "avg_insertion_counter", "avg_insertion_bit"])
    writer.writerow([avg_elapsed_time_counter_each_group_set/len(monitored_groups_set),
                     avg_elapsed_time_bit_each_group_set/len(monitored_groups_set),
                     avg_new_window_counter_each_group_set/len(monitored_groups_set),
                     avg_new_window_bit_each_group_set/len(monitored_groups_set),
                     avg_insertion_counter_each_group_set/len(monitored_groups_set),
                     avg_insertion_bit_each_group_set/len(monitored_groups_set)])
    print("\n\n\n")
    print("difference between two results by percentage. Each column is the average value of a set of monitored groups")
    # the difference between two results by percentage
    writer.writerow(["insert_time", [(result_insertion_counter[i] - result_insertion_bit[i]) / result_insertion_counter[i] for i in range(len(result_insertion_counter))]])
    writer.writerow(["new_window_time", [(result_new_window_counter[i] - result_new_window_bit[i]) / result_new_window_counter[i] for i in range(len(result_new_window_counter))]])
    writer.writerow(["elapsed_time", [(result_elapsed_counter[i] - result_elapsed_bit[i]) / result_elapsed_counter[i] for i in range(len(result_elapsed_counter))]])



print("avg_elapsed_time_counter: ", avg_elapsed_time_counter_each_group_set/len(monitored_groups_set))
print("avg_elapsed_time_bit: ", avg_elapsed_time_bit_each_group_set/len(monitored_groups_set))
print("avg_insertion_counter: ", avg_insertion_counter_each_group_set/len(monitored_groups_set))
print("avg_insertion_bit: ", avg_insertion_bit_each_group_set/len(monitored_groups_set))
print("avg_new_window_counter: ", avg_new_window_counter_each_group_set/len(monitored_groups_set))
print("avg_new_window_bit: ", avg_new_window_bit_each_group_set/len(monitored_groups_set))

print("the difference between two results by percentage elapsed time: ",
        [(result_elapsed_counter[i] - result_elapsed_bit[i]) / result_elapsed_counter[i] for i in range(len(result_elapsed_counter))])

print("the difference between two results by percentage insertion: ",
      [(result_insertion_counter[i] - result_insertion_bit[i]) / result_insertion_counter[i] for i in range(len(result_insertion_counter))])

print("the difference between two results by percentage new window: ",
      [(result_new_window_counter[i] - result_new_window_bit[i]) / result_new_window_counter[i] for i in range(len(result_new_window_counter))])




########################################## end running time ##########################################
# repeat 1000
# avg elapsed_time_base out of 1000 executions: 0.13568266344070434
# avg elapsed_time_opt out of 1000 executions: 0.13563484144210816
