import csv
# import colorsys
import math
import pandas as pd
# import matplotlib.pyplot as plt
import pytz
import datetime
from line_profiler_pycharm import profile

from algorithm.dynamic_window import Accuracy_workload as workload

import seaborn as sns


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
exmained_time = "14-00--14-10"
chunk_size=1
date = "20241015"
data = pd.read_csv(f"../1h_interval/prediction_result_xnas-itch-{date}_{exmained_time}_chunk_size_{chunk_size}_v3.csv")

print("len of original data", len(data))

date_column = "ts_event"
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True


# Define the New York time zone
# ny_tz = pytz.timezone('America/New_York')

time_start = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')


# get data between time_start and time_end
data = data[(data["ts_event"] >= time_start) & (data["ts_event"] <= time_end)]

# reset the index
# data = data.reset_index(drop=True)
print("len of selected data", len(data))
print(data["sector"].unique())

print(data[:5])
alpha = 0.9997


threshold = 0.3
label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True

#
# num_window_reset = ((data["ts_event"].max() - data["ts_event"].min()).total_seconds() * 1e9 /
#                     int(checking_interval.split(" ")[0]))
#
# print(f"num of window resets: {num_window_reset}")




def processing(time_unit, T_in, T_b, checking_interval, repeat):


    label_prediction = "predicted_direction"
    label_ground_truth = "next_price_direction"
    correctness_column = "prediction_binary_correctness"
    use_two_counters = True
    use_nanosecond = True
    num_window_reset = ((data["ts_event"].max() - data["ts_event"].min()).total_seconds() * 1e9 /
                        int(checking_interval.split(" ")[0]))

    print(f"num of window resets: {num_window_reset}")

########################################## start running time ##########################################

    def one_run(monitored_groups, num_repeat=1):
        total_elapsed_time_bit = 0
        total_elapsed_time_counter = 0
        total_elasped_time_new_window_bit = 0
        total_elasped_time_new_window_counter = 0
        total_elapsed_time_insertion_bit = 0
        total_elapsed_time_insertion_counter = 0
        total_num_time_window = 0
        total_space_bit = 0
        total_space_counter = 0
        total_num_accuracy_check = 0
        total_elapsed_time_query_bit = 0
        total_elapsed_time_query_counter = 0

        for i in range(num_repeat):
            DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
                window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
            time_new_window_bit, time_new_window_counter, \
            time_insertion_bit, time_insertion_counter, num_accuracy_check, \
                time_query_bit, time_query_counter, num_time_window \
                = workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                                date_time_format,
                                                                monitored_groups,
                                                                threshold, alpha, time_unit, T_b, T_in,
                                                                checking_interval, label_prediction,
                                                                label_ground_truth, correctness_column,
                                                                use_two_counters, use_nanosecond)
            print(elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter,
                  time_insertion_bit, time_insertion_counter,
                  time_new_window_bit, time_new_window_counter, num_accuracy_check,
                time_query_bit, time_query_counter, num_time_window)
            if time_insertion_bit + time_new_window_bit != elapsed_time_DFMonitor_bit:
                raise Exception("error, new window time + insertion time != total")
            if time_insertion_counter + time_new_window_counter != elapsed_time_DFMonitor_counter:
                raise Exception("error, new window time + insertion time != total")

            print("insertion time bit: ", time_insertion_bit, "insertion time counter: ", time_insertion_counter)
            print("new window time bit: ", time_new_window_bit, "new window time counter: ", time_new_window_counter)
            total_elapsed_time_bit += elapsed_time_DFMonitor_bit
            total_elapsed_time_counter += elapsed_time_DFMonitor_counter
            total_elasped_time_new_window_bit += time_new_window_bit
            total_elasped_time_new_window_counter += time_new_window_counter
            total_elapsed_time_insertion_bit += time_insertion_bit
            total_elapsed_time_insertion_counter += time_insertion_counter
            total_num_time_window += num_time_window
            total_num_accuracy_check += num_accuracy_check
            total_elapsed_time_query_bit += time_query_bit
            total_elapsed_time_query_counter += time_query_counter
            total_space_bit += DFMonitor_bit.get_size()
            total_space_counter += DFMonitor_counter.get_size()

        print("monitored groups: ", monitored_groups)
        item_num = 1
        avg_elapsed_time_bit_per_item = total_elapsed_time_bit / num_repeat / item_num
        avg_elapsed_time_counter_per_item = total_elapsed_time_counter / num_repeat / item_num
        avg_new_window_bit = total_elasped_time_new_window_bit / total_num_time_window
        avg_new_window_counter = total_elasped_time_new_window_counter / total_num_time_window
        avg_insertion_bit_per_item = total_elapsed_time_insertion_bit / num_repeat / item_num
        avg_insertion_counter_per_item = total_elapsed_time_insertion_counter / num_repeat / item_num
        avg_space_bit = total_space_bit / num_repeat
        avg_space_counter = total_space_counter / num_repeat
        avg_elapsed_time_query_bit = total_elapsed_time_query_bit / total_num_accuracy_check
        avg_elapsed_time_query_counter = total_elapsed_time_query_counter / total_num_accuracy_check
        avg_num_time_window = total_num_time_window / num_repeat
        avg_num_accuracy_check = total_num_accuracy_check / num_repeat

        print("-------------- ", avg_elapsed_time_bit_per_item, avg_elapsed_time_counter_per_item)

        return (avg_elapsed_time_bit_per_item, avg_elapsed_time_counter_per_item, avg_new_window_bit,
                avg_new_window_counter, avg_insertion_bit_per_item, avg_insertion_counter_per_item,
                avg_space_bit, avg_space_counter, avg_elapsed_time_query_bit, avg_elapsed_time_query_counter,
                avg_num_time_window, avg_num_accuracy_check)




    monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Financial Services'}, {"sector": 'Consumer Defensive'}, {"sector": 'Healthcare'}]
    print("monitored_groups: ", monitored_groups)
    (avg_elapsed_time_bit_per_item, avg_elapsed_time_counter_per_item, avg_new_window_bit,
     avg_new_window_counter, avg_insertion_bit_per_item, avg_insertion_counter_per_item,
     avg_space_bit, avg_space_counter, avg_elapsed_time_query_bit, avg_elapsed_time_query_counter,
         avg_num_time_window, avg_num_accuracy_check)   =   (
        one_run(monitored_groups, repeat))



    with open("dynamic_window_running_time_Accuracy_v5.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([avg_elapsed_time_bit_per_item, avg_elapsed_time_counter_per_item, avg_new_window_bit,
     avg_new_window_counter, avg_insertion_bit_per_item, avg_insertion_counter_per_item,
     avg_space_bit, avg_space_counter, avg_elapsed_time_query_bit, avg_elapsed_time_query_counter,
         avg_num_time_window, avg_num_accuracy_check])

    print("\n\n\n")
    print("difference between two results by percentage. Each column is the average value of a set of monitored groups")
    print((avg_elapsed_time_counter_per_item - avg_elapsed_time_bit_per_item) / avg_elapsed_time_bit_per_item)

    print(f"avg_elapsed_time_counter: {avg_elapsed_time_counter_per_item} s = {avg_elapsed_time_counter_per_item*1000} ms ")
    print(f"avg_elapsed_time_bit: {avg_elapsed_time_bit_per_item} s = {avg_elapsed_time_bit_per_item*1000} ms")
    print(f"avg_space_counter: {avg_space_counter}")
    print(f"avg_space_bit: {avg_space_bit}")
    print(f"num of time window: {avg_num_time_window}")

    print("the difference between two results by percentage elapsed time: ",
          (avg_elapsed_time_counter_per_item - avg_elapsed_time_bit_per_item) / avg_elapsed_time_bit_per_item)
    print("the difference between two results by percentage insertion: ",
          (avg_insertion_counter_per_item - avg_insertion_bit_per_item) / avg_insertion_bit_per_item)
    print("the difference between two results by percentage new window: ",
          (avg_new_window_counter - avg_new_window_bit) / avg_new_window_bit)
    print("the difference between two results by percentage query",
          (avg_elapsed_time_query_counter - avg_elapsed_time_query_bit) / avg_elapsed_time_query_bit)
    print("the difference between two results by percentage space: ",
          (avg_space_counter - avg_space_bit) / avg_space_bit)



with open("dynamic_window_running_time_Accuracy_v5.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["avg_elapsed_time_bit_per_item", "avg_elapsed_time_counter_per_item", "avg_new_window_bit",
                     "avg_new_window_counter", "avg_insertion_bit_per_item", "avg_insertion_counter_per_item",
                     "avg_space_bit", "avg_space_counter", "avg_elapsed_time_query_bit",
                     "avg_elapsed_time_query_counter",
                     "avg_num_time_window", "avg_num_accuracy_check"])


time_unit = "10000 nanosecond"
window_size_units = "1"
checking_interval = "100000 nanosecond"
use_nanosecond = True
repeat = 1
parameters = [(5, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 5000)]

for Tin, Tb in parameters:
    processing(time_unit, Tin, Tb, checking_interval, repeat)


########################################## end running time ##########################################
# repeat 1000
# avg elapsed_time_base out of 1000 executions: 0.13568266344070434
# avg elapsed_time_opt out of 1000 executions: 0.13563484144210816
