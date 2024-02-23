import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import Accuracy_workload as workload
import seaborn as sns
from matplotlib import rc
import colorsys
from pympler import asizeof


# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

data = pd.read_csv('../../../data/name_gender/baby_names_1880_2020_US_predicted.csv')
print(data["sex"].unique())
date_column = "year"
date_time_format = False
time_window_str = "10"
monitored_groups = [{"sex": 'male'}, {"sex": 'female'}]

window_size_str_list = ["2 year", "5 year", "10 year", "20 year", "40 year", "60 year", "80 year", "100 year"]
window_size_str_list_brev = ["2", "5", "10", "20", "40", "60", "80", "100"]
threshold = 0.3
alpha = 0.5
label_prediction = "predicted_gender"
label_ground_truth = "sex"


########################################## start running time ##########################################


def one_run(monitored_group, num_repeat=10):

    total_elapsed_time_opt = 0
    total_elapsed_time_base = 0
    total_elasped_time_new_window_opt = 0
    total_elasped_time_new_window_base = 0
    total_elapsed_time_insertion_opt = 0
    total_elapsed_time_insertion_base = 0
    num_items_after_first_window_base = 0
    num_items_after_first_window_opt = 0
    total_time_new_window_base = 0
    total_time_new_window_opt = 0
    num_new_windows_base = 0
    num_new_windows_opt = 0

    for i in range(num_repeat):
        (DFMonitor_baseline, elapsed_time_base, total_time_new_window_base,
         num_items_after_first_window_base, num_new_windows_base) \
            = workload.traverse_data_DF_baseline_only(data,
                                                      date_column,
                                                      time_window_str,
                                                      date_time_format,
                                                      monitored_group,
                                                      threshold,
                                                      alpha, label_prediction, label_ground_truth)

        DFMonitor, elapsed_time_opt, total_time_new_window_opt, num_items_after_first_window_opt, num_new_windows_opt \
            = workload.traverse_data_DFMonitor_only(data,
                                                    date_column,
                                                    time_window_str,
                                                    date_time_format,
                                                    monitored_group,
                                                    threshold,
                                                    alpha, label_prediction, label_ground_truth)

        total_elapsed_time_opt += elapsed_time_opt
        total_elapsed_time_base += elapsed_time_base
    print("monitored groups: ", monitored_groups)
    # print("avg elapsed_time_base out of {} executions: {}".format(num_repeat, total_elapsed_time_base / num_repeat))
    # print("avg elapsed_time_opt out of {} executions: {}".format(num_repeat, total_elapsed_time_opt / num_repeat))
    # print("num_items_after_first_window_base: ", num_items_after_first_window_base)
    # print("avg time of an operation base: ", total_elapsed_time_base / (num_items_after_first_window_base * num_repeat))
    # print("avg time of an operation opt: ", total_elapsed_time_opt / (num_items_after_first_window_base * num_repeat))

    avg_newwindow_base = total_time_new_window_base / (num_repeat * num_new_windows_base)
    avg_newwindow_opt = total_time_new_window_opt / (num_new_windows_opt * num_repeat)
    avg_insertion_base = (total_elapsed_time_base - total_time_new_window_base) / \
                            (num_items_after_first_window_base * num_repeat - (num_repeat * num_new_windows_base))
    avg_insertion_opt = (total_elapsed_time_opt - total_time_new_window_opt) / \
                            (num_items_after_first_window_opt * num_repeat - (num_new_windows_opt * num_repeat))

    return avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt





monitored_groups_set = [
                        [{"sex": "male"}], [{"sex": "female"}],
[{"sex": "male"}], [{"sex": "female"}],[{"sex": "female"}],

                        [{"sex": "female"}, {"sex": "male"}],
    [{"sex": "female"}, {"sex": "male"}],
    [{"sex": "female"}, {"sex": "male"}],
    [{"sex": "female"}, {"sex": "male"}],
    [{"sex": "female"}, {"sex": "male"}],
                        ]


avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt = 0, 0, 0, 0
result_insertion_base = []
result_insertion_opt = []
result_newwindow_base = []
result_newwindow_opt = []
repeat = 10
for i, group in enumerate(monitored_groups_set):
    print("group: ", group, "i: ", i)
    a, b, c, d = one_run(group, repeat)
    avg_insertion_base += a
    avg_insertion_opt += b
    avg_newwindow_base += c
    avg_newwindow_opt += d
    if (i+1) % 5 == 0:
        result_insertion_base.append(avg_insertion_base / 5)
        result_insertion_opt.append(avg_insertion_opt / 5)
        result_newwindow_base.append(avg_newwindow_base / 5)
        result_newwindow_opt.append(avg_newwindow_opt / 5)
        avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt = 0, 0, 0, 0
with open("running_time_Accuracy.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["avg_insertion_base", "avg_insertion_opt", "avg_newwindow_base", "avg_newwindow_opt"])
    writer.writerow([result_insertion_base, result_insertion_opt, result_newwindow_base, result_newwindow_opt])
    # the difference between two results by percentage
    writer.writerow([(result_insertion_base[i] - result_insertion_opt[i]) / result_insertion_base[i] for i in range(len(result_insertion_base))])
    writer.writerow([(result_newwindow_base[i] - result_newwindow_opt[i]) / result_newwindow_base[i] for i in range(len(result_newwindow_base))])

print("avg_insertion_base: ", result_insertion_base)
print("avg_insertion_opt: ", result_insertion_opt)
print("avg_newwindow_base: ", result_newwindow_base)
print("avg_newwindow_opt: ", result_newwindow_opt)
print("the difference between two results by percentage insertion: ",
      [(result_insertion_base[i] - result_insertion_opt[i]) / result_insertion_base[i] for i in range(len(result_insertion_base))])
print("the difference between two results by percentage new window: ",
      [(result_newwindow_base[i] - result_newwindow_opt[i]) / result_newwindow_base[i] for i in range(len(result_newwindow_base))])



########################################## end running time ##########################################
# repeat 1000
# avg elapsed_time_base out of 1000 executions: 0.13568266344070434
# avg elapsed_time_opt out of 1000 executions: 0.13563484144210816
