import colorsys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
from algorithm.dynamic_window import Accuracy_workload as dynamic_window_workload
from algorithm.fixed_window import Accuracy_workload as fixed_window_workload
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)



data = pd.read_csv('../../../data/name_gender/baby_names_1880_2020_US_predicted.csv')
print(data["sex"].unique())
date_column = "year"
date_time_format = False
time_window_str = "10"
monitored_groups = [{"sex": 'male'}, {"sex": 'female'}]
print(data[:5])
alpha = 0.5
threshold = 0.3
label_prediction = "predicted_gender"
label_ground_truth = "sex"



# ######################################## Fixed window ########################################
# print("fixed window:\n")
#
# DFMonitor, uf_list_DF, accuracy_list, counter_list_correct_DF, counter_list_incorrect_DF = fixed_window_workload.traverse_data_DFMonitor(
#     data, date_column,
#     time_window_str, date_time_format,
#     monitored_groups,
#     threshold,
#     alpha, label_prediction,
#     label_ground_truth)
#
#
#
#
#
# # save the result to a file
# male_time_decay = [x[0] for x in accuracy_list]
# female_time_decay = [x[1] for x in accuracy_list]
#
#
#
# filename = f"fixed_window_Accuracy_{time_window_str}.csv"
# with open(filename, "w", newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(["male_time_decay", "female_time_decay"])
#     for i in range(len(male_time_decay)):
#         writer.writerow([accuracy_list[i][0], accuracy_list[i][1]])


######################################## adaptive window ########################################
print("adaptive window:\n")
correctness_column = "correctness_column"
use_two_counters = True
time_unit = "86400 second"
checking_interval = "2592000 second"
use_nanosecond = False

data['correctness_column'] = data[label_prediction] == data[label_ground_truth]


T_b = 30
T_in = 5
start_date="1880"
end_date="2020"
DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect,\
        window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter,\
        total_time_new_window_bit, total_time_new_window_counter,\
        total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,\
        total_time_query_bit, total_time_query_counter, total_num_time_window \
    = dynamic_window_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, T_b, T_in,
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column,
                                                    use_two_counters, use_nanosecond, start_date, end_date)




# save the result to a file
male_time_decay = [x[0] for x in accuracy_list]
female_time_decay = [x[1] for x in accuracy_list]



filename = f"adaptive_window_Accuracy_{time_window_str}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "female_time_decay"])
    for i in range(len(male_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1]])



####################################### traditional ########################################
# print("traditional:\n")
#
# time_window_str = "10"
#
# uf_list_trad, accuracy_list_trad, counter_list_correct_trad, counter_list_incorrect_trad \
#     = fixed_window_workload.Accuracy_traditional(data, date_column,
#                                     time_window_str, date_time_format,
#                                     monitored_groups,
#                                     threshold,
#                                     label_prediction,
#                                     label_ground_truth)
#
#
# male_traditional = [x[0] for x in accuracy_list_trad]
# female_traditional = [x[1] for x in accuracy_list_trad]
#
#
# with open(f"traditional_Accuracy_{time_window_str}.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow([ "male_traditional",
#                      "female_traditional"])
#     for i in range(len(male_traditional)):
#         writer.writerow([male_traditional[i],  female_traditional[i]])
#







#
#
# plt.savefig("curves_smoothness_score.png", bbox_inches='tight')
# plt.show()
#
#

