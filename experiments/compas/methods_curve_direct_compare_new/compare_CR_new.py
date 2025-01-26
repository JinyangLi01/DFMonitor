import colorsys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
from algorithm.dynamic_window import CR_workload as dynamic_window_workload
from algorithm.fixed_window import CR_workload as fixed_window_workload
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



data = pd.read_csv('../../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
print(data["race"].unique())
# get distribution of compas_screening_date
data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
# data['compas_screening_date'].hist()
date_column = "compas_screening_date"
time_window_str = "1 month"
monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Asian"}, {"race": "Hispanic"},
                    {"race": "Other"}, {"race": "Native American"}]
date_time_format = True
alpha = 0.5
threshold = 0.3





label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "86400 second"
checking_interval = "2592000 second"
use_nanosecond = True



######################################## Fixed window ########################################



DFMonitor_baseline, uf_list_baseline, counter_list_baseline, cr_list_baseline \
    = fixed_window_workload.traverse_data_DFMonitor_baseline(data, date_column,
                                                time_window_str, date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha)
# use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
DFMonitor, uf_list_DF, cr_list_DF, counter_list_DF = fixed_window_workload.traverse_data_DFMonitor(data, date_column,
                                                                                      time_window_str,
                                                                                      date_time_format,
                                                                                      monitored_groups,
                                                                                      threshold,
                                                                                      alpha)






filename = f"fixed_window_CR_{time_window_str}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["white_time_decay", "black_time_decay", "asian_time_decay","hispanic_time_decay"])
    for i in range(len(cr_list_DF)):
        writer.writerow([cr_list_DF[i][0], cr_list_DF[i][1], cr_list_DF[i][2], cr_list_DF[i][3]])




######################################## adaptive window ########################################
T_b = 50
T_in = 20

DFMonitor_bit, DFMonitor_counter, uf_list, cr_list, counter_list_correct, counter_list_incorrect, \
    window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
    time_new_window_bit, time_new_window_counter, \
    time_insertion_bit, time_insertion_counter, num_cr_check, \
    time_query_bit, time_query_counter, num_time_window \
    = dynamic_window_workload.traverse_data_DFMonitor_and_baseline_CR(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, T_b, T_in,
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column,
                                                    use_two_counters, use_nanosecond)




# save the result to a file
white_time_decay = [x[0] for x in cr_list]
black_time_decay = [x[1] for x in cr_list]
asian_time_decay = [x[2] for x in cr_list]
hispanic_time_decay = [x[3] for x in cr_list]


filename = f"adaptive_window_CR_{time_window_str}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["white_time_decay", "black_time_decay", "asian_time_decay",
                     "hispanic_time_decay"])
    for i in range(len(white_time_decay)):
        writer.writerow([cr_list[i][0], cr_list[i][1], cr_list[i][2], cr_list[i][3]])



######################################## traditional ########################################

#
# time_window_str = "1 month"
#
# counter_list_trad, cr_list_trad, uf_list_trad = fixed_window_workload.CR_traditional(data, date_column,
#                                                                         time_window_str, date_time_format,
#                                                                         monitored_groups,
#                                                                         threshold)
#
#
#
# white_traditional = [x[0] for x in cr_list_trad]
#
# black_traditional = [x[1] for x in cr_list_trad]
#
# asian_traditional = [x[2] for x in cr_list_trad]
#
# hispanic_traditional = [x[3] for x in cr_list_trad]
#
#
#
# with open(f"traditional_CR_time_window_{time_window_str}.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow([ "white_traditional",
#                      "black_traditional", "asian_traditional", "hispanic_traditional"])
#     for i in range(len(white_time_decay)):
#         writer.writerow([white_traditional[i],  black_traditional[i],
#                             asian_traditional[i], hispanic_traditional[i]])
#
#
#
#
#


#
#
#
# plt.savefig("curves_smoothness_score.png", bbox_inches='tight')
# plt.show()
#
#

