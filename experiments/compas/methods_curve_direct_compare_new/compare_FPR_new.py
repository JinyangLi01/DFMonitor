import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
from algorithm.dynamic_window import FPR_workload_1 as dynamic_window_workload
from algorithm.fixed_window import FPR_workload as fixed_window_workload


sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

data = pd.read_csv('../../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
print(data["race"].unique())

# Set up parameters
date_column = "compas_screening_date"
data[date_column] = pd.to_datetime(data[date_column])
time_window_str = "1 month"
monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Asian"}, {"race": "Hispanic"},
                    {"race": "Other"}, {"race": "Native American"}]
threshold = 0.3
alpha = 0.95
date_time_format = True
time_unit = "86400 second"

checking_interval = "2592000 second"
use_nanosecond = False

print(len(data))
print("dates:", data[date_column].min(), data[date_column].max())

######################################## Fixed Window ########################################

# DFMonitor, uf_list_DF, fpr_list_DF, counter_list_TN_DF, counter_list_FP_DF = fixed_window_workload.traverse_data_DFMonitor(
#     data, date_column, time_window_str, monitored_groups, threshold, alpha)
#
# filename = f"fixed_window_FPR_{time_window_str}.csv"
# with open(filename, "w", newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(["white_time_decay", "black_time_decay", "asian_time_decay", "hispanic_time_decay"])
#     for i in range(len(fpr_list_DF)):
#         writer.writerow([fpr_list_DF[i][0], fpr_list_DF[i][1], fpr_list_DF[i][2], fpr_list_DF[i][3]])

######################################## adaptive Window ########################################
T_b = 5
T_in = 3
mechanism="fixed"
start_date="2013-01-01"
end_date="2014-12-31"
label_prediction="predicted"
label_ground_truth = "ground_truth"
use_two_counters = False
correctness_column = ""
DFMonitor_bit, DFMonitor_counter, uf_list, fpr_list, counter_list_fp, counter_list_fn,\
        counter_list_tp, counter_list_tn,\
        window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter,\
        total_time_new_window_bit, total_time_new_window_counter,\
        total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,\
        total_time_query_bit, total_time_query_counter, total_num_time_window = (
        dynamic_window_workload.traverse_data_DFMonitor_and_baseline(
        data, date_column, date_time_format, monitored_groups, threshold, alpha, time_unit, T_b, T_in,
        checking_interval, label_prediction, label_ground_truth, correctness_column, use_two_counters, use_nanosecond,
            start_date, end_date))

print(check_points)

filename = f"adaptive_window_FPR_{time_window_str}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["white_time_decay", "black_time_decay", "asian_time_decay", "hispanic_time_decay"])
    for i in range(len(fpr_list)):
        writer.writerow([fpr_list[i][0], fpr_list[i][1], fpr_list[i][2], fpr_list[i][3]])

# ######################################## Traditional ########################################
#
# uf_list_trad, fpr_list_trad, counter_list_TN_trad, counter_list_FP_trad = fixed_window_workload.FPR_traditional(data, date_column,
#                                                                                                    time_window_str,
#                                                                                                    monitored_groups,
#                                                                                                    threshold)
#
# filename = f"traditional_FPR_{time_window_str}.csv"
# with open(filename, "w", newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(["white_traditional", "black_traditional", "asian_traditional", "hispanic_traditional"])
#     for i in range(len(fpr_list_trad)):
#         writer.writerow([fpr_list_trad[i][0], fpr_list_trad[i][1], fpr_list_trad[i][2], fpr_list_trad[i][3]])
