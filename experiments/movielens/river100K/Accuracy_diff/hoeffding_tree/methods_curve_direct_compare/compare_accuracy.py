import colorsys
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
from algorithm.dynamic_window import Accuracy_workload as dynamic_window_workload
from algorithm.per_item import Accuracy_workload as fixed_window_workload
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


#  all time:
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
print(data[:5])
alpha = 0.995

threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "1 hour"
checking_interval = "7 day"
use_nanosecond = False


####################################### Dynamic window ########################################

Tb = 24*7
Tin = 24


DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
    window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
    total_time_new_window_bit, total_time_new_window_counter,\
total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,\
total_time_query_bit, total_time_query_counter, total_num_time_window \
    = dynamic_window_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, Tb, Tin,
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column,
                                                    use_two_counters, use_nanosecond)

final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]


# save the result to a file
male_time_decay = [x[0] for x in accuracy_list]
female_time_decay = [x[1] for x in accuracy_list]

filename = f"adaptive_window_{checking_interval}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "female_time_decay", "check_points"])
    for i in range(len(male_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1], check_points[i]])


######################################## Fixed window ########################################

window_size_units = "24*7"

DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
    window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
    total_time_new_window_bit, total_time_new_window_counter,\
total_time_insertion_bit, total_time_insertion_counter, total_num_accuracy_check,\
total_time_query_bit, total_time_query_counter, total_num_time_window \
    = fixed_window_workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, eval(window_size_units),
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column, use_nanosecond,
                                                    use_two_counters)


final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]
# print("final_accuracy", final_accuracy)

# save the result to a file
male_time_decay = [x[0] for x in accuracy_list]
female_time_decay = [x[1] for x in accuracy_list]

filename = f"fixed_window_{checking_interval}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "female_time_decay", "check_points"])
    for i in range(len(male_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1], check_points[i]])



######################################## traditional ########################################

#
#
# window_size = "1W"
# method_name = "hoeffding_classifier"
# df = pd.read_csv(f"../traditional/movielens_compare_Accuracy_{method_name}_traditional_gender_{window_size}.csv")
# print(df)
# value_col_name = "calculated_value"
#
#
#
#
#
# def plot_certain_time_window(df, value_col_name, window_size, axs):
#     df_female = df[df["gender"] == 'F']
#     # df_female = df_female[df_female[window_size].notna()]
#     df_female = df_female[["datetime", value_col_name]]
#
#     df_male = df[df["gender"] == 'M']
#     # df_male = df_male[df_male[window_size].notna()]
#     df_male = df_male[["datetime", value_col_name]]
#
#     print("df_male: \n", df_male)
#     print("\ndf_female: \n", df_female)
#
#     x_list = np.arange(0, len(df_male))
#
#     from datetime import datetime
#     datetime = df_male['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y')).tolist()
#
#     window_size_list = df.columns.tolist()[2:]
#     print("window_size_list", window_size_list)
#
#
#     axs.grid(True)
#     female_lst = df_female[value_col_name].dropna().tolist()
#     male_lst = df_male[value_col_name].dropna().tolist()
#     print(male_lst, female_lst)
#     df = pd.DataFrame({'datetime': datetime, 'female': female_lst, 'male': male_lst})
#     df.to_csv("traditional.csv", index=False)
#
#     # axs.plot(np.arange(len(male_lst)), male_lst, linewidth=2.5, markersize=3.5,
#     #          label="male", linestyle='-', marker='o', color="blue")
#     # axs.plot(np.arange(len(female_lst)), female_lst, linewidth=2.5, markersize=3.5,
#     #             label='female', linestyle='-', marker='o', color="orange")
#     # axs.legend(loc='lower right',
#     #        bbox_to_anchor=(1.0, 0.07),  # Adjust this value (lower the second number)
#     #        fontsize=12, ncol=1, labelspacing=0.2, handletextpad=0.5,
#     #        markerscale=1, handlelength=1, columnspacing=0.6,
#     #        borderpad=0.2, frameon=True)
#     #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
