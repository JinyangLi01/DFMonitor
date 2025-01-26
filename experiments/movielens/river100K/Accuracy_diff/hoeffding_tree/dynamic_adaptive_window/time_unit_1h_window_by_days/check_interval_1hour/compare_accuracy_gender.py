import colorsys
import csv
import math
from tabnanny import check

import pandas as pd
import matplotlib.pyplot as plt
from joblib.parallel import method
from numpy import dtype

from algorithm.dynamic_window import Accuracy_workload as workload_dynamic
from algorithm.fixed_window import Accuracy_workload as workload_fixed
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(3.5, 2))
plt.rcParams['font.size'] = 20


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


#  all time:
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True


monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
print(data[:5])
alpha = 0.996
threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "1 hour"
T_b = 4
T_in = 2
checking_interval = "1 hour"
# print("====================== fixed window workload ==================")
#
# DFMonitor_bit_fixed, DFMonitor_counter_fixed, uf_list_fixed, accuracy_list_fixed, counter_list_correct_fixed, counter_list_incorrect_fixed \
#     = workload_fixed.traverse_data_DFMonitor_and_baseline(data, date_column,
#                                                 time_window_str, date_time_format,
#                                                 monitored_groups,
#                                                 threshold,
#                                                 alpha, label_prediction,
#                                                 label_ground_truth, correctness_column)


print("====================== dynamic workload ==================")
print("T_b = {}, T_in = {}".format(T_b, T_in))
(DFMonitor_bit_dynamic, DFMonitor_counter_dynamic, uf_list_dynamic, accuracy_list_dynamic,
 counter_list_correct_dynamic, counter_list_incorrect_dynamic, window_reset_times, check_points) \
= workload_dynamic.traverse_data_DFMonitor_and_baseline(data, date_column, date_time_format,
                                                monitored_groups,  threshold, alpha,
                                                time_unit, checking_interval, label_prediction,
                                                label_ground_truth, correctness_column, T_b, T_in, use_two_counters)


# print("accuracy_list_dynamic", accuracy_list_dynamic)
# print("counter_list_incorrect_dynamic", counter_list_incorrect_dynamic)
# print("counter_list_correct_dynamic", counter_list_correct_dynamic)
# print("uf_list_dynamic", uf_list_dynamic)
print(len(uf_list_dynamic))

# accuracy_list_dynamic = DFMonitor_counter_dynamic.get_accuracy_list()
print(accuracy_list_dynamic[:5])

# save the result to a file
male_time_decay_dynamic = [x[0] for x in accuracy_list_dynamic]
# male_time_decay_fixed = [x[0] for x in accuracy_list_fixed]
female_time_decay_dynamic = [x[1] for x in accuracy_list_dynamic]
# female_time_decay_fixed = [x[1] for x in accuracy_list_fixed]

filename = f"movielens_compare_Accuracy_{method_name}_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay_dynamic", "female_time_decay_dynamic", "check_points"])
    for i in range(len(female_time_decay_dynamic)):
        writer.writerow([accuracy_list_dynamic[i][0], accuracy_list_dynamic[i][1], check_points[i]])



filename = f"movielens_dynamic_window_reset_time_Tb_{T_b}_Tin_{T_in}_time_unit_{time_unit}.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["dynamic_window_reset_time"])
    for i in range(len(window_reset_times)):
        writer.writerow([window_reset_times[i]])


# ################################################## draw the plot #####################################################
import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys
import colormaps as cmaps

sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(3.5, 2))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


df = pd.read_csv(f"movielens_compare_Accuracy_hoeffding_classifier_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.csv")
print(df)

x_list = np.arange(0, len(df))
male_time_decay = df["male_time_decay_dynamic"].tolist()
female_time_decay = df["female_time_decay_dynamic"].tolist()


# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange"]

fig, ax = plt.subplots(figsize=(3.5, 2))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)


ax.plot(x_list, male_time_decay, linewidth=1, markersize=1, label='Male', linestyle='-', marker='o', color=pair_colors[0])
ax.plot(x_list, female_time_decay, linewidth=1, markersize=1, label='Female', linestyle='-', marker='o', color=pair_colors[1])


plt.xticks([], [], rotation=0, fontsize=20)
plt.ylabel('Accuracy', fontsize=14, labelpad=-1)
plt.grid(True, axis='y')
plt.yticks(fontsize=12)
plt.ylim(0.4, 1.0)
plt.yticks([0.4, 0.6, 0.8, 1.0], ["0.4", "0.6", "0.8", "1.0"], fontsize=12)
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.3), fontsize=12,
               ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
               columnspacing=0.2, borderpad=0.2, frameon=True)
plt.savefig(f"Acc_hoeffding_time_decay_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.png", bbox_inches='tight')
plt.show()