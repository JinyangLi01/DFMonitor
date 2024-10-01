import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype

from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20


# # activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


#  all time:
method_name = "hoeffding_classifier"
data = pd.read_csv('../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
print(data[:5])
alpha = 0.9997
threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "30 min"
window_size_units = 4
checking_interval_units = 2

DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, fixed_window_times \
    = workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha, time_unit, window_size_units, checking_interval_units, label_prediction,
                                                label_ground_truth, correctness_column, use_two_counters)
# already check the correctness of the accuracy finally got
final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]
print("final_accuracy", final_accuracy)

# save the result to a file
male_time_decay = [x[0] for x in accuracy_list]
female_time_decay = [x[1] for x in accuracy_list]

filename = f"movielens_compare_Accuracy_{method_name}_gender_per_item_alpha_{str(int(alpha*10000))}_time_unit_{time_unit}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "female_time_decay"])
    for i in range(len(male_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1]])



# get the first time stamp in the data
fixed_window_times = data.iloc[0][date_column]
last_timestamp = data.iloc[len(data)-1][date_column]

# Extract the time unit and value
time_value, time_unit_str = time_unit.split(" ")
time_value = int(time_value)

# Compute the total window size based on the number of time units
total_window_size = pd.Timedelta(minutes=time_value * window_size_units)  # 4 * 30 min = 120 min (2 hours)


# List to store reset times
reset_times = []

# Loop to generate all reset times from fixed_window_times to last_timestamp
current_time = fixed_window_times

while current_time <= last_timestamp:
    reset_times.append(current_time)
    current_time += total_window_size  # Move forward by the total window size (2 hours)

# Now, reset_times contains all the reset times for the 2-hour windows
print(reset_times)


filename = f"movielens_fixed_window_resets_time_unit_{time_unit}.csv"

with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["fixed_window_reset_times"])
    for i in range(len(fixed_window_times)):
        writer.writerow([fixed_window_times[i]])


# ################################################## draw the plot #####################################################
#
#import csv

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

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)



df = pd.read_csv(f"movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_alpha_{str(int(alpha*10000))}_time_unit_{time_unit}.csv")
print(df)

x_list = np.arange(0, len(df))
male_time_decay = df["male_time_decay"].tolist()
female_time_decay = df["female_time_decay"].tolist()


# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "orange"]

num_lines = len(x_list)
pair_colors = cmaps.set1.colors

fig, ax = plt.subplots(figsize=(6, 3.5))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)


ax.plot(x_list, male_time_decay, linewidth=1, markersize=2, label='Male time decay', linestyle='-', marker='o', color=pair_colors[1])
ax.plot(x_list, female_time_decay, linewidth=1, markersize=2, label='Female time decay', linestyle='-', marker='o', color=pair_colors[4])


plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
plt.xlabel('time stamps\n from 01/01/2013, 1m interval',
           fontsize=20, labelpad=-2).set_position((0.47, 0.1))
plt.ylabel('Accuracy', fontsize=20, labelpad=-1)
plt.ylim(0.6, 1.0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.3), fontsize=15,
               ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
               columnspacing=0.2, borderpad=0.2, frameon=True)
plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_per_item_alpha_{str(int(alpha*10000))}_time_unit_{time_unit}.png", bbox_inches='tight')
plt.show()