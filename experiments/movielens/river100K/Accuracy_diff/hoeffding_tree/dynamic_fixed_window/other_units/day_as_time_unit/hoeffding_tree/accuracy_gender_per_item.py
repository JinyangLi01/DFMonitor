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
alpha = 0.9
threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "day"




DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect \
    = workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha, time_unit, label_prediction,
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

filename = "movielens_compare_Accuracy_" + method_name + "_gender_per_item_50.csv"
with open(filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "female_time_decay"])
    for i in range(len(male_time_decay)):
        writer.writerow([accuracy_list[i][0], accuracy_list[i][1]])

# ################################################## draw the plot #####################################################
#
# white_time_decay = [x[0] for x in cr_list_DF]
# white_traditional = [x[0] for x in cr_list_trad]
# black_time_decay = [x[1] for x in cr_list_DF]
# black_traditional = [x[1] for x in cr_list_trad]
#
# x_list = np.arange(1880, 2021)
# # pair_colors = ['blue', '#27bcf6', 'green', 'lightgreen', 'chocolate', 'gold']
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
# # Get the "Paired" color palette
# paired_palette = sns.color_palette("Paired")
# # Rearrange the colors within each pair
# # pair_colors = [paired_palette[i + 1] if i % 2 == 0 else paired_palette[i - 1] for i in range(len(paired_palette))]
# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00',
#                'indianred', '#fe01b1']
#
# # Plot the first curve (y1_values)
# ax.plot(x_list, black_time_decay, linewidth=3, markersize=6, label='Black time decay', linestyle='-', marker='o',
#         color=pair_colors[0])
#
# # Plot the second curve (y2_values)
# ax.plot(x_list, black_traditional, linewidth=3, markersize=6, label='Black traditional', linestyle='--', marker='s',
#         color=pair_colors[1])
#
# ax.plot(x_list, white_time_decay, linewidth=3, markersize=6, label='White time decay', linestyle='-', marker='o',
#         color=pair_colors[2])
# ax.plot(x_list, white_traditional, linewidth=3, markersize=6, label='White traditional', linestyle='--', marker='s',
#         color=pair_colors[3])
#
#
#
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yticks([0.0, 0.2, 0.4, 0.6], fontsize=20)
# plt.xlabel('year',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# plt.legend(loc='lower left', bbox_to_anchor=(-0.142, 1), fontsize=15,
#            ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
#            columnspacing=0.2, borderpad=0.2, frameon=True)
# plt.savefig("CR_baby_names.png", bbox_inches='tight')
# plt.show()
