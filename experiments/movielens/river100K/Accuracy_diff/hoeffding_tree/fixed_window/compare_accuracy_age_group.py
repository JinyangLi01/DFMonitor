import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from algorithm.fixed_window import Accuracy_workload as workload
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

data = pd.read_csv('../../../result_hoeffding_classifier.csv')
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
date_time_format = True
time_window_str = "1 month"
monitored_groups = [{"age_group_4": '7-25'}, {"age_group_4": '26-31'}, {'age_group_4': '32-40'}, {'age_group_4': '41-73'}]
print(data[:5])
alpha = 0.5
threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"

DFMonitor_baseline, uf_list_baseline, accuracy_list_baseline, counter_list_correct_baseline, counter_list_incorrect_baseline \
    = workload.traverse_data_DFMonitor_baseline(data, date_column,
                                                time_window_str, date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha, label_prediction,
                                    label_ground_truth, correctness_column)

# use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
DFMonitor, uf_list_DF, accuracy_list_DF, counter_list_correct_DF, counter_list_incorrect_DF \
    = workload.traverse_data_DFMonitor(data, date_column,
                                       time_window_str, date_time_format,
                                       monitored_groups,
                                       threshold,
                                       alpha, label_prediction,
                                    label_ground_truth, correctness_column)

uf_list_trad, accuracy_list_trad, counter_list_correct_trad, counter_list_incorrect_trad \
    = workload.Accuracy_traditional(data, date_column,
                                    time_window_str, date_time_format,
                                    monitored_groups,
                                    threshold, label_prediction,
                                    label_ground_truth, correctness_column)

print(len(uf_list_trad), len(uf_list_trad))



age_7_25_time_decay = [x[0] for x in accuracy_list_DF]
age_7_25_traditional = [x[0] for x in accuracy_list_trad]
age_26_31_time_decay = [x[1] for x in accuracy_list_DF]
age_26_31_traditional = [x[1] for x in accuracy_list_trad]
age_32_40_time_decay = [x[2] for x in accuracy_list_DF]
age_32_40_traditional = [x[2] for x in accuracy_list_trad]
age_41_73_time_decay = [x[3] for x in accuracy_list_DF]
age_41_73_traditional = [x[3] for x in accuracy_list_trad]


# save to file

with open("movielens_compare_Accuracy_hoeffding_classifier_age_group.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["age_7_25_time_decay", "age_7_25_traditional", "age_26_31_time_decay",
                     "age_26_31_traditional", "age_32_40_time_decay", "age_32_40_traditional",
                     "age_41_73_time_decay", "age_41_73_traditional"])
    for i in range(len(age_32_40_traditional)):
        writer.writerow([age_7_25_time_decay[i], age_7_25_traditional[i],
                         age_26_31_time_decay[i], age_26_31_traditional[i],
                         age_32_40_time_decay[i], age_32_40_traditional[i],
                         age_41_73_time_decay[i], age_41_73_traditional[i]])



if len(uf_list_baseline) != len(uf_list_DF):
    print("uf_list_baseline and uf_list_DF have different length")

for i in range(0, len(accuracy_list_DF)):
    if accuracy_list_DF[i] != accuracy_list_baseline[i]:
        print("cr_list_baseline and cr_list_DF have different length")


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
