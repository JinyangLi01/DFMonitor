import colorsys
import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import CR_workload as workload
import seaborn as sns
from matplotlib import rc

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

DFMonitor_baseline, uf_list_baseline, counter_list_baseline, cr_list_baseline \
    = workload.traverse_data_DFMonitor_baseline(data, date_column,
                                                time_window_str, date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha)

# use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
DFMonitor, uf_list_DF, cr_list_DF, counter_list_DF = workload.traverse_data_DFMonitor(data, date_column,
                                                                                      time_window_str,
                                                                                      date_time_format,
                                                                                      monitored_groups,
                                                                                      threshold,
                                                                                      alpha)

counter_list_trad, cr_list_trad, uf_list_trad = workload.CR_traditional(data, date_column,
                                                                        time_window_str, date_time_format,
                                                                        monitored_groups,
                                                                        threshold)





print(len(cr_list_DF), len(cr_list_trad))

if len(uf_list_baseline) != len(uf_list_DF):
    print("uf_list_baseline and uf_list_DF have different length")

for i in range(0, len(cr_list_DF)):
    if cr_list_baseline[i] != cr_list_DF[i]:
        print("cr_list_baseline and cr_list_DF have different length")

# draw chart of the first and second value in all lists in fpr_list and fpr_list1
# 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
# monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Asian"}, {"race": "Hispanic"},
#                     {"race": "Other"}, {"race": "Native American"}]
white_time_decay = [x[0] for x in cr_list_DF]
white_traditional = [x[0] for x in cr_list_trad]
black_time_decay = [x[1] for x in cr_list_DF]
black_traditional = [x[1] for x in cr_list_trad]
asian_time_decay = [x[2] for x in cr_list_DF]
asian_traditional = [x[2] for x in cr_list_trad]
hispanic_time_decay = [x[3] for x in cr_list_DF]
hispanic_traditional = [x[3] for x in cr_list_trad]



with open("case_study_CR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["white_time_decay", "white_traditional", "black_time_decay",
                     "black_traditional", "hispanic_time_decay", "hispanic_traditional"])
    for i in range(len(white_time_decay)):
        writer.writerow([white_time_decay[i], white_traditional[i], black_time_decay[i], black_traditional[i],
                            hispanic_time_decay[i], hispanic_traditional[i]])






#
#
# x_list = np.arange(0, len(cr_list_DF))
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
# ax.plot(x_list, hispanic_time_decay, linewidth=3, markersize=6, label='Hispanic time decay', linestyle='-', marker='o',
#         color=pair_colors[4])
# ax.plot(x_list, hispanic_traditional, linewidth=3, markersize=6, label='Hispanic traditional', linestyle='--',
#         marker='s',
#         color=pair_colors[5])
#
# # ax.plot(x_list, asian_time_decay, linewidth=3, markersize=6, label='Hispanic time decay', linestyle='-', marker='o',
# #         color='red')
# # ax.plot(x_list, asian_traditional, linewidth=3, markersize=6, label='Hispanic traditional', linestyle='--', marker='s',
# #         color='orange')
#
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yticks([0.0, 0.2, 0.4, 0.6], fontsize=20)
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# plt.legend(loc='lower left', bbox_to_anchor=(-0.142, 1), fontsize=15,
#            ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
#            columnspacing=0.2, borderpad=0.2, frameon=True)
# plt.savefig("CR_timedecay_traditional.png", bbox_inches='tight')
# plt.show()