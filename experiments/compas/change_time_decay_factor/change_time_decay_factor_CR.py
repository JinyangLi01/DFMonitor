import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from algorithm import CR_workload as workload
import seaborn as sns
from matplotlib import rc
from algorithm import config
import colorsys

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
monitored_groups = ([{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Asian"}, {"race": "Hispanic"}])
# , {"race": "Other"}, {"race": "Native American"}]

alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
threshold = 0.5
date_time_format = True
black_time_decay_dif_alpha = []
white_time_decay_dif_alpha = []
hispanic_time_decay_dif_alpha = []

for alpha in alpha_list[:-1]:
    print("\nalpha = {}".format(alpha))
    # use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
    DFMonitor, uf_list_DF, cr_list_DF, counter_list_DF = workload.traverse_data_DFMonitor(data, date_column,
                                                                                          time_window_str,
                                                                                          date_time_format,
                                                                                          monitored_groups,
                                                                                          threshold,
                                                                                          alpha)

    # draw chart of the first and second value in all lists in fpr_list and fpr_list1
    # 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
    black_time_decay = [x[1] for x in cr_list_DF]
    white_time_decay = [x[0] for x in cr_list_DF]
    asian_time_decay = [x[2] for x in cr_list_DF]
    hispanic_time_decay = [x[3] for x in cr_list_DF]

    black_time_decay_dif_alpha.append(black_time_decay)
    white_time_decay_dif_alpha.append(white_time_decay)
    hispanic_time_decay_dif_alpha.append(hispanic_time_decay)

## alpha = 1
DFMonitor1, uf_list_DF1, cr_list_DF1, counter_list_DF1 = workload.traverse_data_DFMonitor(data, date_column,
                                                                                          time_window_str,
                                                                                          date_time_format,
                                                                                          monitored_groups,
                                                                                          threshold,
                                                                                          1)

black_time_decay_dif_alpha.append([x[1] for x in cr_list_DF1])
white_time_decay_dif_alpha.append([x[0] for x in cr_list_DF1])
hispanic_time_decay_dif_alpha.append([x[3] for x in cr_list_DF1])

# draw chart of the first and second value in all lists in fpr_list and fpr_list1
# 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
# black_time_decay_alpha1 = [x[1] for x in cr_list_DF1]
# white_time_decay_alpha1 = [x[0] for x in cr_list_DF1]
# asian_time_decay_alpha1 = [x[2] for x in cr_list_DF1]
# hispanic_time_decay_alpha1 = [x[3] for x in cr_list_DF1]

counter_list_trad, cr_list_trad, uf_list_trad = workload.CR_traditional(data, date_column,
                                                                        time_window_str,
                                                                        date_time_format,
                                                                        monitored_groups,
                                                                        threshold)

# draw chart of the first and second value in all lists in fpr_list and fpr_list1
# 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
black_traditional = [x[1] for x in cr_list_trad]
white_traditional = [x[0] for x in cr_list_trad]
# asian_traditional = [x[2] for x in cr_list_trad]
hispanic_traditional = [x[3] for x in cr_list_trad]

with open("CR_time_decay_factor.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["alpha", "black_time_decay", "white_time_decay",
                     "hispanic_time_decay"])
    for i in range(len(alpha_list)):
        writer.writerow([alpha_list[i], black_time_decay_dif_alpha[i],
                         white_time_decay_dif_alpha[i],
                         hispanic_time_decay_dif_alpha[i]])
    writer.writerow(['traditional', black_traditional, white_traditional, hispanic_traditional])


############################################## black ####################################################
# x_list = np.arange(0, len(black_traditional))
# # pair_colors = ['blue', '#27bcf6', 'green', 'lightgreen', 'chocolate', 'gold']
# # Define the number of shades and the colormap
# num_shades = len(alpha_list)
#
# cmap = plt.get_cmap('Blues')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# cmap_modified = matplotlib.colors.LinearSegmentedColormap.from_list(
#     'BluesModified', cmap(np.linspace(0.2, 1.1, num_shades)))
# curve_colors = [cmap_modified(i / (num_shades - 1)) for i in range(num_shades)]
# #
# color = matplotlib.colors.ColorConverter.to_rgb("navy")
# curve_colors = [scale_lightness(color, scale) for scale in [1.1, 1.6, 2.7, 3.3, 3.7]]
# curve_colors = curve_colors[::-1]
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# for i in range(len(alpha_list)):
#     ax.plot(x_list, black_time_decay_dif_alpha[i], linewidth=3, markersize=6,
#             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
# # alpha = 1
# ax.plot(x_list, black_time_decay_alpha1, linewidth=3, markersize=6,
#         label='{}'.format(1), linestyle='-', marker='o', color="black")
#
# # traditional
# ax.plot(x_list, black_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="cyan")
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yscale('log')
# # plt.yscale('symlog')
# # plt.ylim(0.01, 0.6)
# # plt.yticks([0.0, 0.2])
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.25, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("CR_time_decay_factor_black.png", bbox_inches='tight')
# plt.show()
#
# ############################################## white ####################################################
# cmap = plt.get_cmap('Greens')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# cmap_modified = matplotlib.colors.LinearSegmentedColormap.from_list(
#     'GreensModified', cmap(np.linspace(0.5, 1.1, num_shades)))
# curve_colors = [cmap_modified(i / (num_shades - 1)) for i in range(num_shades)]
#
# # color = matplotlib.colors.ColorConverter.to_rgb("#4f9153")
# # curve_colors = [scale_lightness(color, scale) for scale in [0.8, 1.1, 1.5, 1.7, 2.0]]
# # curve_colors = curve_colors[::-1]
#
# # color = matplotlib.colors.ColorConverter.to_rgb("#77ab56")
# # curve_colors = [scale_lightness(color, scale) for scale in [0.6, 0.85, 1.2, 1.4, 1.7]]
# # curve_colors = curve_colors[::-1]
#
# color = matplotlib.colors.ColorConverter.to_rgb("#287c37")
# curve_colors = [scale_lightness(color, scale) for scale in [0.75, 1.25, 1.7, 2.2, 2.7]]
# curve_colors = curve_colors[::-1]
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# for i in range(len(alpha_list)):
#     ax.plot(x_list, white_time_decay_dif_alpha[i], linewidth=3, markersize=6,
#             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
# # alpha = 1
# ax.plot(x_list, white_time_decay_alpha1, linewidth=3, markersize=6,
#         label='{}'.format(1), linestyle='-', marker='o', color="black")
#
# # traditional
# ax.plot(x_list, white_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="#cccc00")
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# # plt.yticks([0.2, 0.3, 0.4], fontsize=20)
# # plt.yscale('symlog', linthresh=0.39)
# plt.yscale('log')
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.25, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("CR_time_decay_factor_white.png", bbox_inches='tight')
# plt.show()
#
# ############################################## hispanic ####################################################
# #
# cmap = plt.get_cmap('Reds')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# cmap_modified = matplotlib.colors.LinearSegmentedColormap.from_list(
#     'RedsModified', cmap(np.linspace(0.3, 1.1, num_shades)))
# curve_colors = [cmap_modified(i / (num_shades - 1)) for i in range(num_shades)]
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# for i in range(len(alpha_list)):
#     ax.plot(x_list, hispanic_time_decay_dif_alpha[i], linewidth=3, markersize=6,
#             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
# # alpha = 1
# ax.plot(x_list, hispanic_time_decay_alpha1, linewidth=3, markersize=6,
#         label='{}'.format(1), linestyle='-', marker='o', color="black")
#
# # traditional
# ax.plot(x_list, hispanic_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="#fe01b1")
#
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# # plt.yticks([0.2, 0.3, 0.4], fontsize=20)
# # plt.yscale('symlog', linthresh=0.4)
# plt.yscale('log')
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.25, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.1, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
#
# plt.savefig("CR_time_decay_factor_hispanic.png", bbox_inches='tight')
# plt.show()
