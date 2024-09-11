import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import Accuracy_workload as workload
import seaborn as sns
import colorsys
import csv

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


data = pd.read_csv('../../../data/name_gender/baby_names_1880_2020_US_predicted.csv')
print(data["sex"].unique())
date_column = "year"
date_time_format = False
time_window_str = "10"
monitored_groups = [{"sex": 'male'}, {"sex": 'female'}]
print(data[:5])
label_prediction = "predicted_gender"
label_ground_truth = "sex"
alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
threshold = 0.5

male_time_decay_dif_alpha = []
female_time_decay_dif_alpha = []

for alpha in alpha_list[:-1]:
    print("\nalpha = {}".format(alpha))
    # use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
    (DFMonitor, uf_list_DF, accuracy_list_DF, counter_list_correct_DF,
     counter_list_incorrect_DF) = workload.traverse_data_DFMonitor(data, date_column,
                                                                   time_window_str, date_time_format,
                                                                   monitored_groups,
                                                                   threshold,
                                                                   alpha, label_prediction,
                                                                   label_ground_truth)

    # draw chart of the first and second value in all lists in fpr_list and fpr_list1
    # 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
    male_time_decay = [x[0] for x in accuracy_list_DF]
    female_time_decay = [x[1] for x in accuracy_list_DF]

    male_time_decay_dif_alpha.append(male_time_decay)
    female_time_decay_dif_alpha.append(female_time_decay)

## alpha = 1
(DFMonitor1, uf_list_DF1, accuracy_list_DF1, counter_list_TN_DF1,
 counter_list_FP_DF1) = workload.traverse_data_DFMonitor(data, date_column,
                                                         time_window_str, date_time_format,
                                                         monitored_groups,
                                                         threshold,
                                                         1, label_prediction,
                                                         label_ground_truth)

male_time_decay_dif_alpha.append([x[0] for x in accuracy_list_DF1])
female_time_decay_dif_alpha.append([x[1] for x in accuracy_list_DF1])


uf_list_trad, accuracy_list_trad, counter_list_correct_trad, counter_list_incorrect_trad \
    = workload.Accuracy_traditional(data, date_column,
                              time_window_str, date_time_format,
                              monitored_groups,
                              threshold,
                              label_prediction,
                              label_ground_truth)

# draw chart of the first and second value in all lists in fpr_list and fpr_list1
# 'Caucasian' 'African-American' 'Other' 'Hispanic' 'Asian' 'Native American'
male_traditional = [x[0] for x in accuracy_list_trad]
female_traditional = [x[1] for x in accuracy_list_trad]


print(len(male_time_decay_dif_alpha), len(female_time_decay_dif_alpha),
      len(male_time_decay_dif_alpha[0]), len(male_traditional))

x_list = np.arange(0, len(male_traditional))
# pair_colors = ['blue', '#27bcf6', 'green', 'lightgreen', 'chocolate', 'gold']
# Define the number of shades and the colormap
num_shades = len(alpha_list)

print("male_time_decay_dif_alpha", len(male_time_decay_dif_alpha))

with open("Accuracy_time_decay_factor.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["alpha", "male_time_decay", "female_time_decay"])
    writer.writerow(['t', male_traditional, female_traditional])
    for i in range(len(alpha_list)):
        writer.writerow([alpha_list[i], male_time_decay_dif_alpha[i],
                         female_time_decay_dif_alpha[i]])


# save results to a file
# with open("FPR_time_decay_factor.txt", "w") as f:
#     f.write("alpha\tmale_time_decay\tmale_traditional\tfemale_time_decay\tfemale_traditional\thispanic_time_decay\thispanic_traditional\n")
#     for i in range(0, len(male_time_decay_dif_alpha)):
#         print("i = ", i)
#         f.write(str(alpha_list[i]) + "\t" + str(male_time_decay_dif_alpha[i]) + "\t" + str(male_traditional[i]) + "\t"
#                 + str(female_time_decay_dif_alpha[i]) + "\t" + str(female_traditional[i]) + "\t"
#                 + str(hispanic_time_decay_dif_alpha[i]) + "\t" + str(hispanic_traditional[i]) + "\n")
#
# #
# ############################################## draw ####################################################
#
#
#
# fig, ax = plt.subplots(3, 2, figsize=(6, 10))
#
# for i in range(len(alpha_list)):
#     ax[0].plot(x_list, male_time_decay_dif_alpha[i], linewidth=3, markersize=6,
#             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
# # alpha = 1
# ax.plot(x_list, male_time_decay_alpha1, linewidth=3, markersize=6,
#         label='{}'.format(1), linestyle='-', marker='o', color="male")
#
# # traditional
# ax.plot(x_list, male_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="cyan")
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yscale('log')
# # plt.ylim(0.01, 0.6)
# # plt.yticks([0.0, 0.2])
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('false positive rate (FPR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.25, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("FPR_time_decay_factor_male.png", bbox_inches='tight')
# plt.show()
#
#
# ############################################## female ####################################################
# cmap = plt.get_cmap('Greens')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# cmap_modified = matplotlib.colors.LinearSegmentedColormap.from_list(
#     'GreensModified', cmap(np.linspace(0.5, 1.1, num_shades)))
# curve_colors = [cmap_modified(i / (num_shades - 1)) for i in range(num_shades)]
#
# color = matplotlib.colors.ColorConverter.to_rgb("#287c37")
# curve_colors = [scale_lightness(color, scale) for scale in [0.75, 1.25, 1.7, 2.2, 2.7]]
# curve_colors = curve_colors[::-1]
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# for i in range(len(alpha_list)):
#     ax.plot(x_list, female_time_decay_dif_alpha[i], linewidth=3, markersize=6,
#             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
# # alpha = 1
# ax.plot(x_list, female_time_decay_alpha1, linewidth=3, markersize=6,
#         label='{}'.format(1), linestyle='-', marker='o', color="male")
#
# # Plot the second curve (y2_values)
# ax.plot(x_list, female_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="#cccc00")
#
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# # plt.yticks([0.2, 0.3, 0.4], fontsize=20)
# plt.yscale('symlog', linthresh=0.28)
#
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('false positive rate (FPR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.12, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("FPR_time_decay_factor_female.png", bbox_inches='tight')
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
#         label='{}'.format(1), linestyle='-', marker='o', color="male")
#
# # traditional
# ax.plot(x_list, hispanic_traditional, linewidth=3, markersize=6, label='traditional', linestyle=':',
#         marker='s', color="#fe01b1")
#
# plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# # plt.yticks([0.2, 0.3, 0.4], fontsize=20)
# plt.yscale('symlog', linthresh=0.4)
#
# plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('false positive rate (FPR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[5], handles[2], handles[4], handles[1], handles[3], handles[0], handles[6]]
# labels = [labels[5], labels[2], labels[4], labels[1], labels[3], labels[0], labels[6]]
# plt.legend(title='Alpha Values', title_fontsize=15, loc='lower left', bbox_to_anchor=(-0.12, 0.96), fontsize=17,
#            ncol=4, labelspacing=0.1, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
#
# plt.savefig("FPR_time_decay_factor_hispanic.png", bbox_inches='tight')
# plt.show()
