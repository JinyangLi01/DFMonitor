import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from algorithm.fixed_window import Accuracy_workload as workload
import seaborn as sns
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

def rgb_to_hex(rgb):
    """Convert an RGB tuple to a hex color code."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


def scale_lightness(rgb, scale):
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Scale the lightness
    l = max(min(l * scale, 1.0), 0.0)  # Ensure lightness remains in [0, 1]
    # Convert HLS back to RGB
    return colorsys.hls_to_rgb(h, l, s)


def generate_scaled_colors(base_color, num_colors, scale_range=(0.5, 1.5)):
    rgb_base = mcolors.to_rgb(base_color)
    if num_colors == 1:
        # If only one color is requested, return the base color
        return [rgb_base]
    scales = [scale_range[0] + (scale_range[1] - scale_range[0]) * i / (num_colors - 1) for i in range(num_colors)]
    print(scales)
    return [scale_lightness(rgb_base, scale) for scale in scales]


data = pd.read_csv('../../../data/name_gender/baby_names_1880_2020_predicted.csv')
print(data["sex"].unique())
date_column = "year"
date_time_format = False
time_window_str = "10"
monitored_groups = [{"sex": 'male'}, {"sex": 'female'}]

window_size_str_list = ["2 year", "5 year", "10 year", "20 year", "40 year", "60 year", "80 year", "100 year"]
window_size_str_list_brev = ["2", "5", "10", "20", "40", "60", "80", "100"]
threshold = 0.3
alpha = 0.5
label_prediction = "predicted_gender"
label_ground_truth = "sex"
male_time_decay_dif_alpha = []
female_time_decay_dif_alpha = []

black_time_decay = []
for time_window_str in window_size_str_list_brev:
    # print("\nwindow_str = {}".format(time_window_str))
    # use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
    (DFMonitor, uf_list_DF, accuracy_list_DF, counter_list_correct_DF,
    counter_list_incorrect_DF) = workload.traverse_data_DFMonitor(data, date_column, time_window_str,
                                                                                      date_time_format,
                                                                                      monitored_groups,
                                                                                      threshold,
                                                                                      alpha, label_prediction,
                                                                                      label_ground_truth)
    male_time_decay = [x[0] for x in accuracy_list_DF]
    female_time_decay = [x[1] for x in accuracy_list_DF]

    male_time_decay_dif_alpha.append(male_time_decay)
    female_time_decay_dif_alpha.append(female_time_decay)



with open("Accuracy_window_size.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["window_size", "male_time_decay", "female_time_decay"])
    for i in range(len(window_size_str_list_brev)):
        writer.writerow([window_size_str_list[i], male_time_decay_dif_alpha[i],
                         female_time_decay_dif_alpha[i]])


# x_list = np.arange(0, len(window_size_str_list))
# # pair_colors = ['blue', '#27bcf6', 'green', 'lightgreen', 'chocolate', 'gold']
# # Define the number of shades and the colormap
# num_shades = len(window_size_str_list)
#
# ############################################## black ####################################################
#
#
# # cmap = plt.get_cmap('Blues')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# # cmap_modified = matplotlib.colors.LinearSegmentedColormap.from_list(
# #     'BluesModified', cmap(np.linspace(0.2, 1.1, num_shades)))
# # curve_colors = [cmap_modified(i / (num_shades - 1)) for i in range(num_shades)]
# # #
#
# # hex to rgb
# c1 = str(plotly.colors.hex_to_rgb("#3873F1"))
# c2 = str(plotly.colors.hex_to_rgb("#55F909"))
# n_steps = len(window_size_str_list)
# colorscale = plotly.colors.n_colors(c1, c2, n_steps, colortype='rgb')
# colorscale = [x[4:-1].split(', ') for x in colorscale]
# rgb_list = [[int(float(y)) for y in x] for x in colorscale]
# # print(rgb_list, type(rgb_list), type(rgb_list[0]))
# curve_colors = ['#%02x%02x%02x' % tuple(x for x in y) for y in rgb_list]
#
#
# palette = sns.color_palette("PuBu", n_colors=256)  # Generate a continuous palette
# curve_colors = [palette[int(i)] for i in np.linspace(20, 255, len(window_size_str_list)+2, endpoint=False)]
# curve_colors = curve_colors[::-1]
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# # print(window_size_str_list)
# # print(len(black_time_decay_dif_alpha))
# # for j in range(len(black_time_decay_dif_alpha)):
# #     print(j, len(black_time_decay_dif_alpha[j]))
#
# # Find the maximum number of points among all curves to define a common x-axis length
# max_length = max(len(curve) for curve in black_time_decay_dif_alpha)
#
# for i in range(len(window_size_str_list)):
#     curve = black_time_decay_dif_alpha[i]
#     print(curve)
#     # Calculate the proportional x positions for the current curve's data points
#     proportional_x = np.linspace(0, max_length - 1, len(curve))
#     plt.plot(proportional_x, curve, linewidth=2, markersize=6,
#              label='{}'.format(window_size_str_list_brev[i]), linestyle='-', marker='o',
#              color=curve_colors[i])
#
# plt.xticks(np.arange(0, len(black_time_decay_dif_alpha[0])), [], rotation=0, fontsize=20)
# plt.yscale('log')
# # plt.yscale('symlog', linthresh=0.15)
# plt.yticks([0.4, 0.5, 0.6], ['0.4', '0.5', '0.6'])
#
# plt.xlabel('normalized measuring time',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[0], handles[4], handles[1], handles[5], handles[2], handles[6], handles[3], handles[7]]
# labels = [labels[0], labels[4], labels[1], labels[5], labels[2], labels[6], labels[3], labels[7]]
# plt.legend(title='Alpha Values', title_fontsize=17, loc='lower left', bbox_to_anchor=(-0.12, 0.96), fontsize=20,
#            ncol=4, labelspacing=0.1, handletextpad=0.2, markerscale=1.4,
#            columnspacing=0.3, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("CR_window_size_black.png", bbox_inches='tight')
# plt.show()
#
# ############################################## white ####################################################
# cmap = plt.get_cmap('Greens')
#
# # # Generate a list of shades of blue
# # blue_shades = [cmap(i / (num_shades - 1)) for i in range(num_shades)]
#
# c1 = str(plotly.colors.hex_to_rgb("#297E1B"))
# c2 = str(plotly.colors.hex_to_rgb("#0BF5E4"))
# n_steps = len(window_size_str_list)
# colorscale = plotly.colors.n_colors(c1, c2, n_steps, colortype='rgb')
# colorscale = [x[4:-1].split(', ') for x in colorscale]
# rgb_list = [[int(float(y)) for y in x] for x in colorscale]
# print(rgb_list, type(rgb_list), type(rgb_list[0]))
# curve_colors = ['#%02x%02x%02x' % tuple(x for x in y) for y in rgb_list]
#
# palette = sns.color_palette("summer", n_colors=256)  # Generate a continuous palette
# curve_colors = [palette[int(i)] for i in np.linspace(0, 255, len(window_size_str_list)*2-1, endpoint=False)]
#
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# for i in range(len(window_size_str_list)):
#     curve = white_time_decay_dif_alpha[i]
#     proportional_x = np.linspace(0, max_length - 1, len(curve))
#     plt.plot(proportional_x, curve, linewidth=2, markersize=6,
#              label='{}'.format(window_size_str_list_brev[i]), linestyle='-', marker='o',
#              color=curve_colors[i*2])
#
# plt.xticks(np.arange(0, len(white_time_decay_dif_alpha[0])), [], rotation=0, fontsize=20)
# plt.yscale('log')
# # plt.yscale('symlog', linthresh=0.3)
# plt.yticks([0.25, 0.3, 0.4, 0.5], ['0.25', '0.3', '0.4', '0.5'])
# plt.xlabel('normalized measuring time',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[0], handles[4], handles[1], handles[5], handles[2], handles[6], handles[3], handles[7]]
# labels = [labels[0], labels[4], labels[1], labels[5], labels[2], labels[6], labels[3], labels[7]]
# plt.legend(title='Alpha Values', title_fontsize=17, loc='lower left', bbox_to_anchor=(-0.15, 0.96), fontsize=20,
#            ncol=4, labelspacing=0.1, handletextpad=0.2, markerscale=1.4,
#            columnspacing=0.3, borderpad=0.2, frameon=True, handles=handles, labels=labels)
# plt.savefig("CR_window_size_white.png", bbox_inches='tight')
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
# c1 = str(plotly.colors.hex_to_rgb("#EA4A13"))
# c2 = str(plotly.colors.hex_to_rgb("#F4FB03"))
# n_steps = len(window_size_str_list)
# colorscale = plotly.colors.n_colors(c1, c2, n_steps, colortype='rgb')
# colorscale = [x[4:-1].split(', ') for x in colorscale]
# rgb_list = [[int(float(y)) for y in x] for x in colorscale]
# print(rgb_list, type(rgb_list), type(rgb_list[0]))
# curve_colors = ['#%02x%02x%02x' % tuple(x for x in y) for y in rgb_list]
#
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
#
# # palette = sns.color_palette("YlOrBr", n_colors=256)
# palette = sns.color_palette("YlOrRd", n_colors=256)
# curve_colors = [palette[int(i)] for i in np.linspace(0, 255, len(window_size_str_list)+1, endpoint=False)]
# curve_colors = curve_colors[::-1]
#
# for i in range(len(window_size_str_list)):
#     curve = hispanic_time_decay_dif_alpha[i]
#     proportional_x = np.linspace(0, max_length - 1, len(curve))
#     plt.plot(proportional_x, curve, linewidth=2, markersize=6,
#              label='{}'.format(window_size_str_list_brev[i]), linestyle='-', marker='o',
#              color=curve_colors[i])
# plt.xticks(np.arange(0, len(hispanic_time_decay_dif_alpha[0])), [], rotation=0, fontsize=20)
# # plt.yscale('log')
# plt.yscale('symlog', linthresh=0.2)
# plt.yticks([0.03, 0.06, 0.1, 0.15], ['0.03', '0.06', '0.1', '0.15'])
#
# plt.xlabel('normalized measuring time',
#            fontsize=20, labelpad=-2).set_position((0.47, -0.1))
# plt.ylabel('coverage ratio (CR)', fontsize=20, labelpad=-1)
# plt.grid(True)
# plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# handles = [handles[0], handles[4], handles[1], handles[5], handles[2], handles[6], handles[3], handles[7]]
# labels = [labels[0], labels[4], labels[1], labels[5], labels[2], labels[6], labels[3], labels[7]]
# plt.legend(title='Alpha Values', title_fontsize=17, loc='lower left', bbox_to_anchor=(-0.15, 0.96), fontsize=20,
#            ncol=4, labelspacing=0.1, handletextpad=0.2, markerscale=1.4,
#            columnspacing=0.3, borderpad=0.2, frameon=True, handles=handles, labels=labels)
#
# plt.savefig("CR_window_size_hispanic.png", bbox_inches='tight')
# plt.show()
