import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.transforms import Affine2D
# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)

# monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
#                     {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
#                     {"sector": 'Financial Services'}]

# sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
#                           'Healthcare', 'Financial Services']

sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services']
#
# window_size_unit_list = ["100ms", "200ms", "500ms", "1s", "2s", "5s"]


window_size_unit_list = ["20ms", "100ms", "500ms"]


result_file_list = dict()
for window_size in window_size_unit_list:
    file_name = f"traditional_accuracy_time_window_{window_size}.csv"
    df = pd.read_csv(file_name)
    # df = df[df['sector'].isin(sorted_sector_list)]
    df = df[df['sector'].isin(sector_list)]
    result_file_list[window_size] = df



print("draw the plot")

# Calculate the global y-axis limits based on all data
y_min = min(df['accuracy'].min() for df in result_file_list.values())
y_max = max(df['accuracy'].max() for df in result_file_list.values())
# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.4))
plt.subplots_adjust(top=0.85, bottom=0.35, hspace=0.3, wspace=0.27, left=0.05, right=0.99)



time_start = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')

curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ 'blue', 'limegreen', '#ffb400', 'darkviolet','cyan', 'black',
                                            "red", 'magenta'])
# curve_colors = sns.color_palette([
#     'Turquoise',  # Indigo
#     '#ff7f0e',  # Bright Orange
#     '#228b22',  # Forest Green
#     '#8a2be2',  # Soft Purple
#     '#e41a1c',  # Bright Red
#     '#ffb400',  # Golden Yellow
#     'black'   # Charcoal Gray
# ])
# curve_colors = sns.color_palette("tab10", 7)

# Plot the data
sector_order = sector_list
lines = []
labels = []
for i in range(0, len(window_size_unit_list)):
    window_size = window_size_unit_list[i]
    if i == 0:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                         y='accuracy', hue='sector', legend=True, marker='o', linestyle='-', linewidth=1,
                         markersize=3, palette=curve_colors, hue_order=sector_order)
    else:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                         y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=2.2,
                         markersize=3, palette=curve_colors, hue_order=sector_order)
    xlabel = f"({chr(97 + i)}) window = {window_size}"
    axes[i].grid(True, alpha=1)
    axes[i].tick_params(axis='both', labelsize=15, pad=1.5)
    axes[i].set_ylabel('')
    axes[i].set_xlabel(xlabel, fontsize=14, labelpad=2.3)



    # Apply manual ticks and accuracy < 55% markers only to the first subplot
    df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.6]

    # Extract unique timestamps where accuracy is below the threshold
    unique_below_threshold_times = df_below_threshold["ts_event"].unique()

    # Convert these timestamps to integer values for indexing and interpolation
    lst = result_file_list[window_size]['ts_event'].unique().tolist()
    timestamp_values = np.array([pd.Timestamp(time).value for time in lst])
    tick_idx = []
    tick_labels = []

    for tick_time in unique_below_threshold_times:
        tick_timestamp = pd.Timestamp(tick_time).value

        # Interpolate the approximate index for each time point where accuracy < 55%
        interpolated_index = np.interp(tick_timestamp, timestamp_values, range(len(lst)))
        tick_idx.append(interpolated_index)
    if i < 1:
        tick_labels = [' '] * len(tick_idx)
    # elif i == 5:
    #     tick_idx = [0]
    #     tick_labels = [" "]
    else:
        tick_labels = [pd.Timestamp(tick_time).strftime('.%f')[:-4] for tick_time in unique_below_threshold_times]

    # Set these timestamps as xticks with the simplified labels
    axes[i].set_xticks(tick_idx)
    if i == 2:
        # Set the tick labels for the last two subplots
        axes[i].set_xticklabels(tick_labels, rotation=0, fontsize=12)
    else:
        # Set the tick labels for the first subplot
        axes[i].set_xticklabels(tick_labels, rotation=0, fontsize=12)




# Redraw the figure to apply the transformation
plt.draw()

# Set y-label only for the first subplot
# axes[0].set_ylabel('Accuracy', fontsize=14, labelpad=-1)
# axes[1][0].set_ylabel('Accuracy', fontsize=14, labelpad=-1)



axes[0].set_ylim(0.0, 1)
axes[0].set_yscale("symlog", linthresh=0.9)
axes[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8], ["0.0", "0.2", "0.4", "0.6", "0.8"], fontsize=14)

axes[1].set_ylim(0.55, 0.82)
axes[1].set_yscale("symlog", linthresh=0.9)
axes[1].set_yticks([0.6, 0.7, 0.8], ["0.6", "0.7", "0.8"], fontsize=14)

axes[2].set_ylim(0.55, 0.73)
axes[2].set_yscale("symlog", linthresh=0.9)
axes[2].set_yticks([ 0.6, 0.7], ["0.6", "0.7"], fontsize=14)
axes[2].xaxis.set_label_coords(0.5, -0.34)  # adjust second number as needed


#
sns.move_legend(axes[0], "upper left", fontsize=14,
                bbox_to_anchor=(0, 1.5), ncol=3,
                labelspacing=0.4, draggable=True,
                handletextpad=0.3, markerscale=2, handlelength=1.8, handleheight=1,
               columnspacing=0.5, borderpad=0.2, frameon=False, title=None)


fig.savefig("traditional_accuracy_time_window_change_1x3.png")
plt.show()
