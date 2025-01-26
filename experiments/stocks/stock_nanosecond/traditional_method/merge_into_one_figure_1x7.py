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

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
                    {"sector": 'Financial Services'}]

sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
                          'Healthcare', 'Financial Services']

window_size_unit_list = ["100ms", "200ms", "500ms", "800ms", "1s", "2s", "5s"]

result_file_list = dict()
for window_size in window_size_unit_list:
    file_name = f"traditional_accuracy_time_window_{window_size}.csv"
    df = pd.read_csv(file_name)
    df = df[df['sector'].isin(sorted_sector_list)]
    result_file_list[window_size] = df



print("draw the plot")

# Calculate the global y-axis limits based on all data
y_min = min(df['accuracy'].min() for df in result_file_list.values())
y_max = max(df['accuracy'].max() for df in result_file_list.values())
# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 7, figsize=(16, 1.8))
plt.subplots_adjust(top=0.85, bottom=0.33, hspace=0, wspace=0.25, left=0.04, right=0.99)



time_start = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')

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
sector_order = sorted_sector_list
lines = []
labels = []
for i in range(0, len(window_size_unit_list)):
    window_size = window_size_unit_list[i]
    if i == 0:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                         y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=0.8,
                         markersize=3, palette=curve_colors, hue_order=sector_order)
    elif i == 1:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                         y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=1,
                         markersize=3, palette=curve_colors, hue_order=sector_order)
    elif i == len(window_size_unit_list)-1:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                         y='accuracy', hue='sector', legend=True, marker='o', linestyle='-', linewidth=3,
                         markersize=6, palette=curve_colors, hue_order=sector_order)
    else:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event',
                     y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=3,
                         markersize=6, palette=curve_colors, hue_order=sector_order)
    xlabel = f"({chr(97 + i)}) window = {window_size_unit_list[i]}"
    axes[i].grid(True, alpha=1)
    axes[i].tick_params(axis='both', which='major', labelsize=15, pad=1.5)
    axes[i].set_yticks([0, 0.5, 1.0])
    axes[i].set_ylabel('')
    axes[i].set_xlabel(xlabel, fontsize=17, labelpad=0).set_position([0.5, 1])


    # Define the common x-tick labels for the first subplot
    manual_ticks = ['14:00:05.1+00:00', '14:00:10.1+00:00', '14:00:15.1+00:00']
    manual_tick_labels = [':05', ':10', ':15']
    df_below_threshold = []
    # Apply manual ticks and accuracy < 55% markers only to the first subplot
    if i == 0:
        print("Applying manual ticks for the first subplot.")

        # Convert the manual tick times to timestamps and interpolate positions
        lst = result_file_list[window_size]['ts_event'].unique().tolist()
        timestamp_values = np.array([pd.Timestamp(time).value for time in lst])  # Convert to numeric values
        tick_idx = []

        for tick_time in manual_ticks:
            tick_timestamp = pd.Timestamp("2024-10-15 " + tick_time).value  # Convert to numeric
            if timestamp_values[0] <= tick_timestamp and tick_timestamp <= timestamp_values[-1]:
                # Interpolate position if the tick timestamp is within the range
                interpolated_index = np.interp(tick_timestamp, timestamp_values, np.arange(len(lst)))
                tick_idx.append(interpolated_index)

        # Ensure the number of ticks matches the number of labels
        if len(tick_idx) == len(manual_tick_labels):
            axes[i].set_xticks(tick_idx)
            axes[i].set_xticklabels(manual_tick_labels, rotation=0, fontsize=17, fontweight='bold')
        else:
            print("Warning: Mismatch between tick positions and labels.")

    else:
        if i == 1:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.2]
        elif i == 2:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.45]
        elif i == 3:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.52]
        elif i == 4:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.54]
        elif i == 5:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.58]
        elif i == 6:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.63]
        elif i == 7:
            df_below_threshold = result_file_list[window_size][result_file_list[window_size]['accuracy'] < 0.68]

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

            # Create simplified label for the tick (e.g., just the minute and seconds)
            tick_labels.append(pd.Timestamp(tick_time).strftime(':%S'))

        # Set these timestamps as xticks with the simplified labels
        axes[i].set_xticks(tick_idx)
        axes[i].set_xticklabels(tick_labels, rotation=0, fontsize=17, fontweight='bold')


first_label = axes[1].get_xticklabels()[2]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(-4, 0))  # Shift 5 points left
first_label = axes[1].get_xticklabels()[3]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(8, 0))  # Shift 5 points left



first_label = axes[3].get_xticklabels()[0]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(-8, 0))  # Shift 5 points left
first_label = axes[3].get_xticklabels()[1]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(5, 0))  # Shift 5 points left


# Apply a shift to the first x-tick label
first_label = axes[4].get_xticklabels()[0]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(-8, 0))  # Shift 5 points left
first_label = axes[4].get_xticklabels()[1]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(5, 0))  # Shift 5 points left


# Apply a shift to the first x-tick label
first_label = axes[5].get_xticklabels()[0]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(-8, 0))  # Shift 5 points left
first_label = axes[5].get_xticklabels()[1]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(0, 0))  # Shift 5 points left
first_label = axes[5].get_xticklabels()[2]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(5, 0))  # Shift 5 points left
first_label = axes[5].get_xticklabels()[3]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(-8, 0))  # Shift 5 points left
first_label = axes[5].get_xticklabels()[4]
first_label.set_transform(first_label.get_transform() + Affine2D().translate(2, 0))  # Shift 5 points left


# Select the subplot where you want to remove the 3rd x-tick (index 3 refers to the 4th subplot)
subplot_index = 4

# Get current x-ticks and labels
current_ticks = axes[subplot_index].get_xticks()
current_labels = [label.get_text() for label in axes[subplot_index].get_xticklabels()]

updated_ticks = list(current_ticks)
updated_labels = list(current_labels)
del updated_ticks[3]  # Remove the 3rd tick
del updated_labels[3]  # Remove the 3rd label

# Set the updated ticks and labels
axes[subplot_index].set_xticks(updated_ticks)
axes[subplot_index].set_xticklabels(updated_labels, rotation=0, fontsize=17)




# Redraw the figure to apply the transformation
plt.draw()

# Set y-label only for the first subplot
axes[0].set_ylabel('Accuracy', fontsize=17, labelpad=1)


axes[2].set_ylim(0.3, 0.8)
axes[2].set_yscale("symlog", linthresh=0.68)
axes[2].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7], ["0.3", "0.4", "0.5", "0.6", "0.7"], fontsize=16)



axes[3].set_ylim(0.4, 0.75)
axes[3].set_yscale("symlog", linthresh=0.7)
axes[3].set_yticks([0.4, 0.5, 0.6, 0.7], ["0.4", "0.5", "0.6", "0.7"], fontsize=16)


axes[4].set_ylim(0.45, 0.75)
axes[4].set_yscale("symlog", linthresh=0.66)
axes[4].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"], fontsize=16)



axes[5].set_ylim(0.45, 0.75)
axes[5].set_yscale("symlog", linthresh=0.66)
axes[5].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"], fontsize=16)




axes[6].set_ylim(0.5, 0.75)
axes[6].set_yscale("symlog", linthresh=0.66)
axes[6].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"], fontsize=16)


#
sns.move_legend(axes[len(window_size_unit_list)-1], "upper left", fontsize=17,
                bbox_to_anchor=(-7.9, 1.45), ncol=9,
                labelspacing=0.3, draggable=True,
                handletextpad=0.3, markerscale=2, handlelength=1.8, handleheight=1,
               columnspacing=0.3, borderpad=0.2, frameon=False, title=None)



plt.tight_layout()
fig.savefig("traditional_accuracy_time_window_change.png")
plt.show()
