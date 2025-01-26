import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform

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

window_size_unit_list = ["100ms", "200ms", "500ms", "800ms", "1s", "2s", "5s", "6s"]

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
fig, axes = plt.subplots(1, 8, figsize=(20, 2.5))
plt.subplots_adjust(top=0.6, bottom=0.2, hspace=0.1, wspace=0.28)


curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ 'blue', 'darkorange', 'limegreen', 'cyan', "red", 'black',
                                           'darkviolet', 'magenta'])

# curve_colors = sns.color_palette("tab10", 7)

# Plot the data
lines = []
labels = []
for i in range(0, len(window_size_unit_list)):
    print(result_file_list[window_size_unit_list[i]])
    if i == 0:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event_' + window_size_unit_list[i],
                         y='accuracy', hue='sector', legend=True, marker='o', linestyle='-', linewidth=0.8,
                         markersize=3, palette=curve_colors)
    elif i == 1:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event_' + window_size_unit_list[i],
                         y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=0.8,
                         markersize=3, palette=curve_colors)
    else:
        g = sns.lineplot(ax=axes[i], data=result_file_list[window_size_unit_list[i]],
                         x='ts_event_' + window_size_unit_list[i],
                     y='accuracy', hue='sector', legend=False, marker='o', linestyle='-', linewidth=1.2,
                         markersize=4, palette=curve_colors)
    xlabel = f"({chr(97 + i)}) {window_size_unit_list[i]}"
    axes[i].grid(True, alpha=0.2)
    axes[i].tick_params(axis='both', which='major', labelsize=15, pad=1)
    axes[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[i].set_ylabel('')
    axes[i].set_xlabel(xlabel, fontsize=15).set_position([0.5, -1])

    tick_names = ['14:00:05+00:00', '14:00:10+00:00', '14:00:15+00:00']
    tick_labels = [':05', ':10', ':15']
    tick_idx = []

    # Convert tick names to timestamps in the common global range for consistent scaling
    tick_times = [pd.Timestamp("2024-10-15 " + t) for t in tick_names]
    # Extract and convert timestamps to integer time values for current subplot
    lst = result_file_list[window_size_unit_list[i]]['ts_event_' + window_size_unit_list[i]].unique().tolist()
    timestamp_values = np.array([pd.Timestamp(time).value for time in lst])
    print("timestamp_values", timestamp_values)

    for tick_time in tick_names:
        full_tick = "2024-10-15 " + tick_time
        tick_timestamp = pd.Timestamp(full_tick).value

        # Check if tick is within the range of available timestamps for this subplot
        if tick_timestamp < timestamp_values[0] or tick_timestamp > timestamp_values[-1]:
            continue

        # Interpolate the approximate index for the tick
        interpolated_index = np.interp(tick_timestamp, timestamp_values, range(len(lst)))
        tick_idx.append(interpolated_index)

    # Set the ticks and labels for the x-axis
    axes[i].set_xticks(tick_idx)
    axes[i].set_xticklabels(tick_labels, rotation=0, fontsize=15)
    # Collect handles and labels for the legend
    handles, current_labels = axes[i].get_legend_handles_labels()

# Set y-label only for the first subplot
axes[0].set_ylabel('Accuracy', fontsize=15, labelpad=1)


axes[2].set_ylim(0.3, 0.8)
axes[2].set_yscale("symlog", linthresh=0.68)
axes[2].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7], ["0.3", "0.4", "0.5", "0.6", "0.7"])



axes[3].set_ylim(0.4, 0.75)
axes[3].set_yscale("symlog", linthresh=0.7)
axes[3].set_yticks([0.4, 0.5, 0.6, 0.7], ["0.4", "0.5", "0.6", "0.7"])


axes[4].set_ylim(0.45, 0.75)
axes[4].set_yscale("symlog", linthresh=0.66)
axes[4].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"])



axes[5].set_ylim(0.45, 0.75)
axes[5].set_yscale("symlog", linthresh=0.66)
axes[5].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"])




axes[6].set_ylim(0.5, 0.75)
axes[6].set_yscale("symlog", linthresh=0.66)
axes[6].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"])



axes[7].set_ylim(0.5, 0.75)
axes[7].set_yscale("symlog", linthresh=0.66)
axes[7].set_yticks([0.5, 0.6, 0.7], ["0.5", "0.6", "0.7"])



sns.move_legend(axes[0], "upper left", fontsize=15, bbox_to_anchor=(-0.2, 1.4), ncol=9,
                labelspacing=0.5,
                handletextpad=0.5, markerscale=1.5, handlelength=1.8,
               columnspacing=0.8, borderpad=0.2, frameon=True, title=None)

fig.savefig("traditional_accuracy_time_window_change.png")
plt.show()
