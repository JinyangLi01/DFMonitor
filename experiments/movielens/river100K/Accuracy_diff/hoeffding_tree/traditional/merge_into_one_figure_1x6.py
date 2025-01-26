import ast
from datetime import datetime
import matplotlib.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import Affine2D


# Set the global font family to Arial
plt.rcParams['font.family'] = 'arial'
sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)

monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
window_size_unit_list = ["1D", "3D", "1W", "2W", "1M", "3M"]

decline_points = {x:[] for x in window_size_unit_list}
differences = {x:[] for x in window_size_unit_list}
result_file_list = dict()
for window_size in window_size_unit_list:
    method_name = "hoeffding_classifier"
    df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender_{window_size}.csv", index_col=0)
    df = df[["gender", "datetime", "calculated_value"]]
    df['datetime'] = pd.to_datetime(df['datetime'])
    result_file_list[window_size] = df

datetime_list = dict()
print("Draw the plot")


# Set up a row of 6 subplots with a shared y-axis
fig, axes = plt.subplots(1, 6, figsize=(14, 1.7))
plt.subplots_adjust(top=0.87, bottom=0.42, hspace=0, wspace=0.25, left=0.03, right=0.99)

# Define color palette for consistent coloring
curve_colors = sns.color_palette(['blue', '#ffb400'])

# Plot the data
sector_order = ["M", "F"]
for i, window_size in enumerate(window_size_unit_list):
    data = result_file_list[window_size]
    subdata = data[data["gender"] == "M"]
    datetime_list[window_size] = subdata['datetime'].apply(lambda x: x.strftime('%m/%d/%Y')).tolist()  # Corrected line
    print(result_file_list[window_size].head())  # Debug: Print the head of each DataFrame
    x_list = range(len(data))
    male_list = data[data["gender"] == "M"]["calculated_value"].tolist()
    female_list = data[data["gender"] == "F"]["calculated_value"].tolist()
    if i == 0:
        axes[i].plot(np.arange(len(male_list)), male_list, linewidth=1, markersize=1.5,
                     label="male", linestyle='-', marker='o', color="blue")
        axes[i].plot(np.arange(len(female_list)), female_list, linewidth=1, markersize=1.5,
                     label='female', linestyle='-', marker='o', color="orange")
    elif i == 1 or i == 2:
        axes[i].plot(np.arange(len(male_list)), male_list, linewidth=2, markersize=2.5,
                     label="male", linestyle='-', marker='o', color="blue")
        axes[i].plot(np.arange(len(female_list)), female_list, linewidth=2, markersize=2.5,
                     label='female', linestyle='-', marker='o', color="orange")
    elif i == 3 or i == 4 or i == 5:
        axes[i].plot(np.arange(len(male_list)), male_list, linewidth=2, markersize=5,
                 label="male", linestyle='-', marker='o', color="blue")
        axes[i].plot(np.arange(len(female_list)), female_list, linewidth=2, markersize=5,
                 label='female', linestyle='-', marker='o', color="orange")

    axes[i].grid(True, which='both', linestyle='-', linewidth=0.8)

    xlabel = f"({chr(97 + i)}) window = {window_size}"
    axes[i].set_ylabel('')
    axes[i].set_xlabel(xlabel, fontsize=17, labelpad=-1).set_position([0.5, 0.9])
    axes[i].tick_params(axis='y', pad=-1)
    if i == 1:
        axes[i].set_xlabel(xlabel, fontsize=17, labelpad=-5).set_position([0.5, 0.9])
    decline_threshold = 0.1
    if i == 4:
        decline_threshold = 0.07
    elif i == 5:
        decline_threshold = 0.07
    print(decline_points)
    for k in range(0, len(female_list)):
        if k == 5:
            if abs(female_list[k - 1] - female_list[k]) > decline_threshold:
                #axes[i].axvline(x=i, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
                # plt.text(i, y_margin, check_points[i].replace(" ", "\n"), color='red', fontsize=13,
                #          verticalalignment='bottom', horizontalalignment='center')
                decline_points[window_size].append(k)
                differences[window_size].append(female_list[k - 1] - female_list[i])
        else:
            if (female_list[k - 1] - female_list[k]) > decline_threshold:
                #axes[i].axvline(x=i, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
                decline_points[window_size].append(k)
                differences[window_size].append(female_list[k - 1] - female_list[k])




axes[0].set_yticks([0.3, 0.6, 1.0], ["0.3", "0.5", "1.0"], fontsize=16)
axes[1].set_yticks([0.5, 0.7, 1.0], ["0.5", "0.7", "1.0"], fontsize=16)
axes[2].set_yticks([0.5, 0.7, 1.0], ["0.5", "0.7", "1.0"], fontsize=16)
axes[3].set_yticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"], fontsize=16)
axes[4].set_yticks([0.7, 0.8, 0.9], ["0.7", "0.8", "0.9"], fontsize=16)
axes[5].set_yticks([0.7, 0.8, 0.9], ["0.7", "0.8", "0.9"], fontsize=16)

ticks = []
window_size = "1D"
idx = datetime_list[window_size].index('11/02/1997')
ticks.append(idx)
idx = datetime_list[window_size].index('01/25/1998')
ticks.append(idx)
idx = datetime_list[window_size].index('03/22/1998')
ticks.append(idx)
axes[0].set_xticks(ticks, ['11/02/1997         ', '01/25/1998  ', '03/22/1998  '], fontsize=15, rotation=10)

labels = axes[0].get_xticklabels()
labels[0].set_y(0.16)
labels[0].set_x(labels[0].get_position()[0] - 3)
labels[1].set_y(0.12)
labels[1].set_x(labels[1].get_position()[0] - 1)
labels[2].set_y(0)
labels[2].set_x(labels[2].get_position()[0] - 3)





ticks = []
window_size = "3D"
axes[1].set_xticks([13, 31, 40, 59], [datetime_list[window_size][13] + "         ",
                                      datetime_list[window_size][31] + "        ",
                                      "      "  + datetime_list[window_size][40],
                                      "     "+datetime_list[window_size][59]],
               fontsize=13, rotation=10)


labels = axes[1].get_xticklabels()
axes[1].tick_params(axis='x', pad=-1)
labels[0].set_y(0.13)
labels[0].set_x(labels[0].get_position()[0] - 3)
labels[1].set_y(0.03)
labels[1].set_x(labels[1].get_position()[0] - 2)
labels[2].set_y(-0.1)
labels[2].set_x(labels[2].get_position()[0] - 0)
labels[3].set_y(-0.19)
labels[3].set_x(labels[3].get_position()[0] + 2)




labels = axes[2].get_xticklabels()
axes[2].tick_params(axis='x', pad=0)
labels[0].set_y(0.1)
labels[0].set_x(labels[0].get_position()[0] - 3)
labels[1].set_y(0.07)
labels[1].set_x(labels[1].get_position()[1] - 1.5)
labels[2].set_y(-0.03)
labels[2].set_x(labels[2].get_position()[1] - 1)
# axes[2].set_xlabel(window_size_unit_list[2], labelpad=-5)



axes[2].set_xticks(decline_points["1W"]+[26], [datetime_list["1W"][decline_points["1W"][i]].replace(" ", "\n") +
                                         "      " for i in range(len(decline_points["1W"]))] + ["     " +
                                                                datetime_list["1W"][26].replace(" ", "\n")],
           color='black',
            rotation=10, fontsize=15)





xticks = [datetime_list["2W"][decline_points["2W"][0]] + "       ", "      " +
          datetime_list["2W"][decline_points["2W"][1]]]
axes[3].set_xticks(decline_points["2W"], xticks, fontsize=15, rotation=10)

labels = axes[3].get_xticklabels()
axes[3].tick_params(axis='x', pad=0)
labels[0].set_y(0.1)
labels[0].set_position((labels[0].get_position()[0] -19, labels[0].get_position()[1]))
labels[1].set_y(-0.02)
labels[1].set_position((labels[1].get_position()[0] -10, labels[1].get_position()[1]))



xticks = [datetime_list["1M"][decline_points["1M"][0]]]
axes[4].set_xticks(decline_points["1M"]+[1], [datetime_list["1M"][decline_points["1M"][i]].replace(" ", "\n")
                                              for i in range(len(decline_points["1M"]))] +
                   [datetime_list["1M"][1].replace(" ", "\n")],
               color='black',
                rotation=10, fontsize=15)

labels = axes[4].get_xticklabels()
axes[4].tick_params(axis='x', pad=0)
labels[0].set_y(-0.05)
labels[0].set_position((labels[0].get_position()[0], labels[0].get_position()[1]))
labels[1].set_y(0.0)
labels[1].set_position((labels[1].get_position()[0] , labels[1].get_position()[1]))


print(datetime_list["3M"])
# axes[5].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=15)
axes[5].set_xticks([0, 1, 2, len(datetime_list["3M"]) - 1])  # Adjust as needed for more or fewer points
axes[5].set_xticklabels(["", datetime_list["3M"][1] + "       ",
                         "       " + datetime_list["3M"][2], ""], color='black', rotation=10, fontsize=15)
labels = axes[5].get_xticklabels()
axes[5].tick_params(axis='x', pad=0)
labels[0].set_y(0.05)
labels[1].set_y(0.05)
# Set the x-axis limits to the data range to prevent squishing
axes[5].set_xlim(0, len(datetime_list["3M"]) - 1)


# Add legend in the last subplot
handles, labels = axes.flat[0].get_legend_handles_labels()
leg = axes[0].legend(handles=handles, labels=labels, loc="upper left", fontsize=16,
                      bbox_to_anchor=(0, 1.5), ncol=2, labelspacing=0.8, draggable=True,
                      handletextpad=0.8, markerscale=6, handlelength=2, handleheight=1,
                      columnspacing=0.8, borderpad=0.2, frameon=False)

# Modify the line width in the legend handles
for line in leg.get_lines():
    line.set_linewidth(4)


fig.savefig("traditional_accuracy_time_window_change.png")
plt.show()
