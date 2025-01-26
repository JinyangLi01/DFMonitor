import ast
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from polars.dependencies import subprocess
import matplotlib.transforms as mtransforms

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)




def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)




window_size_unit_list = ['1',  '12', '24*7']

result_file_list = dict()
alpha = 995
for window_size in window_size_unit_list:
    file_name = f"movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_alpha_{alpha}_time_unit_1 hour*{window_size}_check_interval_1 hour.csv"
    # if file exists
    if not os.path.exists(file_name):
        subprocess.run(["python3", "accuracy_gender_per_item_1hour.py", window_size])
        # Check again after running the script
        if not os.path.exists(file_name):
            raise Exception(f"Error: {file_name} was not created by the script.")
    df = pd.read_csv(file_name)
    print("window_size", window_size, df)
    result_file_list[window_size] = df



# draw the plot

fig, axs = plt.subplots(1, 3, figsize=(7, 1.4))
plt.subplots_adjust(top=0.86, bottom=0.3, hspace=0, wspace=0.3, left=0.07, right=0.99)
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ '#0000ff', 'cyan', '#57d357', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])


axis_number = np.arange(0, 10)
decline_threshold = 0.05
decline_points = []
differences = []


df = result_file_list[window_size_unit_list[0]]
x_list = np.arange(0, len(df))
df["check_points"] = pd.to_datetime(df["check_points"])
check_points = df["check_points"].tolist()
male_time_decay = df["male_time_decay"].tolist()
female_time_decay = df["female_time_decay"].tolist()
first_drop_index = None
first_drop_value = None
second_drop_index = None
second_drop_value = None
for j in range(100, len(female_time_decay) // 2):
    if first_drop_value is None:
        first_drop_value = female_time_decay[j]
    else:
        if first_drop_value > female_time_decay[j]:
            first_drop_index = j
            first_drop_value = female_time_decay[j]
for j in range(len(female_time_decay) // 2, len(female_time_decay) // 6 * 4):
    if second_drop_value is None:
        second_drop_value = female_time_decay[j]
    else:
        if second_drop_value > female_time_decay[j]:
            second_drop_index = j
            second_drop_value = female_time_decay[j]




pair_colors = ["blue", "orange"]
for i in range(len(window_size_unit_list)):
    window_size = window_size_unit_list[i]
    df = result_file_list[window_size]
    x_list = np.arange(0, len(df))
    male_time_decay = df["male_time_decay"].tolist()
    female_time_decay = df["female_time_decay"].tolist()
    axs[axis_number[i]].plot(x_list, male_time_decay, linewidth=1, markersize=1.2, label='Male', linestyle='-', marker='o', color="blue")
    axs[axis_number[i]].plot(x_list, female_time_decay, linewidth=1, markersize=1.1, label='Female', linestyle='-', marker='o',
            color="darkorange")
    axs.flat[i].set_xticks([], [], rotation=0, fontsize=14)
    axs.flat[i].set_ylim(0.65, 0.9)
    axs[axis_number[i]].tick_params(axis='x', which='major', pad=-3)
    axs[axis_number[i]].tick_params(axis='y', which='major', pad=0)
    axs[axis_number[i]].axvline(x=first_drop_index, color='black', linestyle='--', linewidth=1.5)
    axs[axis_number[i]].axvline(x=second_drop_index, color='black', linestyle='--', linewidth=1.5)
    xticks_idx = [first_drop_index, second_drop_index]
    xtick_label = [check_points[first_drop_index].strftime("%m/%d/%Y").replace("/19", "/"),
                   check_points[second_drop_index].strftime("%m/%d/%Y").replace("/19", "/")]
    axs[axis_number[i]].set_xticks(xticks_idx, xtick_label, rotation=0, fontsize=14)
    first_label = axs[axis_number[i]].get_xticklabels()[0]
    first_label.set_transform(
        first_label.get_transform() + mtransforms.Affine2D().translate(-13, -6))  # Shift 5 points left
    first_label = axs[axis_number[i]].get_xticklabels()[1]
    first_label.set_transform(
        first_label.get_transform() + mtransforms.Affine2D().translate(10, -6))  # Shift 5 points left
    axs[axis_number[i]].grid(True)


# axs[0, 0].set_title('(a) window size = 1 second', y=-0.19, pad=-0.5)
# axs[0, 1].set_title('(d) CR, Black', y=-0.19, pad=-0.5)
# axs[1, 0].set_title('(b) FPR, White', y=-0.19, pad=-0.5)
# axs[1, 1].set_title('(e) CR, White', y=-0.19, pad=-0.5)
# axs[2, 0].set_title('(c) FPR, Hispanic', y=-0.19, pad=-0.5)
# axs[2, 1].set_title('(f) CR, Hispanic', y=-0.19, pad=-0.5)

# for ax in axs.flat:
#     ax.grid(True)
#     ax.set_xticks(np.arange(0, max_length), [], rotation=0, fontsize=20)
#


# axs[0, 0].set_yscale('log')
axs.flat[0].set_yticks([0.65, 0.75, 0.85], ['0.65', '0.75', '0.85'], fontsize=14)
axs.flat[0].set_xlabel('(a) 1 hour', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs.flat[0].set_ylim(0.6, 0.88)

axs.flat[1].set_yticks([0.75, 0.8, 0.85], ['0.75', '0.80', '0.85'], fontsize=14)
axs.flat[1].set_xlabel('(b) 12 hours', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs.flat[1].set_ylim(0.73, 0.85)

# axs.flat[2].set_yscale('symlog', linthresh=0.22)
axs.flat[2].set_yticks([0.75, 0.8, 0.85], ['0.75', '0.80', '0.85'], fontsize=14)
axs.flat[2].set_xlabel('(c) 1 week', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs.flat[2].set_ylim(0.73, 0.85)



# fig.text(0.5, -1, 'differen time windows',
#          ha='center', va='center', fontsize=14)
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=14, rotation='vertical')

# create a common legend
handles, labels = axs[0].get_legend_handles_labels()

x = fig.legend(title_fontsize=14, loc='upper left', bbox_to_anchor=(0, 1.1), fontsize=14,
           ncol=2, labelspacing=1, handletextpad=0.4, markerscale=6,
           columnspacing=1, borderpad=0.2, frameon=False, handles=handles, labels=labels)


# Modify the line width in the legend handles
for line in x.get_lines():
    line.set_linewidth(3)
    line.set_alpha(1)
    line.set_linestyle('-')
    line.markersize = 70

plt.savefig(f"movielens_accuracy_fixed_windows_change_window_size_alpha_{str(get_integer(alpha))}.png")
plt.show()


