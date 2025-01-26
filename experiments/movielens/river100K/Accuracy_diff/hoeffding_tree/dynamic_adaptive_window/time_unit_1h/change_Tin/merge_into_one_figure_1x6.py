import ast
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from polars.dependencies import subprocess
from scipy.signal import find_peaks
import matplotlib.transforms as mtransforms
# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)



Tin_list = [1, 12, 24, 24*2, 24*3, 24*7]
Tb = 24*7
result_file_list = dict()
alpha = 993
time_unit = "1 hour"
checking_interval = "1 hour"
for Tin in Tin_list:
    file_name = f"movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_alpha_{alpha}_time_unit_1 hour_check_interval_{checking_interval}_Tin_{Tin}_Tb_{Tb}.csv"
    # if file exists
    if not os.path.exists(file_name):
        subprocess.run(["python3", "accuracy_gender_per_item_1hour.py", str(Tin)])
        # Check again after running the script
        if not os.path.exists(file_name):
            raise Exception(f"Error: {file_name} was not created by the script.")
    df = pd.read_csv(file_name)
    print("Tin", Tin, df)
    result_file_list[Tin] = df



# draw the plot

fig, axs = plt.subplots(1, 6, figsize=(11, 0.85))
plt.subplots_adjust(top=0.84, bottom=0, hspace=0, wspace=0.33, left=0.0, right=1)
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



df = result_file_list[Tin_list[0]]
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



pair_colors = ["blue", "darkorange"]
for i in range(len(Tin_list)):
    Tin = Tin_list[i]
    df = result_file_list[Tin]
    x_list = np.arange(0, len(df))
    df["check_points"] = pd.to_datetime(df["check_points"])
    check_points = df["check_points"].tolist()
    male_time_decay = df["male_time_decay"].tolist()
    female_time_decay = df["female_time_decay"].tolist()
    axs[axis_number[i]].plot(x_list, male_time_decay, linewidth=1, markersize=1.3, label='Male', linestyle='-', marker='o', color="blue")
    axs[axis_number[i]].plot(x_list, female_time_decay, linewidth=1, markersize=1.3, label='Female', linestyle='-', marker='o',
            color="darkorange")
    axs.flat[i].set_ylim(0.6, 0.9)

    axs[axis_number[i]].axvline(x=first_drop_index, color='black', linestyle='--', linewidth=1.5)
    axs[axis_number[i]].axvline(x=second_drop_index, color='black', linestyle='--', linewidth=1.5)
    xticks_idx = [first_drop_index, second_drop_index]
    xtick_label = [check_points[first_drop_index].strftime("%m/%d/%Y"),
                   check_points[second_drop_index].strftime("%m/%d/%Y")]
    axs[axis_number[i]].set_xticks(xticks_idx, xtick_label, rotation=10, fontsize=14)
    axs[axis_number[i]].tick_params(axis='x', which='major', pad=-3)
    axs[axis_number[i]].tick_params(axis='y', which='major', pad=0)
    first_label = axs[axis_number[i]].get_xticklabels()[0]
    first_label.set_transform(first_label.get_transform() + mtransforms.Affine2D().translate(-13, 3))  # Shift 5 points left
    first_label = axs[axis_number[i]].get_xticklabels()[1]
    first_label.set_transform(first_label.get_transform() + mtransforms.Affine2D().translate(10, 0))  # Shift 5 points left
    axs[axis_number[i]].grid(True)
    axs[axis_number[i]].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
    xlabel = f"({chr(97 + i)}) $T_{{in}} = {Tin}$"
    axs[axis_number[i]].set_xlabel(xlabel, fontsize=15, labelpad=-2).set_position([0.4, -0.1])
    # axs[axis_number[i]].set_x(labels[i].get_position()[0] - 3)


# create a common legend
handles, labels = axs[0].get_legend_handles_labels()

x = fig.legend(title_fontsize=14, loc='upper left', bbox_to_anchor=(-0.02, 1.28), fontsize=15,
           ncol=2, labelspacing=1, handletextpad=0.4, markerscale=6,
           columnspacing=1, borderpad=0.2, frameon=False, handles=handles, labels=labels)


# Modify the line width in the legend handles
for line in x.get_lines():
    line.set_linewidth(3)
    line.set_alpha(1)
    line.set_linestyle('-')
    line.markersize = 70



# plt.tight_layout()
plt.savefig("movielens_accuracy_adaptive_windows_change_Tin.png", bbox_inches='tight')
plt.show()


