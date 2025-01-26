import ast
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from polars.dependencies import subprocess

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)



window_size_unit_list = ['1', '60', '60*24', '60*24*7', '60*24*7*4', '60*24*7*4*2', '60*24*7*4*3']

result_file_list = dict()
alpha = 99993
for window_size in window_size_unit_list:
    file_name = f"movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_alpha_{alpha}_time_unit_1 min*{window_size}_check_interval_1 hour.csv"
    # if file exists
    if not os.path.exists(file_name):
        subprocess.run(["python3", "accuracy_gender_per_item_1min.py", window_size])
        # Check again after running the script
        if not os.path.exists(file_name):
            raise Exception(f"Error: {file_name} was not created by the script.")
    df = pd.read_csv(file_name)
    print("window_size", window_size, df)
    result_file_list[window_size] = df



# draw the plot


fig, axs = plt.subplots(1, 7, figsize=(12, 1))
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
pair_colors = ["blue", "darkorange"]
for i in range(len(window_size_unit_list)):
    window_size = window_size_unit_list[i]
    df = result_file_list[window_size]
    x_list = np.arange(0, len(df))
    male_time_decay = df["male_time_decay"].tolist()
    female_time_decay = df["female_time_decay"].tolist()
    axs[axis_number[i]].plot(x_list, male_time_decay, linewidth=1, markersize=1.3, label='Male', linestyle='-', marker='o', color="blue")
    axs[axis_number[i]].plot(x_list, female_time_decay, linewidth=1, markersize=1.3, label='Female', linestyle='-', marker='o',
            color="darkorange")
    axs.flat[i].set_xticks([], [], rotation=0, fontsize=14)
    axs.flat[i].set_ylim(0.6, 0.9)
    for j in range(1, len(female_time_decay)):
        if abs(female_time_decay[j - 1] - female_time_decay[j]) > decline_threshold:
            axs[axis_number[i]].axvline(x=j, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
            # plt.text(i, y_margin, check_points[i].replace(" ", "\n"), color='red', fontsize=15,
            #          verticalalignment='bottom', horizontalalignment='center')
            decline_points.append(j)
            differences.append(female_time_decay[j - 1] - female_time_decay[j])
    print("decline_points", decline_points)


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
axs.flat[0].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[0].set_xlabel('(a) 1 second', fontsize=15, labelpad=3).set_position([0.4, -0.1])

# axs.flat[2].set_yscale('symlog', linthresh=0.22)
axs.flat[2].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[2].set_xlabel('(c) 1 week', fontsize=15, labelpad=3).set_position([0.4, -0.1])

# axs.flat[4].set_yscale('symlog', linthresh=0.35)
axs.flat[4].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[4].set_xlabel('(e) 1 month', fontsize=15, labelpad=3).set_position([0.4, -0.1])

axs.flat[1].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[1].set_xlabel('(b) 1 day', fontsize=15, labelpad=3).set_position([0.4, -0.1])

axs.flat[3].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[3].set_xlabel('(d) 2 weeks', fontsize=15, labelpad=3).set_position([0.4, -0.1])

# axs.flat[3].set_ylabel('coverage ratio (CR)', fontsize=16, labelpad=1)

axs.flat[5].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=15)
axs.flat[5].set_xlabel('(f) 3 months', fontsize=15, labelpad=3).set_position([0.4, -0.1])



# fig.text(0.5, -1, 'differen time windows',
#          ha='center', va='center', fontsize=14)
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=15, rotation='vertical')

# create a common legend
handles, labels = axs[0].get_legend_handles_labels()

plt.legend(title_fontsize=14, loc='upper left', bbox_to_anchor=(-7, 1.43), fontsize=15,
           ncol=2, labelspacing=1, handletextpad=0.4, markerscale=2.5,
           columnspacing=1, borderpad=0.2, frameon=False, handles=handles, labels=labels)

plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.05, wspace=0.38, hspace=0.35)
plt.tight_layout()
plt.savefig("movielens_accuracy_fixed_windows_change_sizes.png", bbox_inches='tight')
plt.show()


