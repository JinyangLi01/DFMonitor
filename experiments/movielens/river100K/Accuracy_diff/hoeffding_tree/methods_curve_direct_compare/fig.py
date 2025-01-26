import colorsys
import csv

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
from algorithm.dynamic_window import Accuracy_workload as dynamic_window_workload
from algorithm.per_item import Accuracy_workload as fixed_window_workload
import seaborn as sns
from matplotlib.gridspec import GridSpec
# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'  # Replace with your desired font name


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


#  all time:
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
print(data[:5])
alpha = 0.995

threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "1 hour"
# window_size_units = "24*7*4*2"
checking_interval = "7 day"
use_nanosecond = False



fig, axs = plt.subplots(2, 2, figsize=(5, 1.7), gridspec_kw={'width_ratios': [5, 2]})
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0, wspace=0.3, hspace=0.5)




plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])

curve_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']

colors_set1 = ["#1f77b4", "#76b7b2", "#17becf"]  # Blue, teal, light blue
colors_set2 = ["#ff7f0e", "#ffbb78", "#d62728"]  # Orange, light orange, red

curve_colors = sns.color_palette("magma", 3) + sns.color_palette("viridis", 3)
curve_colors = ["blue", "red", "hotpink", "DarkGrey", "Lime", "cyan"]

######################################## Dynamic fixed window ########################################



# Inset zoom
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(axs[0][0], width="30%", height="30%", loc="upper right")

df_fixed = pd.read_csv(f"fixed_window_{checking_interval}.csv")

x_list = np.arange(len(df_fixed))
axs[0][0].plot(x_list, df_fixed['male_time_decay'], linewidth=1.5, markersize=2.5, label = 'male fixed window',
            linestyle='-', marker='o',
            color=curve_colors[0])
axs[1][0].plot(x_list, df_fixed['female_time_decay'], linewidth=1.5, markersize=2.5, label = 'female fixed window',
            linestyle='-', marker='o',
            color=curve_colors[1])



######################################## Dynamic adaptive window ########################################

Tb = 24*7
Tin = 24

df_adaptive = pd.read_csv(f"adaptive_window_{checking_interval}.csv")
x_list = np.arange(len(df_adaptive))
axs[0][0].plot(x_list, df_adaptive["male_time_decay"], linewidth=1.5, markersize=2.5, label='male adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[2])
axs[1][0].plot(x_list, df_adaptive["female_time_decay"], linewidth=1.5, markersize=2.5, label='female adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[3])



######################################## traditional ########################################



window_size = "1W"
method_name = "hoeffding_classifier"
df_traditional = pd.read_csv(f"../traditional/movielens_compare_Accuracy_{method_name}_traditional_gender_{window_size}.csv")
print(df_traditional)
value_col_name = "calculated_value"




def plot_certain_time_window(df, value_col_name, window_size, axs):
    df_female = df[df["gender"] == 'F']
    # df_female = df_female[df_female[window_size].notna()]
    df_female = df_female[["datetime", value_col_name]]

    df_male = df[df["gender"] == 'M']
    # df_male = df_male[df_male[window_size].notna()]
    df_male = df_male[["datetime", value_col_name]]

    print("df_male: \n", df_male)
    print("\ndf_female: \n", df_female)

    x_list = np.arange(0, len(df_male))

    from datetime import datetime
    datetime = df_male['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y')).tolist()

    female_lst = df_female[value_col_name].dropna().tolist()
    male_lst = df_male[value_col_name].dropna().tolist()
    print(male_lst, female_lst)

    axs[0][0].plot(np.arange(len(male_lst)), male_lst, linewidth=1.5, markersize=2,
             label="male", linestyle='--', marker='o', color=curve_colors[4])
    axs[1][0].plot(np.arange(len(female_lst)), female_lst, linewidth=1.5, markersize=2,
             label='female', linestyle='--', marker='o', color=curve_colors[5])
    # axs[0][0].legend(loc='lower left',
    #            bbox_to_anchor=(-0.15, 0.08),  # Adjust this value (lower the second number)
    #            fontsize=12, ncol=1, labelspacing=0.2, handletextpad=0.5,
    #            markerscale=1, handlelength=1, columnspacing=0.6,
    #            borderpad=0.2, frameon=True)
    return male_lst, female_lst


male_lst, female_lst = plot_certain_time_window(df_traditional, value_col_name,'1W', axs)

# axs[0][0].set_ylabel('Accuracy', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs[0][0].set_ylim(0.6, 0.9)
axs[0][0].set_yscale('symlog', linthresh=8)
axs[0][0].set_title('(a) Male Accuracy', y=-0.15, pad=-10, fontsize=14)
axs[0][0].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=14)
axs[0][0].set_xticks(range(len(male_lst)), [""]*len(male_lst), rotation=0, fontsize=14)

axs[0][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[0][0].minorticks_on()


# axs[1][0].set_ylabel('accuracy', fontsize=14, labelpad=-1)
axs[1][0].set_ylim(0.6, 0.9)
axs[1][0].set_yscale('symlog', linthresh=8)
axs[1][0].set_title('(c) Female Accuracy', y=-0.15, pad=-10, fontsize=14)
axs[1][0].set_yticks([0.6, 0.7, 0.8, 0.9], ['0.6', '0.7', '0.8', '0.9'], fontsize=14)
axs[1][0].set_xticks([], [], rotation=0, fontsize=14)
axs[1][0].set_xticks(range(len(female_lst)), [""]*len(female_lst), rotation=0, fontsize=14)

axs[1][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][0].grid(True, axis="x", linestyle='--', alpha=0.6)

fig.savefig("curves.png", bbox_inches='tight')


######################################## roughness ########################################

def roughness2smoothness(roughness_FPR):
    print("Roughness: ", roughness_FPR)

    # Scale roughness to smoothness
    max_roughness = max(roughness_FPR.values())
    smoothness_scores = {key: 1 - (value / max_roughness) for key, value in roughness_FPR.items()}
    print("Smoothness Scores (Before Normalization):", smoothness_scores)

    # # Normalize smoothness scores to [0, 1]
    # max_smoothness = max(smoothness_scores.values())
    # min_smoothness = min(smoothness_scores.values())
    # normalized_smoothness = {key: score/max_smoothness
    #                          for key, score in smoothness_scores.items()}
    # print("Normalized Smoothness Scores:", normalized_smoothness)
    # print("\n")
    return smoothness_scores


smooth_male = {}
smooth_female = {}

with open("roughness_compas_FPR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness", "roughness", "check_points"])
    writer.writerow(["male_fixed_window", np.std(np.diff(df_fixed["male_time_decay"]) / np.diff(x_list))])
    smooth_male["male_fixed_window_smoothness"] = np.std(np.diff(df_fixed["male_time_decay"]) / np.diff(x_list))

    writer.writerow(["male_adaptive_window", np.std(np.diff(df_adaptive["male_time_decay"]) / np.diff(x_list))])
    smooth_male["male_adaptive_window_smoothness"] = np.std(np.diff(df_fixed["male_time_decay"]) / np.diff(x_list))

    tra = df_traditional["calculated_value"].tolist()
    tra = [x for x in tra if str(x) != 'nan']
    dydx = np.diff(tra) / np.diff(np.arange(0, len(tra)))
    sm = np.std(dydx)
    writer.writerow(["male_traditional", sm])
    smooth_male["male_traditional_smoothness"] = sm


print(smooth_male)

# smoothness_scores_male = [1/sd for sd in list(smooth_male.values())]  # Example data
# # Normalize based on the maximum value observed
# max_smoothness_male = max(smoothness_scores_male)
# smoothness_scores_normalized_male = [score / max_smoothness_male for score in smoothness_scores_male]
smoothness_scores_normalized_male = roughness2smoothness(smooth_male)
print("Normalized Smoothness Scores male fixed, adaptive, tradidtional:", smoothness_scores_normalized_male)


with open("roughness_compas_FPR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # do the same for females
    writer.writerow(["female_fixed_window", np.std(np.diff(df_fixed["female_time_decay"]) / np.diff(x_list))])
    smooth_female["female_fixed_window_smoothness"] = np.std(np.diff(df_fixed["female_time_decay"]) / np.diff(x_list))
    writer.writerow(["female_adaptive_window", np.std(np.diff(df_adaptive["female_time_decay"]) / np.diff(x_list))])
    smooth_female["female_adaptive_window_smoothness"] = np.std(np.diff(df_adaptive["female_time_decay"]) / np.diff(x_list))
    tra = df_traditional["calculated_value"].tolist()
    tra = [x for x in tra if str(x) != 'nan']
    dydx = np.diff(tra) / np.diff(np.arange(0, len(tra)))
    sm = np.std(dydx)
    writer.writerow(["female_traditional_smoothness", sm])
    smooth_female["female_traditional_smoothness"] = sm


print(smooth_female)

# smoothness_scores_female = [1/sd for sd in list(smooth_female.values())]  # Example data
# # Normalize based on the maximum value observed
# max_smooth_female = max(smoothness_scores_female)
# smoothness_scores_normalized_female = [score / max_smooth_female for score in smoothness_scores_female]
smoothness_scores_normalized_female = roughness2smoothness(smooth_female)
print("Normalized smoothness Scores female fixed, adaptive, traditional:", smoothness_scores_normalized_female)


bar_colors = ["blue", "hotpink",  "Lime",  "red",  "DarkGrey", "cyan"]



# Male Smoothness Bar Chart
axs[0][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_male.values(), color=bar_colors[:3], width=0.4,
              label = ["Male Fixed", "Male Adaptive", "Male Traditional"])
axs[0][1].set_title("(b) Male Smoothness Score", y=-0.15, pad=-10, fontsize=14).set_position([0.32, -0.14])


axs[0][1].set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"], fontsize=14)
axs[0][1].grid(True, linestyle='--', alpha=0.6)
axs[0][1].set_xticks([], [], rotation=0, fontsize=14)
# axs[0][1].set_ylabel('Normalized Smoothness Score', fontsize=14, labelpad=-1).set_position([0.4, -0.1])


# Female Smoothness Bar Chart
axs[1][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_female.values(), color=bar_colors[3:], width=0.4,
              label = ["Female Fixed", "Female Adaptive", "Female Traditional"])
axs[1][1].set_title("(d) Female Smoothness Score", y=-0.15, pad=-10, fontsize=14).set_position([0.32, -0.14])
axs[1][1].set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"], fontsize=14)
axs[1][1].grid(True, linestyle='--', alpha=0.6)
axs[1][1].set_xticks([], [], rotation=0, fontsize=14)


handles, labels = axs[0][1].get_legend_handles_labels()
handles1, labels1 = axs[1][1].get_legend_handles_labels()
handles = handles + handles1
labels = labels + labels1
desired_order = [0, 3,  1,4,  2, 5]  # Reorder as needed

# Reorder handles and labels
reordered_handles = [handles[i] for i in desired_order]
reordered_labels = [labels[i] for i in desired_order]

axs[0][0].legend(reordered_handles, reordered_labels, title_fontsize=14, loc='upper left',
                 bbox_to_anchor=(-0.1, 1.8),
                 fontsize=13, ncol=3, labelspacing=0.2, handletextpad=0.4, markerscale=1,
                 columnspacing=0.8, borderpad=0.2, frameon=False)


fig.savefig("curves_smoothness_score.png", bbox_inches='tight')
plt.show()




