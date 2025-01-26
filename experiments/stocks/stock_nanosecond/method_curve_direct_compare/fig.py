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
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
# Set the font size for labels, legends, and titles


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


threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "1 hour"
# window_size_units = "24*7*4*2"
checking_interval =  "100000 nanosecond"
use_nanosecond = False



fig, axs = plt.subplots(3, 2, figsize=(5.6, 2.7), gridspec_kw={'width_ratios': [5, 2]})
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0, wspace=0.3, hspace=0.38)




plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])

curve_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']

colors_set1 = ["#1f77b4", "#76b7b2", "#17becf"]  # Blue, teal, light blue
colors_set2 = ["#ff7f0e", "#ffbb78", "#d62728"]  # Orange, light orange, red

curve_colors = sns.color_palette("magma", 3) + sns.color_palette("viridis", 3)
curve_colors = ["blue", "red", "green",  "hotpink", "DarkGrey", "darkorchid",  "Lime", "cyan", "gold",]

######################################## Dynamic fixed window ########################################



# Inset zoom
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(axs[0][0], width="30%", height="30%", loc="upper right")

df_fixed = pd.read_csv(f"fixed_window_{checking_interval}.csv")

x_list = np.arange(len(df_fixed))
axs[0][0].plot(x_list, df_fixed['Technology_time_decay'], linewidth=1.0, markersize=1.0, label = 'Technology',
            linestyle='-', marker='o',
            color=curve_colors[0])
axs[1][0].plot(x_list, df_fixed['ConsumerCyclical_time_decay'], linewidth=1.0, markersize=1.0, label = 'Consumer Cyclical',
            linestyle='-', marker='o',
            color=curve_colors[1])
axs[2][0].plot(x_list, df_fixed['CommunicationServices_time_decay'], linewidth=1.0, markersize=1.0, label = "Communication Services",
            linestyle='-', marker='o',
            color=curve_colors[2])



######################################## Dynamic adaptive window ########################################

Tb = 24*7
Tin = 24

df_adaptive = pd.read_csv(f"adaptive_window_{checking_interval}.csv")
x_list_adaptive = np.arange(len(df_adaptive))
axs[0][0].plot(x_list_adaptive, df_adaptive["Technology_time_decay"], linewidth=1.0, markersize=1.0, label='Technology',
            linestyle='-', marker='o',
            color=curve_colors[3])
axs[1][0].plot(x_list_adaptive, df_adaptive["ConsumerCyclical_time_decay"], linewidth=1.0, markersize=1.0, label='Consumer Cyclical',
            linestyle='-', marker='o',
            color=curve_colors[4])
axs[2][0].plot(x_list_adaptive, df_adaptive['CommunicationServices_time_decay'], linewidth=1.0, markersize=1.0, label = "Communication Services",
            linestyle='-', marker='o',
            color=curve_colors[5])


######################################## traditional ########################################



window_size = "500ms"
df_traditional = pd.read_csv(f"../traditional_method/traditional_accuracy_time_window_{window_size}.csv")
print(df_traditional)
value_col_name = "accuracy"
df_traditional['ts_event'] = pd.to_datetime(df_traditional['ts_event'], format='mixed')




# Normalize the traditional data to match the same scale as fixed and adaptive windows
min_fixed = min(df_fixed["Technology_time_decay"].min(), df_fixed["ConsumerCyclical_time_decay"].min(), df_fixed["CommunicationServices_time_decay"].min())
max_fixed = max(df_fixed["Technology_time_decay"].max(), df_fixed["ConsumerCyclical_time_decay"].max(), df_fixed["CommunicationServices_time_decay"].max())

# Apply normalization to traditional accuracy values
df_traditional["normalized_accuracy"] = (df_traditional["accuracy"] - min_fixed) / (max_fixed - min_fixed)


# Update the function to handle traditional curves
def plot_certain_time_window(df, value_col_name, axs):
    df_tech = df[df["sector"] == 'Technology']
    df_tech = df_tech[["ts_event", value_col_name]]

    df_consumer = df[df["sector"] == 'Consumer Cyclical']
    df_consumer = df_consumer[["ts_event", value_col_name]]

    df_communication = df[df["sector"] == 'Communication Services']
    df_communication = df_communication[["ts_event", value_col_name]]

    # Extract lists
    tech_lst = df_tech[value_col_name].dropna().tolist()
    consumer_lst = df_consumer[value_col_name].dropna().tolist()
    communication_lst = df_communication[value_col_name].dropna().tolist()

    # Ensure x-axis alignment
    x_list_traditional = np.arange(len(tech_lst))

    # Plot traditional curves
    axs[0][0].plot(x_list_traditional, tech_lst, linewidth=1.0, markersize=2,
                   label="Technology (Traditional)", linestyle='--', marker='o', color=curve_colors[6])
    axs[1][0].plot(x_list_traditional, consumer_lst, linewidth=1.0, markersize=2,
                   label='Consumer Cyclical (Traditional)', linestyle='--', marker='o', color=curve_colors[7])
    axs[2][0].plot(x_list_traditional, communication_lst, linewidth=1.0, markersize=2,
                   label='Communication Services (Traditional)', linestyle='--', marker='o', color=curve_colors[8])

    return tech_lst, consumer_lst, communication_lst

# Plot traditional curves
tech_lst, consumer_lst, communication_lst = plot_certain_time_window(df_traditional, "normalized_accuracy", axs)

#
# def plot_certain_time_window(df, value_col_name, window_size, axs):
#     df_tech = df[df["sector"] == 'Technology']
#     df_tech = df_tech[["ts_event", value_col_name]]
#
#     df_Consumer = df[df["sector"] == 'Consumer Cyclical']
#     # df_male = df_male[df_male[window_size].notna()]
#     df_Consumer = df_Consumer[["ts_event", value_col_name]]
#
#     df_Communication = df[df["sector"] == 'Communication Services']
#     df_Communication = df_Communication[["ts_event", value_col_name]]
#
#
#     # print("df_tech: \n", df_tech)
#     # print("\ndf_Consumer: \n", df_Consumer)
#
#     x_list = np.arange(0, len(df_Consumer))
#
#     from datetime import datetime
# #    datetime = df_tech['ts_event'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y')).tolist()
#     datetime = df_tech['ts_event'].apply(lambda x: x.strftime('%m/%d/%Y')).tolist()
#
#     tech_lst = df_tech[value_col_name].dropna().tolist()
#     consumer_lst = df_Consumer[value_col_name].dropna().tolist()
#     communication_lst = df_Communication[value_col_name].dropna().tolist()
#
#     # df = pd.DataFrame({'datetime': datetime, 'Technology': tech_lst, 'Consumer Cyclical': consumer_lst, 'Communication Services': communication_lst})
#
#
#     # Interpolate the second curve to match x1's points
#     interp_func = interp1d(np.arange(len(tech_lst)), tech_lst, kind='linear', fill_value="extrapolate")
#     tech_interp = interp_func(x_list_adaptive)
#
#
#     axs[0][0].plot(x_list_adaptive, tech_interp, linewidth=1.0, markersize=2,
#              label="Technology", linestyle='--', marker='o', color=curve_colors[6])
#
#     interp_func = interp1d(np.arange(len(consumer_lst)), consumer_lst, kind='linear', fill_value="extrapolate")
#     consumer_interp = interp_func(x_list_adaptive)
#
#
#     axs[1][0].plot(x_list_adaptive, consumer_interp, linewidth=1.0, markersize=2,
#              label='Consumer Cyclical', linestyle='--', marker='o', color=curve_colors[7])
#
#     interp_func = interp1d(np.arange(len(communication_lst)), communication_lst, kind='linear', fill_value="extrapolate")
#     communication_interp = interp_func(x_list_adaptive)
#
#
#     axs[2][0].plot(x_list_adaptive, communication_interp, linewidth=1.0, markersize=2,
#                    label='Communication Service', linestyle='--', marker='o', color=curve_colors[8])
#     # axs[0][0].legend(loc='lower left',
#     #            bbox_to_anchor=(-0.15, 0.08),  # Adjust this value (lower the second number)
#     #            fontsize=12, ncol=1, labelspacing=0.2, handletextpad=0.5,
#     #            markerscale=1, handlelength=1, columnspacing=0.6,
#     #            borderpad=0.2, frameon=True)
#     return tech_lst, consumer_lst, communication_lst
#
#
# tech_lst, consumer_lst, communication_lst = plot_certain_time_window(df_traditional, value_col_name,'500ms', axs)

# axs[0][0].set_ylabel('Accuracy', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs[0][0].set_ylim(0.6, 0.7)
axs[0][0].set_yscale('symlog', linthresh=8)
axs[0][0].set_title('(a) Tech Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[0][0].set_yticks([0.6, 0.7], ['0.6', '0.7'], fontsize=14)
# axs[0][0].set_xticks(range(len(tech_lst)), [""]*len(tech_lst), rotation=0, fontsize=14)

axs[0][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[0][0].minorticks_on()


# axs[1][0].set_ylabel('accuracy', fontsize=14, labelpad=-1)
axs[1][0].set_ylim(0.58, 0.7)
axs[1][0].set_yscale('symlog', linthresh=8)
axs[1][0].set_title('(c) Con. Cyclical Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[1][0].set_yticks([0.6, 0.7], ['0.6', '0.7'], fontsize=14)
axs[1][0].set_xticks([], [], rotation=0, fontsize=14)
# axs[1][0].set_xticks(range(len(consumer_lst)), [""]*len(consumer_lst), rotation=0, fontsize=14)

axs[1][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[1][0].minorticks_on()


axs[2][0].set_ylim(0.5, 0.7)
axs[2][0].set_yscale('symlog', linthresh=8)
axs[2][0].set_title('(e) Comm. Serv. Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[2][0].set_yticks([0.5, 0.6, 0.7], ['0.5', '0.6', '0.7'], fontsize=14)
axs[2][0].set_xticks([], [], rotation=0, fontsize=14)
# axs[2][0].set_xticks(range(len(communication_lst)), [""]*len(communication_lst), rotation=0, fontsize=14)

axs[2][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[2][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[2][0].minorticks_on()

fig.savefig("curves.png", bbox_inches='tight')


######################################## roughness ########################################


smooth_tech = {}
smooth_consumer_cyclical = {}
smooth_communication_services = {}


with open("roughness_compas_FPR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness", "roughness", "check_points"])
    writer.writerow(["tech_fixed_window", np.std(np.diff(df_fixed["Technology_time_decay"]) / np.diff(x_list))])
    smooth_tech["tech_fixed_window_smoothness"] = np.std(np.diff(df_fixed["Technology_time_decay"]) / np.diff(x_list))
    
    writer.writerow(["tech_adaptive_window", np.std(np.diff(df_adaptive["Technology_time_decay"]) / np.diff(x_list))])
    smooth_tech["tech_adaptive_window_smoothness"] = np.std(np.diff(df_fixed["Technology_time_decay"]) / np.diff(x_list))

    tra = df_traditional["accuracy"].tolist()
    tra = [x for x in tra if str(x) != 'nan']
    dydx = np.diff(tra) / np.diff(np.arange(0, len(tra)))
    sm = np.std(dydx)
    writer.writerow(["tech_traditional", sm])
    smooth_tech["tech_traditional_smoothness"] = sm


print(smooth_tech)

smoothness_scores_tech = [1/sd for sd in list(smooth_tech.values())]  # Example data
# Normalize based on the maximum value observed
max_smoothness_tech = max(smoothness_scores_tech)
smoothness_scores_normalized_tech = [score / max_smoothness_tech for score in smoothness_scores_tech]
print("Normalized Smoothness Scores male fixed, adaptive, traditional:", smoothness_scores_normalized_tech)


with open("roughness_compas_FPR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # do the same for females
    writer.writerow(["ConsumerCyclical_fixed_window", np.std(np.diff(df_fixed["ConsumerCyclical_time_decay"]) / np.diff(x_list))])
    smooth_consumer_cyclical["ConsumerCyclical_fixed_window_smoothness"] = np.std(np.diff(df_fixed["ConsumerCyclical_time_decay"]) / np.diff(x_list))
    writer.writerow(["ConsumerCyclical_adaptive_window", np.std(np.diff(df_adaptive["ConsumerCyclical_time_decay"]) / np.diff(x_list))])
    smooth_consumer_cyclical["ConsumerCyclical_adaptive_window_smoothness"] = np.std(np.diff(df_adaptive["ConsumerCyclical_time_decay"]) / np.diff(x_list))
    tra = df_traditional["accuracy"].tolist()
    tra = [x for x in tra if str(x) != 'nan']
    dydx = np.diff(tra) / np.diff(np.arange(0, len(tra)))
    sm = np.std(dydx)
    writer.writerow(["ConsumerCyclical_traditional_smoothness", sm])
    smooth_consumer_cyclical["ConsumerCyclical_traditional_smoothness"] = sm


print(smooth_consumer_cyclical)


smoothness_scores_consumer_cyclical = [1/sd for sd in list(smooth_consumer_cyclical.values())]  # Example data
# Normalize based on the maximum value observed
max_smooth_consumer_cyclical = max(smoothness_scores_consumer_cyclical)
smoothness_scores_normalized_consumer_cyclical = [score / max_smooth_consumer_cyclical for score in smoothness_scores_consumer_cyclical]
print("Normalized smoothness Scores consumer_cyclical fixed, adaptive, traditional:", smoothness_scores_normalized_consumer_cyclical)


with open("roughness_compas_FPR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["CommunicationServices_fixed_window", np.std(np.diff(df_fixed["CommunicationServices_time_decay"]) / np.diff(x_list))])
    smooth_communication_services["CommunicationServices_fixed_window_smoothness"] = np.std(np.diff(df_fixed["CommunicationServices_time_decay"]) / np.diff(x_list))
    writer.writerow(["CommunicationServices_adaptive_window", np.std(np.diff(df_adaptive["CommunicationServices_time_decay"]) / np.diff(x_list))])
    smooth_communication_services["CommunicationServices_adaptive_window_smoothness"] = np.std(np.diff(df_adaptive["CommunicationServices_time_decay"]) / np.diff(x_list))
    tra = df_traditional["accuracy"].tolist()
    tra = [x for x in tra if str(x) != 'nan']
    dydx = np.diff(tra) / np.diff(np.arange(0, len(tra)))
    sm = np.std(dydx)
    writer.writerow(["CommunicationServices_traditional_smoothness", sm])
    smooth_communication_services["CommunicationServices_traditional_smoothness"] = sm

print(smooth_communication_services)

smoothness_scores_communication_services = [1/sd for sd in list(smooth_communication_services.values())]  # Example data
# Normalize based on the maximum value observed
max_smooth_communication_services = max(smoothness_scores_communication_services)
smoothness_scores_normalized_communication_services = [score / max_smooth_communication_services for score in smoothness_scores_communication_services]
print("Normalized smoothness Scores communication_services fixed, adaptive, traditional:", smoothness_scores_normalized_communication_services)




bar_colors = ["blue", "hotpink",  "Lime",  "red",  "DarkGrey", "cyan",  "green", "darkorchid","gold", ]



# Male Smoothness Bar Chart
axs[0][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_tech, color=bar_colors[:3], width=0.4,
              label = ["Tech Fixed", "Tech Adaptive", "Tech Traditional"])
axs[0][1].set_title("(b) Tech Smoothness", y=-0.15, pad=-8, fontsize=14).set_position([0.32, -0.14])


axs[0][1].set_yscale("symlog", linthresh=0.1)
axs[0][1].set_yticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1.0"], fontsize=14)
axs[0][1].grid(True, linestyle='--', alpha=0.6)
axs[0][1].set_xticks([], [], rotation=0, fontsize=14)
# axs[0][1].set_ylabel('Normalized Smoothness Score', fontsize=14, labelpad=-1).set_position([0.4, -0.1])


# Female Smoothness Bar Chart
axs[1][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_consumer_cyclical, color=bar_colors[3:6], width=0.4,
              label = ["Con. Cyclical Fixed", "Con. Cyclical Adaptive",
                       "Con. Cyclical Traditional"])
axs[1][1].set_title("(d) Cons. Cyclical Smoothness", y=-0.15, pad=-8, fontsize=14).set_position([0.32, -0.14])
axs[1][1].set_yscale("symlog", linthresh=0.1)
axs[1][1].set_yticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1.0"], fontsize=14)
axs[1][1].grid(True, linestyle='--', alpha=0.6)
axs[1][1].set_xticks([], [], rotation=0, fontsize=14)



axs[2][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
              smoothness_scores_normalized_communication_services, color=bar_colors[6:], width=0.4,
                  label = ["Comm. Serv. Fixed", "Comm. Serv. Adaptive",
                           "Comm. Serv. Traditional"])
axs[2][1].set_title("(f) Comm. Serv. Smoothness", y=-0.15, pad=-8, fontsize=14).set_position([0.32, -0.14])
axs[2][1].set_yscale("symlog", linthresh=0.1)
axs[2][1].set_yticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1.0"], fontsize=14)
axs[2][1].grid(True, linestyle='--', alpha=0.6)
axs[2][1].set_xticks([], [], rotation=0, fontsize=14)




handles, labels = axs[0][1].get_legend_handles_labels()
handles1, labels1 = axs[1][1].get_legend_handles_labels()
handles2, labels2 = axs[2][1].get_legend_handles_labels()
handles = handles + handles1 + handles2
labels = labels + labels1 + labels2


desired_order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
# Reorder handles and labels
reordered_handles = [handles[i] for i in desired_order]
reordered_labels = [labels[i] for i in desired_order]

axs[0][0].legend(handles=reordered_handles, labels=reordered_labels, title_fontsize=14, loc='upper left',
                 bbox_to_anchor=(-0.13, 2), handlelength=1.5,
                 fontsize=13, ncol=3, labelspacing=0.1, handletextpad=0.2, markerscale=0.1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)


fig.savefig("curves_smoothness_score.png", bbox_inches='tight')
plt.show()




