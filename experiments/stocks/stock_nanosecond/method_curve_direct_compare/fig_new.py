import colorsys
import csv
from matplotlib.ticker import MaxNLocator
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
checking_interval =  "200 millisecond"
use_nanosecond = False



fig, axs = plt.subplots(3, 2, figsize=(5.6, 2.7), gridspec_kw={'width_ratios': [5, 2]})
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0, wspace=0.22, hspace=0.55)




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
df_fixed['check_points'] = pd.to_datetime(df_fixed['check_points'])  # Ensure datetime format


warm_up_time = pd.Timestamp('2024-10-15 14:00:05.8', tz='UTC')

df_fixed = df_fixed[df_fixed["check_points"] >= warm_up_time]



axs[0][0].plot(df_fixed['check_points'], df_fixed['Technology_time_decay'], linewidth=2.0, markersize=1.0, label = 'Technology',
            linestyle='-', marker='o',
            color=curve_colors[0])
axs[1][0].plot(df_fixed['check_points'], df_fixed['ConsumerCyclical_time_decay'], linewidth=2.0, markersize=1.0, label = 'Consumer Cyclical',
            linestyle='-', marker='o',
            color=curve_colors[1])
axs[2][0].plot(df_fixed['check_points'], df_fixed['CommunicationServices_time_decay'], linewidth=2.0, markersize=1.0, label = "Communication Services",
            linestyle='-', marker='o',
            color=curve_colors[2])



######################################## Dynamic adaptive window ########################################

Tb = 24*7
Tin = 24

df_adaptive = pd.read_csv(f"adaptive_window_{checking_interval}.csv")
df_adaptive['check_points'] = pd.to_datetime(df_adaptive['check_points'])  # Ensure datetime format


df_adaptive = df_adaptive[df_adaptive["check_points"] >= warm_up_time]


axs[0][0].plot(df_adaptive["check_points"], df_adaptive["Technology_time_decay"], linewidth=2.0, markersize=1.0, label='Technology',
            linestyle='-', marker='o',
            color=curve_colors[3])
axs[1][0].plot(df_adaptive["check_points"], df_adaptive["ConsumerCyclical_time_decay"], linewidth=2.0, markersize=1.0, label='Consumer Cyclical',
            linestyle='-', marker='o',
            color=curve_colors[4])
axs[2][0].plot(df_adaptive["check_points"], df_adaptive['CommunicationServices_time_decay'], linewidth=2.0, markersize=1.0, label = "Communication Services",
            linestyle='-', marker='o',
            color=curve_colors[5])


######################################## traditional ########################################



window_size = "500ms"
df_traditional = pd.read_csv(f"../traditional_method/traditional_accuracy_time_window_{window_size}.csv")
print(df_traditional)
value_col_name = "accuracy"
df_traditional['ts_event'] = pd.to_datetime(df_traditional['ts_event'], format='mixed')

df_traditional = df_traditional[df_traditional["ts_event"] >= warm_up_time]

def plot_certain_time_window(df, value_col_name, window_size, axs, df_fixed):
    df_tech = df[df["sector"] == 'Technology']
    df_tech = df_tech[["ts_event", value_col_name]]

    df_Consumer = df[df["sector"] == 'Consumer Cyclical']
    # df_male = df_male[df_male[window_size].notna()]
    df_Consumer = df_Consumer[["ts_event", value_col_name]]

    df_Communication = df[df["sector"] == 'Communication Services']
    df_Communication = df_Communication[["ts_event", value_col_name]]

    # Convert timestamps to datetime format
    df_tech['ts_event'] = pd.to_datetime(df_tech['ts_event'], errors='coerce')
    df_Consumer['ts_event'] = pd.to_datetime(df_Consumer['ts_event'], errors='coerce')
    df_Communication['ts_event'] = pd.to_datetime(df_Communication['ts_event'], errors='coerce')

    # Plot Technology data
    axs[0][0].plot(df_tech["ts_event"], df_tech[value_col_name], linewidth=1.8,
                   markersize=1.0, label='Technology', linestyle='-', marker='o', color=curve_colors[6])
    axs[1][0].plot(df_Consumer["ts_event"], df_Consumer[value_col_name], linewidth=1.8,
                     markersize=1.0, label='Consumer Cyclical', linestyle='-', marker='o', color=curve_colors[7])
    axs[2][0].plot(df_Communication["ts_event"], df_Communication[value_col_name], linewidth=1.8,
                        markersize=1.0, label='Communication Services', linestyle='-', marker='o', color=curve_colors[8])
    return df_tech, df_Consumer, df_Communication

df_tech, df_Consumer, df_Communication = plot_certain_time_window(df_traditional, value_col_name, window_size, axs, df_fixed)

# axs[0][0].set_ylabel('Accuracy', fontsize=14, labelpad=-1).set_position([0.4, -0.1])
axs[0][0].set_ylim(0.6, 0.7)
axs[0][0].set_yscale('symlog', linthresh=8)
axs[0][0].set_title('(a) Tech Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[0][0].set_yticks([0.6, 0.7], ['0.6', '0.7'], fontsize=14)
# axs[0][0].set_xticks(range(len(df_tech)), [""]*len(df_tech), rotation=0, fontsize=14)

axs[0][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[0][0].minorticks_on()
axs[0][0].xaxis.set_major_locator(MaxNLocator(nbins=20))
axs[0][0].set_xticklabels([])

# axs[1][0].set_ylabel('accuracy', fontsize=14, labelpad=-1)
axs[1][0].set_ylim(0.58, 0.7)
axs[1][0].set_yscale('symlog', linthresh=8)
axs[1][0].set_title('(c) Con. Cyclical Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[1][0].set_yticks([0.6, 0.7], ['0.6', '0.7'], fontsize=14)
axs[1][0].set_xticklabels([])
axs[1][0].xaxis.set_major_locator(MaxNLocator(nbins=20))
axs[1][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[1][0].minorticks_on()


axs[2][0].set_ylim(0.5, 0.7)
axs[2][0].set_yscale('symlog', linthresh=8)
axs[2][0].set_title('(e) Comm. Serv. Accuracy', y=-0.15, pad=-8, fontsize=14)
axs[2][0].set_yticks([0.5, 0.6, 0.7], ['0.5', '0.6', '0.7'], fontsize=14)
axs[2][0].set_xticklabels([])
axs[2][0].xaxis.set_major_locator(MaxNLocator(nbins=20))
axs[2][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[2][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[2][0].minorticks_on()

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



smooth_tech = {}
smooth_consumer_cyclical = {}
smooth_communication_services = {}

# Ensure time intervals are calculated correctly
time_intervals_fixed = (df_fixed['check_points'] - df_fixed['check_points'].iloc[0]).dt.total_seconds()

with open("roughness_compas_FPR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness", "roughness", "check_points"])

    # Technology Fixed Window Smoothness
    dydx_tech_fixed = np.diff(df_fixed["Technology_time_decay"]) / np.diff(time_intervals_fixed)
    smooth_tech["tech_fixed_window_smoothness"] = np.std(dydx_tech_fixed)
    writer.writerow(["tech_fixed_window", smooth_tech["tech_fixed_window_smoothness"]])

    # Technology Adaptive Window Smoothness
    time_intervals_adaptive = (df_adaptive['check_points'] - df_adaptive['check_points'].iloc[0]).dt.total_seconds()
    dydx_tech_adaptive = np.diff(df_adaptive["Technology_time_decay"]) / np.diff(time_intervals_adaptive)
    smooth_tech["tech_adaptive_window_smoothness"] = np.std(dydx_tech_adaptive)
    writer.writerow(["tech_adaptive_window", smooth_tech["tech_adaptive_window_smoothness"]])

    # Technology Traditional Smoothness
    tra = df_traditional["accuracy"].dropna().tolist()
    time_intervals_traditional = np.arange(0, len(tra))  # Assuming traditional is indexed by order
    dydx_tech_traditional = np.diff(tra) / np.diff(time_intervals_traditional)
    smooth_tech["tech_traditional_smoothness"] = np.std(dydx_tech_traditional)
    writer.writerow(["tech_traditional", smooth_tech["tech_traditional_smoothness"]])

print(smooth_tech)


smoothness_scores_normalized_tech = roughness2smoothness(smooth_tech)

# smoothness_scores_tech = [1 / sd for sd in list(smooth_tech.values())]  # Calculate inverse for smoothness
# max_smoothness_tech = max(smoothness_scores_tech)
# smoothness_scores_normalized_tech = [score / max_smoothness_tech for score in smoothness_scores_tech]
print("Normalized Smoothness Scores Tech:", smoothness_scores_normalized_tech)




# Repeat for Consumer Cyclical
with open("roughness_compas_FPR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    # Consumer Cyclical Fixed Window Smoothness
    dydx_consumer_fixed = np.diff(df_fixed["ConsumerCyclical_time_decay"]) / np.diff(time_intervals_fixed)
    smooth_consumer_cyclical["ConsumerCyclical_fixed_window_smoothness"] = np.std(dydx_consumer_fixed)
    writer.writerow(
        ["ConsumerCyclical_fixed_window", smooth_consumer_cyclical["ConsumerCyclical_fixed_window_smoothness"]])

    # Consumer Cyclical Adaptive Window Smoothness
    dydx_consumer_adaptive = np.diff(df_adaptive["ConsumerCyclical_time_decay"]) / np.diff(time_intervals_adaptive)
    smooth_consumer_cyclical["ConsumerCyclical_adaptive_window_smoothness"] = np.std(dydx_consumer_adaptive)
    writer.writerow(
        ["ConsumerCyclical_adaptive_window", smooth_consumer_cyclical["ConsumerCyclical_adaptive_window_smoothness"]])

    # Consumer Cyclical Traditional Smoothness
    tra = df_traditional["accuracy"].dropna().tolist()
    dydx_consumer_traditional = np.diff(tra) / np.diff(time_intervals_traditional)
    smooth_consumer_cyclical["ConsumerCyclical_traditional_smoothness"] = np.std(dydx_consumer_traditional)
    writer.writerow(
        ["ConsumerCyclical_traditional", smooth_consumer_cyclical["ConsumerCyclical_traditional_smoothness"]])

print(smooth_consumer_cyclical)

# smoothness_scores_consumer_cyclical = [1 / sd for sd in list(smooth_consumer_cyclical.values())]
# max_smooth_consumer_cyclical = max(smoothness_scores_consumer_cyclical)
# smoothness_scores_normalized_consumer_cyclical = [score / max_smooth_consumer_cyclical for score in
#                                                   smoothness_scores_consumer_cyclical]
smoothness_scores_normalized_consumer_cyclical = roughness2smoothness(smooth_consumer_cyclical)
print("Normalized Smoothness Scores Consumer Cyclical:", smoothness_scores_normalized_consumer_cyclical)

# Repeat for Communication Services
with open("roughness_compas_FPR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    # Communication Services Fixed Window Smoothness
    dydx_comm_fixed = np.diff(df_fixed["CommunicationServices_time_decay"]) / np.diff(time_intervals_fixed)
    smooth_communication_services["CommunicationServices_fixed_window_smoothness"] = np.std(dydx_comm_fixed)
    writer.writerow(["CommunicationServices_fixed_window",
                     smooth_communication_services["CommunicationServices_fixed_window_smoothness"]])

    # Communication Services Adaptive Window Smoothness
    dydx_comm_adaptive = np.diff(df_adaptive["CommunicationServices_time_decay"]) / np.diff(time_intervals_adaptive)
    smooth_communication_services["CommunicationServices_adaptive_window_smoothness"] = np.std(dydx_comm_adaptive)
    writer.writerow(["CommunicationServices_adaptive_window",
                     smooth_communication_services["CommunicationServices_adaptive_window_smoothness"]])

    # Communication Services Traditional Smoothness
    tra = df_traditional["accuracy"].dropna().tolist()
    dydx_comm_traditional = np.diff(tra) / np.diff(time_intervals_traditional)
    smooth_communication_services["CommunicationServices_traditional_smoothness"] = np.std(dydx_comm_traditional)
    writer.writerow(["CommunicationServices_traditional",
                     smooth_communication_services["CommunicationServices_traditional_smoothness"]])

print(smooth_communication_services)

# smoothness_scores_communication_services = [1 / sd for sd in list(smooth_communication_services.values())]
# max_smooth_communication_services = max(smoothness_scores_communication_services)
# smoothness_scores_normalized_communication_services = [score / max_smooth_communication_services for score in
#                                                        smoothness_scores_communication_services]
smoothness_scores_normalized_communication_services = roughness2smoothness(smooth_communication_services)
print("Normalized Smoothness Scores Communication Services:", smoothness_scores_normalized_communication_services)





bar_colors = ["blue", "hotpink",  "Lime",  "red",  "DarkGrey", "cyan",  "green", "darkorchid","gold", ]



# Male Smoothness Bar Chart
axs[0][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_tech.values(), color=bar_colors[:3], width=0.4,
              label = ["Tech Fixed", "Tech Adaptive", "Tech Traditional"])
axs[0][1].set_title("(b) Tech Smoothness", y=-0.15, pad=-10, fontsize=14).set_position([0.32, -0.14])
axs[0][1].bar_label(axs[0][1].containers[0], fontsize=14)

# axs[0][1].set_yscale("symlog", linthresh=0.1)
axs[0][1].set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"], fontsize=14)
axs[0][1].grid(True, linestyle='--', alpha=0.6)
axs[0][1].set_xticks([], [], rotation=0, fontsize=14)
# axs[0][1].set_ylabel('Normalized Smoothness Score', fontsize=14, labelpad=-1).set_position([0.4, -0.1])


# Female Smoothness Bar Chart
axs[1][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
           smoothness_scores_normalized_consumer_cyclical.values(), color=bar_colors[3:6], width=0.4,
              label = ["Con. Cyclical Fixed", "Con. Cyclical Adaptive",
                       "Con. Cyclical Traditional"])
axs[1][1].set_title("(d) Cons. Cyclical Smoothness", y=-0.15, pad=-10, fontsize=14).set_position([0.32, -0.14])
# axs[1][1].set_yscale("symlog", linthresh=0.1)
axs[1][1].set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"], fontsize=14)
axs[1][1].grid(True, linestyle='--', alpha=0.6)
axs[1][1].set_xticks([], [], rotation=0, fontsize=14)



axs[2][1].bar(["Fixed Window", "Adaptive Window", "Traditional Method"],
              smoothness_scores_normalized_communication_services.values(), color=bar_colors[6:], width=0.4,
                  label = ["Comm. Serv. Fixed", "Comm. Serv. Adaptive",
                           "Comm. Serv. Traditional"])
axs[2][1].set_title("(f) Comm. Serv. Smoothness", y=-0.15, pad=-10, fontsize=14).set_position([0.32, -0.14])
# axs[2][1].set_yscale("symlog", linthresh=0.1)
axs[2][1].set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"], fontsize=14)
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
                 bbox_to_anchor=(-0.13, 2.1), handlelength=1.5,
                 fontsize=13, ncol=3, labelspacing=0.1, handletextpad=0.2, markerscale=0.1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)


fig.savefig("curves_smoothness_score.png", bbox_inches='tight')
plt.show()




plt.figure(figsize=(10, 6))

# Plot each curve for comparison
plt.plot(df_fixed['check_points'], df_fixed['Technology_time_decay'], label='Fixed Window', linestyle='-')
plt.plot(df_adaptive['check_points'], df_adaptive['Technology_time_decay'], label='Adaptive Window', linestyle='--')
plt.plot(np.arange(0, len(tra)), tra, label='Traditional Method', linestyle=':')

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Comparison of Fixed, Adaptive, and Traditional Methods')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

# Derivative comparisons
plt.plot(time_intervals_fixed[:-1], np.abs(np.diff(df_fixed["Technology_time_decay"]) / np.diff(time_intervals_fixed)), label='Fixed Window Derivative', linestyle='-')
plt.plot(time_intervals_adaptive[:-1], np.abs(np.diff(df_adaptive["Technology_time_decay"]) / np.diff(time_intervals_adaptive)), label='Adaptive Window Derivative', linestyle='--')
plt.plot(time_intervals_traditional[:-1], np.abs(np.diff(tra) / np.diff(time_intervals_traditional)), label='Traditional Derivative', linestyle=':')

plt.xlabel('Time')
plt.ylabel('Rate of Change')
plt.title('Comparison of Derivatives (Smoothness)')
plt.legend()
plt.show()



# Standard deviation of derivatives
fixed_roughness = np.std(np.diff(df_fixed["Technology_time_decay"]) / np.diff(time_intervals_fixed))
adaptive_roughness = np.std(np.diff(df_adaptive["Technology_time_decay"]) / np.diff(time_intervals_adaptive))
traditional_roughness = np.std(np.diff(tra) / np.diff(time_intervals_traditional))

print(f"Fixed Window Roughness: {fixed_roughness}")
print(f"Adaptive Window Roughness: {adaptive_roughness}")
print(f"Traditional Method Roughness: {traditional_roughness}")





