import ast
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)




time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
len_chunk = 1


label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "100000 nanosecond"
window_size_units = "1"
checking_interval = "100000 nanosecond"
use_nanosecond = True
Tb = 50
Tin = 30



# monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
#                     {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
#                     {"sector": 'Financial Services'}]
#
# sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
#                           'Healthcare', 'Financial Services']

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'}]

sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services']

alpha_list = [0.9997, 0.99995, 0.99999]





# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(7, 1.7))
plt.subplots_adjust(top=0.76, bottom=0.32, hspace=0, wspace=0.36, left=0.09, right=0.99)

for a, alpha in enumerate(alpha_list):
    filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
                f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_v1.csv")
    df = pd.read_csv(filename)
    df_below_threshold = []
    df["check_points"] = pd.to_datetime(df["check_points"])
    print(alpha, len(df))
    print(df.columns)
    warm_up_time = pd.Timestamp('2024-10-15 14:00:06.7', tz='UTC')

    df = df[df["check_points"] >= warm_up_time]

    print(len(df))

    # df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
    check_points = df["check_points"].tolist()
    x_list = np.arange(0, len(df))

    curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services']

    curve_colors = sns.color_palette(palette=['blue', 'limegreen', '#ffb400', 'darkviolet', 'cyan', 'black',
                                              "red", 'magenta'])

    for i in range(len(curve_names)):
        if i == 0 or i == 1 or i == 2 or i == 3:
            axes[a].plot(x_list, df[curve_names[i].replace(" ", "") + "_time_decay"].tolist(),
                         linewidth=0.1, markersize=0.1,
                         label=curve_names[i], linestyle=':',
                         marker='o', color=curve_colors[i], alpha=0.5)
        else:
            axes[a].plot(x_list, df[curve_names[i].replace(" ", "") + "_time_decay"].tolist(),
                     linewidth=0.3, markersize=0.1,
                label=curve_names[i], linestyle='-',
                marker='o', color=curve_colors[i], alpha=1)

    # Add any additional manual points if desired
    manual_points = [
        # pd.Timestamp('2024-10-15 14:00:08', tz='UTC'),
                     # pd.Timestamp('2024-10-15 14:00:13', tz='UTC'),
                     pd.Timestamp('2024-10-15 14:00:16', tz='UTC')]
    x_ticks_points = sorted(manual_points)

    check_points_seconds = df["check_points"].dt.floor('s').tolist()

    # Find the first index of each timestamp in x_ticks_points within check_points_seconds
    tick_idx = [check_points_seconds.index(t) for t in x_ticks_points if t in check_points_seconds]
    tick_labels = [t.strftime(':%S') for t in x_ticks_points if t in check_points_seconds]
    tick_idx.append(30000)
    tick_labels.append(':13')
    tick_idx.append(5500)
    tick_labels.append(':08')

    tick_idx.sort()
    tick_labels.sort()

    # Set the ticks and labels if tick_idx has values
    if tick_idx:
        axes[a].set_xticks(tick_idx)
        axes[a].set_xticklabels(tick_labels, rotation=0, fontsize=14, fontweight='bold')
    else:
        print("Warning: No matching tick positions found in check_points for the specified manual points.")


    xlabel = f"({chr(97 + a)}) $\\alpha$={alpha}"
    axes[a].grid(True, alpha=1)
    axes[a].tick_params(axis='both', which='major', labelsize=14, pad=1.5)
    axes[a].set_ylabel('')
    axes[a].set_xlabel(xlabel, fontsize=14, labelpad=0).set_position([0.38, 1])

    axes[a].tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
    axes[a].tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.


axes[0].set_yticks([0.55, 0.6, 0.65,  0.7], ["0.55", "0.60", "0.65", "0.70"], fontsize=14)
axes[1].set_yticks([0.55, 0.6, 0.65, 0.7], ["0.55", "0.60", "0.65", "0.70"], fontsize=14)
axes[2].set_yticks([0.55, 0.6, 0.65, 0.7], ["0.55", "0.60", "0.65", "0.70"], fontsize=14)


time_start = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')

handles, labels = [], []
h, l = axes.flat[-1].get_legend_handles_labels()
handles.extend(h)
labels.extend(l)



x = fig.legend(handles, labels, loc="upper left", fontsize=14,
                bbox_to_anchor=(0.05, 1), ncol=3,
                labelspacing=0.6, draggable=True,
                handletextpad=0.6, markerscale=80, handlelength=1.2, handleheight=1,
               columnspacing=0.5, borderpad=0.2, frameon=False, title=None)



# Modify the line width in the legend handles
for line in x.get_lines():
    line.set_linewidth(4)
    line.set_alpha(1)
    line.set_linestyle('-')
    line.markersize = 20


fig.savefig("compare_accuracy_time_decay_adaptive_window_change_alpha.png")
plt.show()
