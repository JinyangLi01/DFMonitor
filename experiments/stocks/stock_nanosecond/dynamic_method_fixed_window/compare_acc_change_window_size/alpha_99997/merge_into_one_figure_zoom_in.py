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
time_unit = "10000 nanosecond"

checking_interval = "10000 nanosecond"
use_nanosecond = True



monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
                    {"sector": 'Financial Services'}]

sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
                          'Healthcare', 'Financial Services']


window_size_units_list = [80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000]


alpha=0.99997


# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 8, figsize=(14.6, 1.7), constrained_layout=True)
plt.subplots_adjust(top=0.84, bottom=0, hspace=0, wspace=0, left=0, right=0.93)

for a, window_size in enumerate(window_size_units_list):
    filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
                f"_time_unit_{time_unit}*{window_size}_check_interval_{checking_interval}.csv")

    with open(filename, 'r') as f:
        contents = f.read()
    if "]\"" in contents or "\"[" in contents:
        updated_contents = contents.replace("]\"", "")
        updated_contents = updated_contents.replace("\"[", "")

        with open(filename, 'w') as f:
            f.write(updated_contents)


    df = pd.read_csv(filename)
    df_below_threshold = []
    df["check_points"] = pd.to_datetime(df["check_points"])
    print(alpha, len(df))
    print(df.columns)
    # Remove timezone from warm_up_time
    time_start = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')
    time_end = pd.Timestamp('2024-10-15 14:00:15.00', tz='UTC')

    df["check_points"] = pd.to_datetime(df["check_points"])
    # get data between time_start and time_end
    df = df[(df["check_points"] >= time_start) & (df["check_points"] <= time_end)]

    print(len(df))

    # df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
    check_points = df["check_points"].tolist()
    x_list = np.arange(0, len(df))

    curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive',
                          'Healthcare']

    curve_colors = sns.color_palette(palette=['cyan', '#ffb400', 'limegreen', 'darkviolet', 'blue', 'black',
                                              "red", 'magenta'])

    for i in range(len(curve_names)):
        if i == 0 or i == 1 or i == 2:
            axes[a].plot(x_list, df[curve_names[i].replace(" ", "") + "_time_decay"].tolist(),
                         linewidth=0.1, markersize=0.7,
                         label=curve_names[i], linestyle='-',
                         marker='o', color=curve_colors[i], alpha=0.5)
        else:
            axes[a].plot(x_list, df[curve_names[i].replace(" ", "") + "_time_decay"].tolist(),
                     linewidth=1, markersize=1,
                label=curve_names[i], linestyle='-',
                marker='o', color=curve_colors[i], alpha=1)

    # # Add any additional manual points if desired
    # manual_points = [
    #     # pd.Timestamp('2024-10-15 14:00:08', tz='UTC'),
    #                  # pd.Timestamp('2024-10-15 14:00:13', tz='UTC'),
    #                  pd.Timestamp('2024-10-15 14:00:16', tz='UTC')]
    # x_ticks_points = sorted(manual_points)
    #
    # # Create a list with second-level precision timestamps
    # check_points_seconds = df["check_points"].dt.floor('s').tolist()
    #
    # # Find the first index of each timestamp in x_ticks_points within check_points_seconds
    # tick_idx = [check_points_seconds.index(t) for t in x_ticks_points if t in check_points_seconds]
    # tick_labels = [t.strftime(':%S') for t in x_ticks_points if t in check_points_seconds]
    # tick_idx.append(30000)
    # tick_labels.append(':13')
    # tick_idx.append(5500)
    # tick_labels.append(':08')
    # tick_idx.sort()
    # tick_labels.sort()
    #
    # # Set the ticks and labels if tick_idx has values
    # if tick_idx:
    #     axes[a].set_xticks(tick_idx)
    #     axes[a].set_xticklabels(tick_labels, rotation=0, fontsize=15, fontweight='bold')
    # else:
    #     print("Warning: No matching tick positions found in check_points for the specified manual points.")


    xlabel = f"({chr(97 + a)}) {window_size}"
    axes[a].grid(True, alpha=1)
    axes[a].tick_params(axis='both', which='major', labelsize=15, pad=1.5)
    axes[a].set_ylabel('')
    axes[a].set_xlabel(xlabel, fontsize=17, labelpad=0).set_position([0.38, 1])
    axes[a].tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
    axes[a].tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.
    axes[a].set_ylim(0.5, 0.7)

# axes[0].set_yticks([0.2, 0.5, 0.8], ["0.20", "0.50", "0.80"], fontsize=15)
# axes[6].set_yticks([0.55, 0.6, 0.65], ['0.55', '0.60', '0.65'], fontsize=15)
# axes[1].set_yticks([0.4, 0.6, 0.8], ["0.40", "0.60", "0.80"], fontsize=15)
# axes[2].set_yticks([0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.7], fontsize=15)



handles, labels = [], []
h, l = axes.flat[-1].get_legend_handles_labels()
handles.extend(h)
labels.extend(l)



fig.legend(handles, labels, loc="upper left", fontsize=16,
                bbox_to_anchor=(0.02, 1.08), ncol=7,
                labelspacing=0.2, draggable=True,
                handletextpad=0.2, markerscale=7, handlelength=1.2, handleheight=1,
               columnspacing=0.2, borderpad=0.2, frameon=False, title=None)



plt.tight_layout()
fig.savefig(f"compare_accuracy_time_decay_fixed_window_change_window_size_zoom_in_{time_start}---{time_end}.png")
plt.show()
