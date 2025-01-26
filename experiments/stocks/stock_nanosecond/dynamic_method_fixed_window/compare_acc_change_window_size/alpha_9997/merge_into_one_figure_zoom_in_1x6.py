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

checking_interval = "100000 nanosecond"
use_nanosecond = True



monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Energy'}, {"sector": 'Healthcare'},
                    {"sector": 'Financial Services'}]

sorted_sector_list = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
                          'Healthcare', 'Financial Services']


window_size_units_list = [10, 100, 1000, 10000, 20000, 30000]


alpha=0.9997
time_start_picture = 0
time_end_picture = 0


# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 6, figsize=(14.6, 1.7), constrained_layout=False)
plt.subplots_adjust(top=0.84, bottom=0.31, hspace=0, wspace=0.3, left=0.03, right=0.99)

time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')


# Choose some timestamps for labeling on the x-axis
selected_check_points = [
    pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.20', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.40', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.60', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.80', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
]


for a, window_size in enumerate(window_size_units_list):
    filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
                f"_time_unit_{time_unit}*{window_size}_check_interval_{checking_interval}_start_time_"
            f"{time_start}_end_time_{time_end}.csv")

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
    time_start_picture = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
    time_end_picture = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
    total_duration_ns = (time_end_picture - time_start_picture).total_seconds() * 1e9

    df["check_points"] = pd.to_datetime(df["check_points"])
    # get data between time_start and time_end
    df = df[(df["check_points"] >= time_start_picture) & (df["check_points"] <= time_end_picture)]

    # df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
    check_points = df["check_points"].tolist()
    x_list = np.arange(0, len(df))

    curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services']

    curve_colors = sns.color_palette(palette=['blue', 'limegreen', '#ffb400', 'darkviolet', 'black',
                                              "red", 'cyan', 'magenta'])
    # Calculate indices based on the proportional position within the time range
    x_tick_indices = []
    x_tick_labels = []
    for cp in selected_check_points:
        if time_start_picture <= cp <= time_end_picture:
            # Calculate the offset in nanoseconds from `time_start_picture`
            offset_ns = (cp - time_start_picture).total_seconds() * 1e9
            # Determine the proportional index within the range of `df`
            index = int((offset_ns / total_duration_ns) * len(df))
            x_tick_indices.append(index)
            x_tick_labels.append(cp.strftime('.%f')[:-4])  # Format as 'SS.ms'

    window_size_smoothing = 50
    for i in range(len(curve_names)):
        # Get the smoothed values using a rolling mean
        smoothed_data = df[curve_names[i].replace(" ", "") + "_time_decay"].rolling(window=window_size_smoothing).mean()
        if i == 0 or i == 1:
            axes[a].plot(x_list, smoothed_data,
                         linewidth=0.1, markersize=0.3,
                         label=curve_names[i], linestyle='-',
                         marker='o', color=curve_colors[i], alpha=0.8)
        else:
            axes[a].plot(x_list, smoothed_data,
                     linewidth=0.08, markersize=0.3,
                label=curve_names[i], linestyle='-',
                marker='o', color=curve_colors[i], alpha=0.7)


    xlabel = f"({chr(97 + a)}) window$={window_size}$"
    axes[a].grid(True, alpha=1)
    axes[a].tick_params(axis='both', which='major', labelsize=15, pad=1.5)
    axes[a].set_ylabel('')
    axes[a].set_xlabel(xlabel, fontsize=17, labelpad=0).set_position([0.38, 1])
    axes[a].tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
    axes[a].tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.
    axes[a].set_xticks(x_tick_indices)
    x_tick_labels = [f'{str(tick).replace(".", "•")}' for tick in x_tick_labels]
    axes[a].set_xticklabels(x_tick_labels, fontsize=15)  # Rotate for readability

# axes[0].set_ylim(0.4, 0.85)
# axes[1].set_ylim(0.4, 0.85)
# axes[2].set_ylim(0.4, 0.85)
# axes[3].set_ylim(0.4, 0.9)
#
# axes[4].set_ylim(0.4, 1.0)
# axes[5].set_ylim(0.4, 1.0)

axes[0].set_yticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8], fontsize=15)
axes[1].set_yticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8], fontsize=15)
axes[2].set_yticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8], fontsize=15)
axes[3].set_yticks([0.4, 0.6, 0.8], [0.4, 0.6, 0.8], fontsize=15)
axes[4].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
axes[5].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
# axes[6].set_yticks([0.55, 0.6, 0.65], ['0.55', '0.60', '0.65'], fontsize=15)
# axes[1].set_yticks([0.4, 0.6, 0.8], ["0.40", "0.60", "0.80"], fontsize=15)
# axes[2].set_yticks([0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.7], fontsize=15)



handles, labels = [], []
h, l = axes.flat[-1].get_legend_handles_labels()
handles.extend(h)
labels.extend(l)


leg = axes[-1].legend(loc="upper left", fontsize=16,
                bbox_to_anchor=(-5, 1.48), ncol=7,
                labelspacing=0.5, draggable=True,
                handletextpad=0.5, markerscale=30, handlelength=1.2, handleheight=1,
               columnspacing=0.5, borderpad=0.2, frameon=False, title=None)
for line in leg.get_lines():
    line.set_linewidth(4)


fig.savefig(f"compare_accuracy_time_decay_fixed_window_change_window_size_zoom_in_{time_start_picture}---{time_start_picture}.png")
plt.show()
