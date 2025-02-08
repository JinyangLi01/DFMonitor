
import colorsys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math

from algorithm.dynamic_window import Accuracy_workload as workload
import seaborn as sns
import matplotlib.dates as mdates


def belong_to_group(row, group):
    for key in group.keys():
        if row[key] != group[key]:
            return False
    return True


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)

#
# plt.rcParams['font.family'] = 'serif'



alpha = 0.99995
curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services']

curve_colors = sns.color_palette(palette=['blue', 'limegreen', '#ffb400', 'darkviolet', 'black', 'cyan',
                                          "red", 'magenta'])
remove_fractions = [70, 80, 90]
draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:05.50', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')

fig, axes = plt.subplots(1, 3, figsize=(6.4, 0.8))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.26, hspace=0)


for idx in range(len(remove_fractions)):
    fractions = remove_fractions[idx]
    filename = f"dynamic_CR_random_decay_fraction_{fractions}_alpha_{get_integer(alpha)}.csv"
    df = pd.read_csv(filename)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # get data between time_start and time_end
    df = df[(df["timestamp"] >= draw_figure_start_time) & (df["timestamp"] <= draw_figure_end_time)]
    # df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
    check_points = df["timestamp"].tolist()
    x_list = np.arange(0, len(df))
    start_floor = draw_figure_start_time.floor('s')  # Floor to nearest second
    end_ceil = draw_figure_end_time.ceil('s')  # Ceiling to nearest second
    xticks_times = pd.date_range(start=start_floor, end=end_ceil, freq='1s')
    ax = axes[idx]
    for i in range(len(curve_names)):
        ax.plot(check_points, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i],
                linestyle='-', marker='o', color=curve_colors[i], alpha=0.5)
    ax.set_xticks([xticks_times[k] for k in range(0, len(xticks_times), 2)])
    ax.set_xticklabels(['05', '07', '09', '11', '13', '15', '17'], fontsize=14, rotation=0)
    ax.set_yscale('log')
    ax.set_yticks([0.01, 0.1, 1])
    ax.set_yticklabels([0.01, 0.1, 1], fontsize=14)
    ax.grid(True)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=0)
    xlabel = f"({chr(97 + idx)}) ${fractions}$% removed"
    ax.set_xlabel(xlabel, fontsize=13, labelpad=0).set_position([0.38, 1])



handles, labels = [], []
for ax in axes:
    for line in ax.get_lines():
        handles.append(line)
        labels.append(line.get_label())
    break



plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.9, 1.53), fontsize=14,
               ncol=3, labelspacing=0.6, handletextpad=0.2, markerscale=6, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)

plt.savefig(f"dynamic_CR_alpha_{str(get_integer(alpha))}_different_fractions.png", bbox_inches='tight')
plt.show()

