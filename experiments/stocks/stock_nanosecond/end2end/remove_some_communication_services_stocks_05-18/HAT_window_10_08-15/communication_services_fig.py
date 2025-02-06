import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys

from scipy.fftpack import idctn

from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'

# # activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)



alpha = 0.99995
threshold = 0.4
label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True
curve_names = ['Technology', 'Communication Services', 'Consumer Cyclical']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]
removed_percentage = [70, 80, 90]

draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')

fig, axes = plt.subplots(1, 3, figsize=(7, 1.4))
plt.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.1, wspace=0.4)


for j in range(len(removed_percentage)):
    stock_fraction = removed_percentage[j]
    data_file_name = f"accuracy_alpha_99995_remove_fraction_{stock_fraction}.csv"
    df = pd.read_csv(data_file_name)
    df["check_points"] = pd.to_datetime(df["check_points"])
    print(len(df))
    print(df[:2])
    df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]
    check_points = df["check_points"].tolist()
    x_list = np.arange(0, len(df))
    start_floor = draw_figure_start_time.floor('s')  # Floor to nearest second
    end_ceil = draw_figure_end_time.ceil('s')  # Ceiling to nearest second
    xticks_times = pd.date_range(start=start_floor, end=end_ceil, freq='1s')
    ax = axes[j]
    for i in range(len(curve_names)):
        ax.plot(check_points, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i],
                linestyle='-',
                marker='o', color=pair_colors[i], alpha=0.5)
    ax.set_xticks([xticks_times[k] for k in range(0, len(xticks_times), 2)])
    ax.set_xticklabels(['05', '07', '09', '11', '13', '15', '17'], fontsize=13, rotation=0)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
    # ax.set_ylabel('Accuracy', fontsize=14, labelpad=-1).set_position([0.08, 0.5])
    ax.set_yticks([0.55, 0.6, 0.65])
    ax.set_yticklabels([0.55, '0.60', 0.65], fontsize=14)
    ax.set_ylim(0.55, 0.65)
    ax.grid(True)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=1)
    xlabel = f"({chr(97 + j)}) remove ${stock_fraction}$% stocks"
    ax.set_xlabel(xlabel, fontsize=13, labelpad=0).set_position([0.38, 1])


#
# fig.supxlabel('Time (s) 14:00:05.00 to 14:00:18.00', fontsize=14, y =-0.25)


handles, labels = [], []
for ax in axes:
    for line in ax.get_lines():
        handles.append(line)
        labels.append(line.get_label())
    break



plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(-1.05, 1.46), fontsize=14,
               ncol=3, labelspacing=0.5, handletextpad=0.2, markerscale=6, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"StockAcc_end2end_HAT_alpha_{get_integer(alpha)}_remove_stock_different_fraction.png",
            bbox_inches='tight')
plt.show()
