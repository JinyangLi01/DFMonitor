import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
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




label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True


stock_fraction = 9
# Prepare the result file for writing
data_file_name = f"accuracy_alpha_99995_remove_fraction_{stock_fraction}.csv"
df = pd.read_csv(data_file_name)


df["check_points"] = pd.to_datetime(df["check_points"])
print(len(df))
print(df[:2])




draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:05.00', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:18.00', tz='UTC')


df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]

print(len(df))

# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["check_points"].tolist()
x_list = np.arange(0, len(df))
curve_names = df.columns.tolist()[:-1]

curve_names = ['Technology', 'Communication Services', 'Consumer Cyclical']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]

# Generate x-axis ticks at every whole second in the time range
start_floor = draw_figure_start_time.floor('s')  # Floor to nearest second
end_ceil = draw_figure_end_time.ceil('s')  # Ceiling to nearest second
xticks_times = pd.date_range(start=start_floor, end=end_ceil, freq='1s')



fig, ax = plt.subplots(figsize=(3.5, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.3)

for i in range(len(curve_names)):
    ax.plot(check_points, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)


ax.set_xticks(xticks_times)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
plt.xticks(rotation=0, ha='right', fontsize=10)  # Rotate labels to avoid overlap



# plt.xlabel('',
#            fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.yticks([0.55, 0.6, 0.65], fontsize=13)
plt.ylim(0.55, 0.65)
plt.grid(True)


plt.grid(True, axis='y')

# plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.4), fontsize=11,
#                ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
#                columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"StockAcc_end2end_HAT_alpha_{get_integer(alpha)}_remove_stock_fraction_{stock_fraction}.png",
            bbox_inches='tight')
plt.show()
