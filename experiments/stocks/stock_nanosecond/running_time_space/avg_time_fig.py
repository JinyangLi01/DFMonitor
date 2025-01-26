from datetime import datetime

#import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anyio.lowlevel import checkpoint

import seaborn as sns
import colorsys
import colormaps as cmaps
import math
#
#
# sns.set_palette("Paired")
# sns.set_context("paper", font_scale=2)
#
# plt.figure(figsize=(6, 3.5))
# plt.rcParams['font.size'] = 20

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

alpha = 0.9997

label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "100000 nanosecond"
window_size_units = "1"
checking_interval = "200000 nanosecond"
use_nanosecond = True

dynamic_file_name = "dynamic_window_running_time_Accuracy_v5.csv"
dynamic_df = pd.read_csv(dynamic_file_name)

fixed_file_name = "fixed_window_running_time_Accuracy_v5.csv"
fixed_df = pd.read_csv(fixed_file_name)

column_names = dynamic_df.columns.tolist()
column_names_dynamic = [x+"dynamic" for x in column_names]
column_names_fixed = [x+"fixed" for x in column_names]








pair_colors = sns.husl_palette(s=.4)
pair_colors = sns.color_palette()[:3] + sns.color_palette()[4:7]

fig, ax = plt.subplots(figsize=(3, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

dynamic_mean = dynamic_df.mean()
fixed_mean = fixed_df.mean()
print("dynamic mean\n", dynamic_mean)
print("fixed mean\n", fixed_mean)

# Plot the bar chart
plt.bar(["fixed\nbit", "fixed\ncounter"], fixed_mean[:2], color=pair_colors, alpha=0.7)
plt.bar(["dyn.\nbit", "dyn.\ncounter"], dynamic_mean[:2], color=pair_colors[2:4], alpha=0.7)




# Add labels and title
plt.ylabel('Average Value', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 0.2)
ax.tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
ax.tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.



plt.grid(True, axis='y')
#
# plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 2), fontsize=11,
#                ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
#                columnspacing=0.6, borderpad=0.2, frameon=True)

plt.savefig(f"Stock_acc_time_decay_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
            f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_v1.png", bbox_inches='tight')
plt.show()