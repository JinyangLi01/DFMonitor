
#import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys
import colormaps as cmaps
import math


sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)

alpha = 0.998
time_unit = "30 min"
window_size_units = 1
df = pd.read_csv(f"movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}*{window_size_units}.csv")
print(df)

x_list = np.arange(0, len(df))
male_time_decay = df["male_time_decay"].tolist()
female_time_decay = df["female_time_decay"].tolist()


# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "orange"]

num_lines = len(x_list)
pair_colors = cmaps.set1.colors

fig, ax = plt.subplots(figsize=(6, 3.5))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)


ax.plot(x_list, male_time_decay, linewidth=1, markersize=1.5, label='Male time decay', linestyle='-', marker='o', color=pair_colors[1])
ax.plot(x_list, female_time_decay, linewidth=1, markersize=1.5, label='Female time decay', linestyle='-', marker='o', color=pair_colors[4])


plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
plt.xlabel('time stamps\n from 01/01/2013, 1m interval',
           fontsize=20, labelpad=-2).set_position((0.47, 0.1))
plt.ylabel('Accuracy', fontsize=20, labelpad=-1)
plt.ylim(0.6, 1.0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.3), fontsize=15,
               ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
               columnspacing=0.2, borderpad=0.2, frameon=True)
plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_per_item_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}*{window_size_units}.png", bbox_inches='tight')
plt.show()