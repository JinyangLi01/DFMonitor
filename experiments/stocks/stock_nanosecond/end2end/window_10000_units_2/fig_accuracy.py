
import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
sys.path.append("../../../../../../")
from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
len_chunk = 1
alpha = 0.9997
# Prepare the result file for writing
#
time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')



threshold = 0.4
label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"
window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True



# Write the updated DataFrame to a CSV
filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
            f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_start_time_"
            f"{time_start}_end_time_{time_end}.csv")


#
# ================================== draw the figure ===========================================

df = pd.read_csv(filename)


df["check_points"] = pd.to_datetime(df["check_points"])
print(len(df))
print(df[:2])




draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:09.00', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:13.00', tz='UTC')


df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]

print(len(df))

df.to_csv(f"accuracy_alpha_{str(get_integer(alpha))}.csv", index=False)

# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["check_points"].tolist()
x_list = np.arange(0, len(df))
curve_names = df.columns.tolist()[:-1]

curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services']

# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]
#
#
# num_lines = len(x_list)
# pair_colors = cmaps.set1.colors

fig, ax = plt.subplots(figsize=(3.5, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

for i in range(len(curve_names)):
    ax.plot(x_list, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)






plt.xlabel('',
           fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.xticks([], [])
plt.yticks([0.3, 0.4, 0.45,  0.5], fontsize=13)

# # Manually place the label for 0.6 with a slight adjustment
# plt.text(-845, 0.58, '0.6', fontsize=17, va='bottom')  # Adjust the 0.6 label higher


plt.grid(True, axis='y')
plt.tight_layout()
# plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.4), fontsize=11,
#                ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
#                columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"StockAcc_{date}_{time_period}_a{str(get_integer(alpha))}_tu{time_unit}*{window_size_units}"
            f"_ci{checking_interval}_"
            f"start_{time_start}_end_{time_end}_fig_{draw_figure_start_time}_to_{draw_figure_end_time}.png",
            bbox_inches='tight')
plt.show()
