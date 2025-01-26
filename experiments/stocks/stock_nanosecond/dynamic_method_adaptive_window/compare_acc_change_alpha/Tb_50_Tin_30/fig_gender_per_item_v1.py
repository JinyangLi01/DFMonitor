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

alpha = 0.999999

label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "100000 nanosecond"
window_size_units = "1"
checking_interval = "200000 nanosecond"
use_nanosecond = True


filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
            f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_v1.csv")
df = pd.read_csv(filename)


df["check_points"] = pd.to_datetime(df["check_points"])
print(len(df))
print(df[:2])


warm_up_time = pd.Timestamp('2024-10-15 14:00:10.005', tz='UTC')

df = df[df["check_points"] >= warm_up_time]


print(len(df))

# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["check_points"].tolist()
x_list = np.arange(0, len(df))
curve_names = df.columns.tolist()[:-1]

curve_names = ["Technology", "CommunicationServices", "ConsumerDefensive", "FinancialServices",
               "ConsumerCyclical", "Energy", "Healthcare"]

# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]

pair_colors = sns.color_palette("tab10", 9)

fig, ax = plt.subplots(figsize=(3.5, 2))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

for i in range(len(curve_names)):
    ax.plot(x_list, df[curve_names[i]+"_time_decay"].tolist(), linewidth=0.7, markersize=1,
            label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)


#
# ax.plot(x_list, technology_time_decay, linewidth=1, markersize=1, label='Tech', linestyle='--', marker='o', color="blue", alpha=0.5)
# ax.plot(x_list, consumer_time_decay, linewidth=1.3, markersize=1, label='Consumer', linestyle='-', marker='o', color="darkorange", alpha=0.7)
# ax.plot(x_list, communication_time_decay, linewidth=1.5, markersize=1, label='Communication', linestyle='-', marker='o', color="green", alpha=0.5)



# y_margin = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.53  # 5% above the lower y-limit
#
#
# # Add a label on the x-axis near the vertical line
# plt.text(0, y_margin, check_points[0].replace(" ", "\n"), color='red', fontsize=13,
#          verticalalignment='bottom', horizontalalignment='center')

# plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=1)
# plt.axvline(x=len(check_points)-1, color='black', linestyle='--', linewidth=1.5, alpha=1)
#
# decline_threshold = 0.1
# decline_point = 0
# for i in range(1, len(check_points)):
#     if female_time_decay[i-1] - female_time_decay[i] > decline_threshold:
#         plt.axvline(x=i, color='black', linestyle='--', linewidth=1.5, alpha=1)
#         # plt.text(i, y_margin, check_points[i].replace(" ", "\n"), color='red', fontsize=13,
#         #          verticalalignment='bottom', horizontalalignment='center')
#         decline_point = i
#         break
#

# # Add a tick under the vertical line
# plt.xticks([0, decline_point, len(check_points)],
#            ["  " + check_points[0].split(" ")[0],
#                 check_points[decline_point].split(" ")[0], ""], rotation=0, fontsize=13)  # Adds tick at x=0 with label
#

ax.tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
ax.tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.


plt.xlabel('',
           fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.xticks([], [])
plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8], fontsize=13)

# # Manually place the label for 0.6 with a slight adjustment
# plt.text(-845, 0.58, '0.6', fontsize=17, va='bottom')  # Adjust the 0.6 label higher


plt.grid(True, axis='y')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 2), fontsize=11,
               ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"Stock_acc_time_decay_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
            f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_v1.png", bbox_inches='tight')
plt.show()