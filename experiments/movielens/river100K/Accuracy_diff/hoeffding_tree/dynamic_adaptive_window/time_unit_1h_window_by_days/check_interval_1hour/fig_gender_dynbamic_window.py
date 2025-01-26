import csv

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

plt.figure(figsize=(3.5, 2))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


time_unit = "1 hour"
alpha = 0.996
T_b = 4
T_in = 2
checking_interval = "1 hour"
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"

df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.csv")
print(df)

import datetime
x_list = np.arange(0, len(df))
male_time_decay = df["male_time_decay_dynamic"].tolist()
female_time_decay = df["female_time_decay_dynamic"].tolist()
df["check_points"] = df["check_points"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))

check_point_times = df["check_points"].tolist()


# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange"]

fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

ax.plot(x_list, male_time_decay, linewidth=1, markersize=1, label='Male', linestyle='-', marker='o', color=pair_colors[0])
ax.plot(x_list, female_time_decay, linewidth=1, markersize=1, label='Female', linestyle='-', marker='o', color=pair_colors[1])

check_point = 1
for j in range(10, len(female_time_decay)):
    if - female_time_decay[j] + female_time_decay[j-1] > 0.1:
        check_point = j
        break
print("check_point", check_point)
plt.axvline(x=check_point, color='black', linestyle='--', linewidth=1.5, alpha=1)




ax.tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
ax.tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.


plt.xticks([check_point], [check_point_times[check_point].split(" ")[0]], rotation=0, fontsize=14)
# plt.xlabel('time stamps',
#            fontsize=20, labelpad=-2).set_position((0.47, 0.1))
plt.ylabel('Accuracy', fontsize=14, labelpad=-1)
plt.ylim(0.5, 1.0)
plt.yticks([0.5, 0.6, 0.8, 1.0], ["0.5", "0.6", "0.8", "1.0"], fontsize=14)
plt.grid(True, axis='y')

plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.3), fontsize=12,
               ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=2,
               columnspacing=0.2, borderpad=0.2, frameon=True, handlelength=1.5)
plt.savefig(f"Acc_hoeffding_time_decay_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.png", bbox_inches='tight')
plt.show()