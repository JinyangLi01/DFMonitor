
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





alpha = 0.99995
fractions = 8
filename = f"dynamic_CR_random_decay_fraction_{fractions}_alpha_9997.csv"

df = pd.read_csv(filename)


df["timestamp"] = pd.to_datetime(df["timestamp"])
print(len(df))
print(df[:2])



time_start = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')


df["timestamp"] = pd.to_datetime(df["timestamp"])
# get data between time_start and time_end
df = df[(df["timestamp"] >= time_start) & (df["timestamp"] <= time_end)]






print(len(df))

# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["timestamp"].tolist()
x_list = np.arange(0, len(df))
curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services']

curve_colors = sns.color_palette(palette=['blue', 'limegreen', '#ffb400', 'darkviolet', 'black', 'cyan',
                                          "red", 'magenta'])

fig, ax = plt.subplots(figsize=(3.5, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

for i in range(len(curve_names)):
    ax.plot(x_list, df[curve_names[i]].tolist(), linewidth=2, markersize=2, label=curve_names[i], linestyle='-',
            marker='o', color=curve_colors[i], alpha=0.5)







#
# plt.ylim(0.4, 0.7)

plt.xlabel('',
           fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.xticks([], [])

# # Manually place the label for 0.6 with a slight adjustment
# plt.text(-845, 0.58, '0.6', fontsize=17, va='bottom')  # Adjust the 0.6 label higher


plt.grid(True, axis='y')

plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.4), fontsize=11,
               ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"dynamic_CR_alpha_{str(get_integer(alpha))}_fraction_{fractions}.png", bbox_inches='tight')
plt.show()

