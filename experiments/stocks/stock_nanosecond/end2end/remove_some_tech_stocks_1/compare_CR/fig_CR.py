
import colorsys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter from matplotlib.ticker

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



file_original = pd.read_csv("dynamic_CR_original_alpha_9997.csv")
CR_expotential_8 = pd.read_csv("dynamic_CR_exponential_decay_rate_8_alpha_9997.csv")
CR_expotential_9 = pd.read_csv("dynamic_CR_exponential_decay_rate_9_alpha_9997.csv")
CR_linear_8 = pd.read_csv("dynamic_CR_linear_decay_rate_8_alpha_9997.csv")
CR_linear_9 = pd.read_csv("dynamic_CR_linear_decay_rate_9_alpha_9997.csv")

CR_exponential_5 = pd.read_csv("dynamic_CR_exponential_decay_rate_5_alpha_9997.csv")
date_column = 'timestamp'



fig, ax = plt.subplots(figsize=(5, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

labels = ["Original", "Exponential 8", "Exponential 9", "Linear 8", "Linear 9", "Exponential 5"]
colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]
time_start = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')


def processing(data_stream, ax, i):
    data_stream[date_column] = pd.to_datetime(data_stream[date_column])
    data_stream = data_stream[(data_stream[date_column] >= time_start) & (data_stream[date_column] <= time_end)]
    ax.plot(data_stream["timestamp"], data_stream["Technology"], linewidth=2, markersize=2, label=labels[i], linestyle='-',
            marker='o', color=colors[i], alpha=0.5)

processing(file_original, ax, 0)
processing(CR_expotential_8, ax, 1)
processing(CR_expotential_9, ax, 2)
processing(CR_linear_8, ax, 3)
processing(CR_linear_9, ax, 4)
processing(CR_exponential_5, ax, 5)

#
# plt.ylim(0.4, 0.7)

plt.xlabel('', fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('CR', fontsize=13, labelpad=-1)
plt.yticks([0.2, 0.5, 0.6], fontsize=13)
# Generate custom x-ticks at 0.1-second intervals
x_ticks = pd.date_range(start=time_start, end=time_end, freq='100ms')  # 100ms = 0.1 seconds

# Extract fractional seconds (1, 2, 3, ...)
fractional_seconds = [(timestamp.microsecond / 1000000) for timestamp in x_ticks]
fractional_seconds = ["10.0"] + fractional_seconds[1:-1] + ["11.0"]

# Format the x-ticks
ax.set_xticks(x_ticks, fractional_seconds)

print("x_ticks", x_ticks)
print("fractional_seconds", fractional_seconds)


# Rotate x-tick labels for better readability
plt.xticks(rotation=0, fontsize=12)
# Set labels and grid
plt.xlabel('Timestamp (Fractional Seconds)', fontsize=14, labelpad=5)
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.grid(True, axis='y')
plt.grid(True, axis='y')

plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.5), fontsize=11,
               ncol=3, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"CR_tech", bbox_inches='tight')
plt.show()

