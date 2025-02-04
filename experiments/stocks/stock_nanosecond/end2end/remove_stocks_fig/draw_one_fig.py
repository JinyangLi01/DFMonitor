from datetime import datetime

#import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anyio.lowlevel import checkpoint

from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys
import colormaps as cmaps
import math

from experiments.stocks.stock_nanosecond.method_curve_direct_compare.fig_new import df_tech


def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)



# Set up the plot
fig, ax = plt.subplots(figsize=(4, 2.3))
window_size_units = 10
alpha = 0.9997
# Write the updated DataFrame to a CSV
filename = f"../remove_some_consumer_cyclical_stock/HAT_window_{window_size_units}_alpha_{get_integer(alpha)}/accuracy_alpha_{get_integer(alpha)}.csv"

df = pd.read_csv(filename)


# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["check_points"].tolist()
x_list = np.arange(0, len(df))
curve_names = df.columns.tolist()[:-1]

curve_names = ['Technology', 'Communication Services', 'Consumer Cyclical']

# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]

for i in range(len(curve_names)):
    ax.plot(x_list, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)














plt.xlabel(f"{time_window_str} intervals")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.5), fontsize=10,
               ncol=3, labelspacing=0.5, handletextpad=0.2, markerscale=1, handlelength=1.5,
               columnspacing=0.6, borderpad=0.2, frameon=True)

plt.savefig(f"traditional_accuracy_time_window_{time_window_str}.png")

# Show the plot
plt.show()


