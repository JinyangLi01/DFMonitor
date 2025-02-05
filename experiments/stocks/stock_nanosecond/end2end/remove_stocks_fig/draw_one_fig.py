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



def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
total_duration_ns = (draw_figure_end_time - draw_figure_start_time).total_seconds() * 1e9



# Set up the plot
fig, ax = plt.subplots(1, 3, figsize=(7, 1.4), constrained_layout=False)
plt.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.33, wspace=0.3)

window_size_units = 10
alpha = 0.9997
curve_names = ['Technology', 'Communication Services', 'Consumer Cyclical']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]

filename = f"../remove_some_tech_stocks_2/HAT_window_{window_size_units}/accuracy_alpha_{get_integer(alpha)}.csv"
df = pd.read_csv(filename)
df["check_points"] = pd.to_datetime(df["check_points"])
df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]

x_list = np.arange(0, len(df))


for i in range(len(curve_names)):
    ax[0].plot(x_list, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)




filename = f"../remove_some_communication_services_stock_2/HAT_window_{window_size_units}_alpha_{get_integer(alpha)}/accuracy_alpha_{get_integer(alpha)}.csv"


df = pd.read_csv(filename)
df["check_points"] = pd.to_datetime(df["check_points"])
df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]


print(len(df))
print(df[:2])

x_list = np.arange(0, len(df))


for i in range(len(curve_names)):
    ax[1].plot(x_list, df[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)



filename = f"../remove_some_consumer_cyclical_stock/HAT_window_10_alpha_9997/accuracy_alpha_9997_remove_fraction_5.csv"

df_consumer_cyclical = pd.read_csv(filename)
df_consumer_cyclical["check_points"] = pd.to_datetime(df_consumer_cyclical["check_points"])

# draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC')
# draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
# df_consumer_cyclical = df_consumer_cyclical[(df_consumer_cyclical["check_points"] >= draw_figure_start_time) & (df_consumer_cyclical["check_points"] <= draw_figure_end_time)]

print(len(df_consumer_cyclical))

x_list = np.arange(0, len(df_consumer_cyclical))


for i in range(len(curve_names)):
    ax[2].plot(x_list, df_consumer_cyclical[curve_names[i]].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)





selected_check_points = [
    pd.Timestamp('2024-10-15 14:00:10.00', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.20', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.40', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.60', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:10.80', tz='UTC'),
    pd.Timestamp('2024-10-15 14:00:11.00', tz='UTC')
]

for a in range(3):
    ax[a].grid(True, alpha=1)
    ax[a].tick_params(axis='both', which='major', labelsize=14, pad=1.3)
    ax[a].set_xlabel("time percentage", fontsize=14, labelpad=0).set_position([0.38, 1])
    ax[a].tick_params(axis='x', pad=2)  # Adjust 'pad' to move the x-ticks closer to the axis.
    ax[a].tick_params(axis='y', pad=0)  # Adjust 'pad' to move the y-ticks closer to the axis.
    # ax[a].set_xticks(x_tick_indices)
    # x_tick_labels = [f'{str(tick).replace(".", "•")}' for tick in x_tick_labels]
    # ax[a].set_xticklabels(x_tick_labels, fontsize=14)  # Rotate for readability
    x_tick_indices = []
    x_tick_labels = []
    for cp in selected_check_points:
        if draw_figure_start_time <= cp <= draw_figure_end_time:
            # Calculate the offset in nanoseconds from `time_start_picture`
            offset_ns = (cp - draw_figure_start_time).total_seconds() * 1e9
            # Determine the proportional index within the range of `df`
            index = int((offset_ns / total_duration_ns) * len(df))
            x_tick_indices.append(index)
            x_tick_labels.append(cp.strftime('.%f')[:-4])  # Format as 'SS.ms'
    ax[a].set_xticks(x_tick_indices)
    x_tick_labels = [f'{str(tick).replace(".", "•")}' for tick in x_tick_labels]
    ax[a].set_xticklabels(x_tick_labels, fontsize=14)  # Rotate for readability


# ax[0].set_xlabel(f"time", fontsize=14)
# ax[1].set_xlabel(f"time", fontsize=14)
# ax[2].set_xlabel(f"time", fontsize=14)

# ax[0].set_ylabel("Accuracy", fontsize=14)
# ax[1].set_ylabel("Accuracy", fontsize=14)
# ax[2].set_ylabel("Accuracy", fontsize=14)
# plt.xticks(rotation=0)
plt.grid(True, axis='y')

handles, labels = [], []
h, l = ax.flat[-1].get_legend_handles_labels()
handles.extend(h)
labels.extend(l)

leg = ax[0].legend(loc="upper left", fontsize=14,
                bbox_to_anchor=(-0.2, 1.55), ncol=7,
                labelspacing=0.5, draggable=True,
                handletextpad=0.5, markerscale=8, handlelength=1.2, handleheight=1,
               columnspacing=0.5, borderpad=0.2, frameon=False, title=None)



plt.savefig(f"accuracy_remove_stocks_{get_integer(alpha)}.png")

# Show the plot
plt.show()


