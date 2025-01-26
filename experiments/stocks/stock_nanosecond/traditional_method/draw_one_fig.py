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


time_window_str = "10us"
accuracy_per_1s_per_group = pd.read_csv(f"traditional_accuracy_time_window_{time_window_str}.csv")


# Set up the plot
fig, ax = plt.subplots(figsize=(4, 2.3))

# Use seaborn to plot each sector's trend line for accuracy over time
sns.lineplot(data=accuracy_per_1s_per_group, x="ts_event_"+time_window_str, y="accuracy", hue="sector", marker="o")

# Set titles and labels
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


