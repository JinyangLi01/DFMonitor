import ast
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from polars.dependencies import subprocess
import matplotlib.transforms as mtransforms

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)




window_size_unit_list = ['1', '24', '24*7', '24*7*4', '24*7*4*2', '24*7*4*3']

result_file_list = dict()
alpha = 993

for window_size in window_size_unit_list:
    file_name = f"movielens_compare_Counters_hoeffding_classifier_gender_per_item_alpha_{alpha}_time_unit_1 hour*{window_size}_check_interval_1 hour.csv"
    # if file exists
    if not os.path.exists(file_name):
        subprocess.run(["python3", "accuracy_gender_per_item_1hour.py", window_size])
        # Check again after running the script
        if not os.path.exists(file_name):
            raise Exception(f"Error: {file_name} was not created by the script.")
    df = pd.read_csv(file_name)
    print("window_size", window_size, df)
    result_file_list[window_size] = df



fig, axs = plt.subplots(1, 1, figsize=(1, 0.85))
plt.subplots_adjust(left=0, right=0.99, top=0.85, bottom=0.0, wspace=0.3, hspace=0)
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ '#0000ff', 'cyan', '#57d357', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])



pair_colors = ["blue", "darkorange"]

window_size = window_size_unit_list[1]
df = result_file_list[window_size]
df.plot()


plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)

plt.savefig(f"movielens_compare_Counters_window_size_{window_size}.png", bbox_inches='tight')
plt.show()



