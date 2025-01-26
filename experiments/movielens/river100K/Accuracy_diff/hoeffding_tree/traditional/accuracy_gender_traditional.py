import ast
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math

from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20


# # activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


#  all time:
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
print(data[:5])
alpha = 0.995
# 0.996^(24*7) = 0.5
# 0.998^(24*7*2) = 0.51 # 2 weeks
# 0.99993^(24*60*7) = 0.4938 # 1 week

threshold = 0.3
label_prediction = "prediction"
label_ground_truth = "rating"
correctness_column = "diff_binary_correctness"
use_two_counters = True
time_unit = "1 hour"
window_size_units = 1
checking_interval_units = 1



# Define a function to calculate accuracy
def calculate_accuracy(group):
    # print(len(group))
    correct = len(group[group[correctness_column] == 1])
    total = len(group)
    return correct / total if total > 0 else 0


window_size_list = ["1H", '1D', '10D', '1W', '2W', '1M', '3M', '6M', '1Y']
accuracy_dict = {}
window_size = ""
for w in window_size_list[:1]:
    window_size = w
    acc = data.groupby(pd.Grouper(key='datetime', freq=window_size)).apply(calculate_accuracy)
    accuracy_dict[window_size] = acc

accuracy_df = pd.DataFrame(accuracy_dict)

# print(accuracy_df)

accuracy_df.to_csv(f"movielens_compare_Accuracy_{method_name}_traditional_{window_size}.csv")



#
# # ################################################## draw the plot #####################################################
#
#import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys
import colormaps as cmaps

sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ 'lightsteelblue', 'cyan', '#57d357', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])

df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_traditional.csv")
print(df)




datetime = df['datetime']
all_col = df.columns.tolist()


max_length = len(datetime)
print("max_length", max_length)


for i in range(len(window_size_list[:1])):
    col = all_col[i+1]
    curve = list(df[col][df[col].notna()])
    proportional_x = np.linspace(0, max_length - 1, len(curve))
    print(len(curve), len(proportional_x))
    plt.plot(proportional_x, curve, linewidth=1.3, markersize=3.5,
                   label='{}'.format(window_size_list[i]), linestyle='-', marker='o', color=curve_colors[i])




# plt.xticks(np.arange(0, len(datetime)), [], rotation=0, fontsize=20)
plt.xlabel('time stamps',
           fontsize=20, labelpad=-2).set_position((0.47, 0.1))
plt.ylabel('Accuracy', fontsize=20, labelpad=-1)
plt.yscale("log")
plt.grid(True, axis='y')
plt.tight_layout()
plt.legend(loc='upper left', bbox_to_anchor=(0, 1.1), fontsize=15,
               ncol=4, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
               columnspacing=0.8, borderpad=0.2, frameon=True)
plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_{window_size}.png", bbox_inches='tight')
plt.show()