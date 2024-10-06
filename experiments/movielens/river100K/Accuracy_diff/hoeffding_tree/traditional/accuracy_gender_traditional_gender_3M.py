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
plt.rcParams['font.family'] = 'arial'
sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)

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


window_size_list = ['1D', '10D', '1W', '2W', '1M', '3M', '6M', '1Y']
accuracy_dict = {}

for w in window_size_list:
    window_size = w
    acc = data.groupby(["gender", pd.Grouper(key='datetime', freq=window_size)]).apply(calculate_accuracy)
    accuracy_dict[window_size] = acc

accuracy_df = pd.DataFrame(accuracy_dict)

print("accuracy_df")
print(accuracy_df)

accuracy_df.to_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender.csv")



#
# # ################################################## draw the plot #####################################################
#
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform


# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)



sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

#
#
# # Register the custom scale
# import matplotlib.scale as mscale
#
# mscale.register_scale(LogLinearLogScale)


fig, axs = plt.subplots(1, 1, figsize=(5, 2.5))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ 'lightsteelblue', 'blue', 'cyan', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])

df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender.csv")
print(df)




#
# max_length = len(datetime)
# print("max_length", max_length)

def plot_certain_time_window(window_size, axs):
    df_female = df[df["gender"] == 'F']
    df_female = df_female[df_female[window_size].notna()]
    df_female = df_female[["datetime", window_size]]

    print(df_female)

    df_male = df[df["gender"] == 'M']
    df_male = df_male[df_male[window_size].notna()]

    x_list = np.arange(0, len(df_male))

    from datetime import datetime
    datetime = df_male['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y')).tolist()

    window_size_list = df.columns.tolist()[2:]
    print("window_size_list", window_size_list)


    axs.grid(True)
    female_lst = df_female[window_size].dropna().tolist()
    male_lst = df_male[window_size].dropna().tolist()
    print(male_lst, female_lst)
    axs.plot(np.arange(len(male_lst)), male_lst, linewidth=1.4, markersize=3.5,
             label="male", linestyle='-', marker='o', color="blue")
    axs.plot(np.arange(len(female_lst)), female_lst, linewidth=1.4, markersize=3.5,
                label='female', linestyle='-', marker='o', color="orange")
    axs.legend(loc='lower right',
           bbox_to_anchor=(1.0, 0),  # Adjust this value (lower the second number)
           fontsize=14, ncol=1, labelspacing=0.2, handletextpad=0.5,
           markerscale=2, handlelength=2, columnspacing=0.6,
           borderpad=0.2, frameon=True)
    axs.set_xlabel('timestamps, window size = 1 month', fontsize=15)
    # fig.text(0.5, 0.04, 'normalized measuring time',
    #          ha='center', va='center', fontsize=16, fontweight='bold')

    decline_threshold = 0.07
    decline_points = []
    differences = []
    print(decline_points)
    for i in range(0, len(df_male)):
        if abs(female_lst[i - 1] - female_lst[i]) > decline_threshold:
            plt.axvline(x=i, color='black', linestyle=(0, (5, 5)), linewidth=1, alpha=0.9)
            # plt.text(i, y_margin, check_points[i].replace(" ", "\n"), color='red', fontsize=13,
            #          verticalalignment='bottom', horizontalalignment='center')
            decline_points.append(i)
            differences.append(female_lst[i - 1] - female_lst[i])

    print(decline_points, differences)
    print(datetime)

    # # Annotate the difference between the 4th and 5th points (manually picked)
    # x4, y4 = 3, female_lst[3]
    # x5, y5 = 4, female_lst[4]
    # distance = y4 - y5
    #
    # # Add annotation with the difference
    # axs.annotate('', xy=(x4, y4), xytext=(x4, y5),
    #               arrowprops=dict(arrowstyle='-', linestyle=':', color='black', lw=1.5),
    #              fontsize=12, ha='center')
    # axs.annotate('', xy=(2.5, y5), xytext=(4.5, y5),
    #               arrowprops=dict(arrowstyle='-', linestyle=":", color='black', lw=1.5),
    #              fontsize=12, ha='center')

    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=15)
    plt.ylabel('accuracy', fontsize=16)

    plt.xticks(np.arange(len(datetime)), datetime,
               color='black',
                rotation=0, fontsize=13)

    plt.axvline(x=1, color='black', linestyle=(0, (5, 5)), linewidth=1, alpha=0.9)
    axs.grid(axis='x')

    plt.text(2.5, 0.75, round(differences[0], 2), fontsize=11, va='bottom')


    plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_{window_size}.png", bbox_inches='tight')
    plt.show()


plot_certain_time_window('3M', axs)

