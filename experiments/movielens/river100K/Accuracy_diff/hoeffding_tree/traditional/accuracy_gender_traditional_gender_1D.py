import ast
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math

from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns

#
# # sns.set_style("whitegrid")
# plt.rcParams['font.family'] = 'arial'
# sns.set_palette("Paired")
# sns.set_context("paper", font_scale=1.6)
#
# # Set the font size for labels, legends, and titles
#
# plt.figure(figsize=(6, 3.5))
# plt.rcParams['font.size'] = 20
#
#
# # # activate latex text rendering
# # rc('text', usetex=True)
# # rc('axes', linewidth=2)
# # rc('font', weight='bold')
#
#
# def scale_lightness(rgb, scale_l):
#     # convert rgb to hls
#     h, l, s = colorsys.rgb_to_hls(*rgb)
#     # manipulate h, l, s values and return as rgb
#     return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
#
# #  all time:
# method_name = "hoeffding_classifier"
# data = pd.read_csv('../../../result_' + method_name + '.csv', dtype={"zip_code": str})
# print(data["gender"].unique())
# date_column = "datetime"
# # get distribution of compas_screening_date
# data[date_column] = pd.to_datetime(data[date_column])
# print(data[date_column].min(), data[date_column].max())
# date_time_format = True
# # time_window_str = "1 month"
# monitored_groups = [{"gender": 'M'}, {"gender": 'F'}]
# print(data[:5])
#
# threshold = 0.3
# label_prediction = "prediction"
# label_ground_truth = "rating"
# correctness_column = "diff_binary_correctness"
# use_two_counters = True
# time_unit = "1 hour"
# window_size_units = 1
# checking_interval_units = 1
#
#
#
# # Define a function to calculate accuracy
# def calculate_accuracy(group):
#     # print(len(group))
#     correct = len(group[group[correctness_column] == 1])
#     total = len(group)
#     return correct / total if total > 0 else 0
#
#
# window_size_list = ['1D', '10D', '1W', '2W', '1M', '3M', '6M', '1Y']
# accuracy_dict = {}
#
# for w in window_size_list:
#     window_size = w
#     acc = data.groupby(["gender", pd.Grouper(key='datetime', freq=window_size)]).apply(calculate_accuracy)
#     accuracy_dict[window_size] = acc
#
# accuracy_df = pd.DataFrame(accuracy_dict)
#
# print("accuracy_df")
# print(accuracy_df)
#
# accuracy_df.to_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender.csv")
#
#

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


fig, axs = plt.subplots(1, 1, figsize=(3.5, 2))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
                                          'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ 'lightsteelblue', 'blue', 'cyan', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])


window_size = "1D"
method_name = "hoeffding_classifier"
df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender_{window_size}.csv")
print(df)
value_col_name = "calculated_value"





#
# max_length = len(datetime)
# print("max_length", max_length)

def plot_certain_time_window(window_size, axs):
    df_female = df[df["gender"] == 'F']
    # df_female = df_female[df_female[window_size].notna()]
    df_female = df_female[["datetime", value_col_name]]

    df_male = df[df["gender"] == 'M']
    # df_male = df_male[df_male[window_size].notna()]
    df_male = df_male[["datetime", value_col_name]]

    print("df_male: \n", df_male)
    print("\ndf_female: \n", df_female)

    x_list = np.arange(0, len(df_male))

    from datetime import datetime
    datetime = df_male['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%m/%d/%Y')).tolist()

    # window_size_list = df.columns.tolist()[2:]
    # print("window_size_list", window_size_list)


    axs.grid(True)
    female_lst = df_female[value_col_name].dropna().tolist()
    male_lst = df_male[value_col_name].dropna().tolist()
    print(male_lst, female_lst)
    axs.plot(np.arange(len(male_lst)), male_lst, linewidth=2.5, markersize=4,
             label="male", linestyle='-', marker='o', color="blue")
    axs.plot(np.arange(len(female_lst)), female_lst, linewidth=2.5, markersize=3.5,
                label='female', linestyle='-', marker='o', color="orange")
    axs.legend(loc='lower left',
           bbox_to_anchor=(-0.15, 0.08),  # Adjust this value (lower the second number)
           fontsize=12, ncol=1, labelspacing=0.2, handletextpad=0.5,
           markerscale=1, handlelength=1, columnspacing=0.6,
           borderpad=0.2, frameon=True)
    axs.set_xlabel('window size = 1 day', fontsize=14, labelpad=0).set_position([0.38, 0])

    ticks = []
    idx = datetime.index('11/02/1997')
    ticks.append(idx)
    plt.axvline(x=idx, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
    idx = datetime.index('01/25/1998')
    ticks.append(idx)
    plt.axvline(x=idx, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
    idx = datetime.index('03/22/1998')
    ticks.append(idx)
    plt.axvline(x=idx, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)

    plt.xticks(ticks, ['11/02/1997     ', '01/25/1998      ', '      03/22/1998'], fontsize=12)

    # Annotate the difference between the 4th and 5th points (manually picked)
    x = []
    y = []
    for i in range(0, len(female_lst)):
        x.append(i)
        y.append(female_lst[i])

    plt.ylabel('accuracy', fontsize=14, labelpad=0)
    plt.yticks(fontsize=13)
    axs.grid(axis='x')
    plt.tick_params(axis='x', pad=1)
    plt.tick_params(axis='y', pad=1)

    plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_{window_size}.png", bbox_inches='tight')
    plt.show()


plot_certain_time_window('1D', axs)

