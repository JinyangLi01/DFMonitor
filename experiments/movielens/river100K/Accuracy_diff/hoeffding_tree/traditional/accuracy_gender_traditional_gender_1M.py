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

plt.rcParams['font.size'] = 20

#
# # ################################################## draw the plot #####################################################
#
import ast

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

window_size = "1M"
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

    window_size_list = df.columns.tolist()[2:]
    print("window_size_list", window_size_list)


    axs.grid(True)
    female_lst = df_female[value_col_name].dropna().tolist()
    male_lst = df_male[value_col_name].dropna().tolist()
    print(male_lst, female_lst)
    axs.plot(np.arange(len(male_lst)), male_lst, linewidth=2.5, markersize=3.5,
             label="male", linestyle='-', marker='o', color="blue")
    axs.plot(np.arange(len(female_lst)), female_lst, linewidth=2.5, markersize=3.5,
                label='female', linestyle='-', marker='o', color="orange")
    axs.legend(loc='lower right',
           bbox_to_anchor=(1.0, 0),  # Adjust this value (lower the second number)
           fontsize=12, ncol=1, labelspacing=0.2, handletextpad=0.5,
           markerscale=1, handlelength=2, columnspacing=0.6,
           borderpad=0.2, frameon=True)
    axs.set_xlabel('(d) window size = 1 month', fontsize=14, labelpad=-1, fontweight='bold').set_position([0.4, 0])
    # fig.text(0.5, 0.04, 'normalized measuring time',
    #          ha='center', va='center', fontsize=16, fontweight='bold')

    decline_threshold = 0.07
    decline_points = []
    differences = []
    print(decline_points)
    for i in range(0, len(df_female)):
        if abs(female_lst[i - 1] - female_lst[i]) > decline_threshold:
            plt.axvline(x=i, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
            # plt.text(i, y_margin, check_points[i].replace(" ", "\n"), color='red', fontsize=13,
            #          verticalalignment='bottom', horizontalalignment='center')
            decline_points.append(i)
            differences.append(female_lst[i - 1] - female_lst[i])

    print(decline_points, differences)
    print(datetime)

    # Annotate the difference between the 4th and 5th points (manually picked)
    x4, y4 = 3, female_lst[3]
    x5, y5 = 4, female_lst[4]
    distance = y4 - y5

    # Add annotation with the difference
    axs.annotate('', xy=(x4, y4), xytext=(x4, y5),
                  arrowprops=dict(arrowstyle='-', linestyle=':', color='black', lw=1.5),
                 fontsize=12, ha='center')
    axs.annotate('', xy=(2.5, y5), xytext=(4.5, y5),
                  arrowprops=dict(arrowstyle='-', linestyle=":", color='black', lw=1.5),
                 fontsize=12, ha='center')

    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=13)
    plt.ylabel('accuracy', fontsize=14, labelpad=0)

    plt.xticks(decline_points+[1], [datetime[decline_points[i]].replace(" ", "\n") for i in range(len(decline_points))] + [datetime[1].replace(" ", "\n")],
               color='black',
                rotation=0, fontsize=12)

    plt.axvline(x=1, color='black', linestyle=(0, (5, 5)), linewidth=1.5, alpha=1)
    axs.grid(axis='x')

    plt.text(2.3, 0.75, round(differences[0], 2), fontsize=12, va='bottom')
    plt.tick_params(axis='x', pad=1)
    plt.tick_params(axis='y', pad=1)

    plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender_{window_size}.png", bbox_inches='tight')
    plt.show()


plot_certain_time_window('1M', axs)

