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
date_column = 'datetime'
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
# use_two_counters = True
# time_unit = "1 hour"
# window_size_units = 1
# checking_interval_units = 1



# Define a function to calculate accuracy
def calculate_accuracy(group):
    correct = len(group[group[correctness_column] == 1])
    total = len(group)
    return correct / total if total > 0 else 0


window_size_list = ['1D', '3D', '5D', '10D', '1W', '2W', '1M', '3M', '6M', '1Y']
accuracy_dict = {}

for window_size in window_size_list:
    acc = data.groupby(["gender", pd.Grouper(key="datetime", freq=window_size, dropna=False)])[["gender", "datetime", "diff_binary_correctness"]].apply(calculate_accuracy)

    # Manually construct a DataFrame, retaining original columns
    acc_df = pd.DataFrame({
        'gender': acc.index.get_level_values(0),
        'datetime': acc.index.get_level_values(1),
        'calculated_value': acc.values  # Replace with appropriate column name
    })
    # Fill in the last partial period if missing
    if data['datetime'].max() not in acc_df['datetime'].values:
        # Update the last row with the calculation results of partial data
        partial_period = data[(data['datetime'] > acc_df['datetime'].max())]
        if not partial_period.empty:
            acc_df.loc[acc_df.index[-1], 'calculated_value'] = calculate_accuracy(partial_period)

    print(acc_df)
    print(acc_df[len(acc_df) - 5:])

    # Store in the dictionary
    accuracy_dict[window_size] = acc_df
#
# print("================== accuracy_dict ==================\n")
# print(accuracy_dict)

# Create a DataFrame to compare accuracies for different window sizes
accuracy_df = pd.DataFrame(accuracy_dict[window_size_list[0]])
accuracy_df.rename(columns={"calculated_value": window_size_list[0]}, inplace=True)
print(accuracy_df[len(accuracy_df)-5:])
for window_size in window_size_list[1:]:
    df2 = accuracy_dict[window_size]
    print("df2: ", df2[len(df2)-5:])
    merged_df = pd.merge(accuracy_df, df2, on=['gender', 'datetime'], how='left')
    accuracy_df = merged_df
    accuracy_df.rename(columns={"calculated_value": window_size}, inplace=True)
    if pd.isna(accuracy_df.loc[len(accuracy_df)-1, window_size]):
        print(accuracy_df.loc[len(accuracy_df)-1, "datetime"])
        accuracy_df.loc[len(accuracy_df)-1, window_size] = accuracy_dict[window_size].loc[len(accuracy_dict[window_size])-1, "calculated_value"]
    print(f"=========================== accuracy_df {window_size} ==========================")
    print(accuracy_df)


accuracy_df.to_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender.csv")

#
#
# #
# # # ################################################## draw the plot #####################################################
# #
# import ast
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.scale import ScaleBase
# from matplotlib.transforms import Transform
#
#
# # Enable LaTeX rendering
# # plt.rcParams['text.usetex'] = True
# # plt.rcParams['font.family'] = 'CMU.Serif'
# # plt.rcParams['font.serif'] = 'Computer Modern'
#
# # Set the global font family to serif
# plt.rcParams['font.family'] = 'arial'
#
# sns.set_palette("Paired")
# sns.set_context("paper", font_scale=1.6)
#
# # sns.set_theme(font='CMU Serif', style='darkgrid')
#
# class LogLinearLogTransform(Transform):
#     input_dims = output_dims = 1
#
#     def __init__(self, linthresh, A, **kwargs):
#         super().__init__(**kwargs)
#         self.linthresh = linthresh
#         self.A = A
#
#     def transform_non_affine(self, a):
#         # Adjust transformation to center linear region around y=A
#         with np.errstate(divide='ignore', invalid='ignore'):
#             # Example adjusted transformation logic; you will need to refine this
#             return np.where(np.abs(a - self.A) <= self.linthresh, a,
#                             np.sign(a - self.A) * np.log10(np.abs(a - self.A) + 1) + self.A)
#
#     def inverted(self):
#         return InvertedLogLinearLogTransform(self.linthresh, self.A)
#
#
# class InvertedLogLinearLogTransform(LogLinearLogTransform):
#     def transform_non_affine(self, a):
#         # Adjust inverse transformation logic accordingly
#         # This is a placeholder; you need to define the exact inverse math
#         return np.where(np.abs(a - self.A) <= self.linthresh, a,
#                         np.sign(a - self.A) * (np.exp(np.abs(a - self.A)) - 1) + self.A)
#
#
# class LogLinearLogScale(ScaleBase):
#     name = 'loglinearlog'
#
#     def __init__(self, axis, *, linthresh=1, A=1, **kwargs):
#         super().__init__(axis)
#         self.linthresh = linthresh
#         self.A = A
#
#     def get_transform(self):
#         return LogLinearLogTransform(self.linthresh, self.A)
#
#     def set_default_locators_and_formatters(self, axis):
#         # Setup locators and formatters as needed
#         pass
#
#
# sns.set_palette("Paired")
# sns.set_context("paper", font_scale=2)
#
# #
# #
# # # Register the custom scale
# # import matplotlib.scale as mscale
# #
# # mscale.register_scale(LogLinearLogScale)
#
#
# fig, axs = plt.subplots(1, 2, figsize=(5, 3))
# plt.rcParams['font.size'] = 10
# curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
#                                           '#41ab5d', '#006837'])
# curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
#                                           'magenta', 'cyan'])
# curve_colors = sns.color_palette(palette=[ 'lightsteelblue', 'blue', 'cyan', '#004b00', 'darkorange', 'firebrick',
#                                            'blueviolet', 'magenta'])
#
# df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_traditional_gender.csv")
# print(df)
# df = df[["gender", "datetime", "1D", "5D", "1W", "2W", "1M", "3M", "6M", "1Y"]]
#
# df_female = df[df["gender"] == 'F']
# df_male = df[df["gender"] == 'M']
# datetime = df['datetime']
# window_size_list = df.columns.tolist()[2:]
# print("window_size_list", window_size_list)
#
#
#
#
# x_list = np.arange(0, len(df_male))
#
#
# max_length = len(datetime)
# print("max_length", max_length)
#
#
# for i in range(len(window_size_list)):
#     col = window_size_list[i]
#
#     curve = list(df_male[col][df_male[col].notna()])
#     proportional_x = np.linspace(0, max_length - 1, len(curve))
#     print(len(curve), len(proportional_x))
#     axs[0].plot(proportional_x, curve, linewidth=1.4, markersize=3.5,
#                 label='{}'.format(window_size_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
#     curve = list(df_female[col][df_female[col].notna()])
#     proportional_x = np.linspace(0, max_length - 1, len(curve))
#     print("curve ", curve)
#     axs[1].plot(proportional_x, curve, linewidth=1.4, markersize=3.5,
#                 label='{}'.format(window_size_list[i]), linestyle='-', marker='o', color=curve_colors[i])
#
#
#
# axs[0].set_title('(a) male', y=-0.19, pad=-0.5)
# axs[1].set_title('(d) female', y=-0.19, pad=-0.5)
#
#
# for ax in axs:
#     ax.grid(False)
#     ax.set_xticks(np.arange(0, max_length), [], rotation=0, fontsize=20)
#     ax.set_ylabel('accuracy', fontsize=14, fontweight='bold', labelpad=-10)
#
#
#
# import matplotlib.ticker as ticker
# from matplotlib.ticker import FormatStrFormatter
# axs[0].set_yscale('symlog', linthresh=0.95)
# axs[0].set_xticks([], [], fontsize=15)
# axs[0].set_ylim(0.4, 1.0)
# axs[0].set_yticks([0.4, 1.0], ["0.4", "1.0"], fontsize=15)
#
# # axs[1].set_yticks([0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0], fontsize=15)
# axs[1].set_yscale('symlog', linthresh=0.97)
# axs[1].set_xticks([], [], fontsize=15)
# axs[1].set_ylim(0.4, 1.0)
# axs[1].set_yticks([0.4, 1.0], ["0.4", "1.0"], fontsize=15)
#
#
#
# fig.text(0.5, 0.04, 'normalized measuring time',
#          ha='center', va='center', fontsize=16, fontweight='bold')
#
# # handles = []
# # labels = []
# # for i in range(0, 2):
# #     handle, label = axs[i].get_legend_handles_labels()
# #     handles.append(handle)
# #     labels.append(label)
#
# handles, labels = axs[1].get_legend_handles_labels()
# plt.subplots_adjust(left=0.1, right=0.95, top=0.7, bottom=0.2, wspace=0.3, hspace=0.3)
#
# fig.legend(title='time window', title_fontsize=14, loc='lower left',
#            bbox_to_anchor=(0.1, 0.7),  # Adjust this value (lower the second number)
#            fontsize=14, ncol=4, labelspacing=0.2, handletextpad=0.5,
#            markerscale=2, handlelength=2, columnspacing=0.6,
#            borderpad=0.2, frameon=True, handles=handles, labels=labels)
#
#
# plt.savefig(f"Acc_hoeffding_timedecay_traditional_gender.png", bbox_inches='tight')
# plt.show()



