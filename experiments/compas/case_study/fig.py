import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from algorithm import FPR_workload as workload
import seaborn as sns
from matplotlib import rc
from algorithm import config
import colorsys
import csv
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



def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


class LogLinearLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, linthresh, A, **kwargs):
        super().__init__(**kwargs)
        self.linthresh = linthresh
        self.A = A

    def transform_non_affine(self, a):
        # Adjust transformation to center linear region around y=A
        with np.errstate(divide='ignore', invalid='ignore'):
            # Example adjusted transformation logic; you will need to refine this
            return np.where(np.abs(a - self.A) <= self.linthresh, a,
                            np.sign(a - self.A) * np.log10(np.abs(a - self.A) + 1) + self.A)

    def inverted(self):
        return InvertedLogLinearLogTransform(self.linthresh, self.A)


class InvertedLogLinearLogTransform(LogLinearLogTransform):
    def transform_non_affine(self, a):
        # Adjust inverse transformation logic accordingly
        # This is a placeholder; you need to define the exact inverse math
        return np.where(np.abs(a - self.A) <= self.linthresh, a,
                        np.sign(a - self.A) * (np.exp(np.abs(a - self.A)) - 1) + self.A)


class LogLinearLogScale(ScaleBase):
    name = 'loglinearlog'

    def __init__(self, axis, *, linthresh=1, A=1, **kwargs):
        super().__init__(axis)
        self.linthresh = linthresh
        self.A = A

    def get_transform(self):
        return LogLinearLogTransform(self.linthresh, self.A)

    def set_default_locators_and_formatters(self, axis):
        # Setup locators and formatters as needed
        pass


# Register the custom scale
import matplotlib.scale as mscale

mscale.register_scale(LogLinearLogScale)


# Define thresholds
# linear_threshold # Transition to log after this value
# log_linear_transition  # Transition back to linear after this value
def get_y_transformed(y, linear_threshold, log_linear_transition):
    # Apply transformation
    y_transformed = np.zeros_like(y)
    for i, value in enumerate(y):
        if value <= linear_threshold:
            y_transformed[i] = value
        elif value <= log_linear_transition:
            y_transformed[i] = linear_threshold + np.log(value - linear_threshold + 1)
        else:
            # Adjust the offset to smoothly transition back to linear
            log_offset = linear_threshold + np.log(log_linear_transition - linear_threshold + 1)
            y_transformed[i] = log_offset + (value - log_linear_transition)
    return y_transformed


# read FPR
df_FPR = pd.read_csv("case_study_FPR.csv", sep=",")

# print(df_FPR.columns)

col_data_FPR = {}

for col_name in df_FPR.columns:
    col_data_FPR[col_name] = df_FPR[col_name].tolist()

# read CR
df_CR = pd.read_csv("case_study_CR.csv", sep=",")
# print(df_CR.head)
col_data_CR = {}
for col_name in df_CR.columns:
    col_data_CR[col_name] = df_CR[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(df_CR))

# print(len(x_list))

fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])


pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']

# Plot the first curve (y1_values)
axs[0].plot(x_list, col_data_FPR["black_time_decay"], linewidth=2.5, markersize=3.5, label='Black time decay',
            linestyle='-', marker='o', color=pair_colors[0])

# Plot the second curve (y2_values)
axs[0].plot(x_list, col_data_FPR["black_traditional"], linewidth=2.5, markersize=3.5, label='Black traditional',
            linestyle='--', marker='s', color=pair_colors[1])

axs[0].plot(x_list, col_data_FPR["white_time_decay"], linewidth=2.5, markersize=3.5, label='White time decay',
            linestyle='-', marker='o', color=pair_colors[2])
axs[0].plot(x_list, col_data_FPR["white_traditional"], linewidth=2.5, markersize=3.5, label='White traditional',
            linestyle='--', marker='s', color=pair_colors[3])

axs[0].plot(x_list, col_data_FPR["hispanic_time_decay"], linewidth=2.5, markersize=3.5, label='Hispanic time decay',
            linestyle='-', marker='o',
            color=pair_colors[4])
axs[0].plot(x_list, col_data_FPR["hispanic_traditional"], linewidth=2.5, markersize=3.5, label='Hispanic traditional',
            linestyle='--', marker='s',
            color=pair_colors[5])


axs[1].plot(x_list, col_data_CR["black_time_decay"], linewidth=2.5, markersize=3.5, label='Black time decay',
            linestyle='-', marker='o', color=pair_colors[0])
axs[1].plot(x_list, col_data_CR["black_traditional"], linewidth=2.5, markersize=3.5, label='Black traditional',
            linestyle='--', marker='s', color=pair_colors[1])
axs[1].plot(x_list, col_data_CR["white_time_decay"], linewidth=2.5, markersize=3.5, label='White time decay',
            linestyle='-', marker='o', color=pair_colors[2])
axs[1].plot(x_list, col_data_CR["white_traditional"], linewidth=2.5, markersize=3.5, label='White traditional',
            linestyle='--', marker='s', color=pair_colors[3])
axs[1].plot(x_list, col_data_CR["hispanic_time_decay"], linewidth=2.5, markersize=3.5, label='Hispanic time decay',
            linestyle='-', marker='o', color=pair_colors[4])
axs[1].plot(x_list, col_data_CR["hispanic_traditional"], linewidth=2.5, markersize=3.5, label='Hispanic traditional',
            linestyle='--', marker='s', color=pair_colors[5])



# add a common x-axis label
fig.text(0.5, -0.03, 'compas screening date, from 01/01/2013 \nto 12/31/2014, time window = 1 month',
            ha='center', va='center', fontsize=16, fontweight='bold')

axs[0].set_ylabel('false positive rate', fontsize=17, labelpad=-1, fontweight='bold')
axs[1].set_ylabel('coverage rate', fontsize=17, labelpad=-1, fontweight='bold')
axs[0].set_yticks([0.0, 0.5, 1.0])
axs[1].set_yticks([0.0, 0.2, 0.4, 0.6])

axs[0].set_title('(a) FPR', y=-0.19, pad=-0.5, fontweight='bold')
axs[1].set_title('(b) CR', y=-0.19, pad=-0.5, fontweight='bold')
axs[0].set_yscale('log')
axs[0].set_yticks([0.1, 0.5, 1.0], [0.1, 0.5, 1.0])
# axs[1].set_yscale('log')

axs[0].grid(True)
axs[1].grid(True)
axs[0].set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
axs[1].set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)


# create a common legend
handles, labels = axs[-1].get_legend_handles_labels()


fig.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0.01, 0.96), fontsize=12.8,
           ncol=3, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
           columnspacing=0.4, borderpad=0.2, frameon=True)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0)

#
# plt.legend(loc='upper left', bbox_to_anchor=(-1, 1.1), fontsize=14,
#            ncol=2, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
#            columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)
#
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0)
plt.tight_layout()
plt.savefig("compas_compare.png", bbox_inches='tight')
plt.show()
