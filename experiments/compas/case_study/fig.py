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

fig, axs = plt.subplots(1, 2, figsize=(6, 2.55))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])


pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']


axs[0].plot(x_list, col_data_FPR["hispanic_time_decay"], linewidth=2.5, markersize=3.5, label='Hispanic time decay',
            linestyle='-', marker='o',
            color=pair_colors[4])
axs[0].plot(x_list, col_data_FPR["hispanic_traditional"], linewidth=2.5, markersize=3.5, label='Hispanic traditional',
            linestyle='--', marker='s',
            color=pair_colors[5])
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
fig.text(0.5, 0.01, 'compas screening date, from 01/01/2013 \nto 12/31/2014, time window = 1 month',
            ha='center', va='center', fontsize=16, fontweight='bold')

axs[0].set_ylabel('false positive rate', fontsize=17, labelpad=-1, fontweight='bold', y=0.46)
axs[0].set_title('(a) FPR', y=-0.3, pad=0, fontweight='bold')
axs[0].set_yscale('log')
axs[0].set_yticks([0.1, 0.3, 0.5, 1.0], [0.1, 0.3, 0.5, 1.0])
x_ticks = np.arange(0, len(x_list))
x_labels = [""] * len(x_list)
x_labels[5] = "6"
x_labels[11] = "12"
x_labels[15] = "16"
axs[0].set_xticks(x_ticks, x_labels, rotation=0, fontsize=14)
axs[0].grid(True)

gridlines = axs[0].xaxis.get_gridlines()
gridlines[5].set_color("k")
gridlines[5].set_linewidth(2)
gridlines[11].set_color("k")
gridlines[11].set_linewidth(2)
gridlines[15].set_color("k")
gridlines[15].set_linewidth(2)
#
# xt=np.append(x_ticks,5)
# xtl = xt.tolist()
# xtl[-1] = "6"
# axs[0].set_xticklabels(xtl)



axs[1].set_ylabel('coverage rate', fontsize=17, labelpad=-1, fontweight='bold')
axs[1].set_yticks([0.0, 0.2, 0.3, 0.4, 0.6])
axs[1].set_title('(b) CR', y=-0.27, pad=-0.5, fontweight='bold')
axs[1].grid(True)
axs[1].set_xticks(x_ticks, [], rotation=0, fontsize=20)


# create a common legend
handles, labels = axs[-1].get_legend_handles_labels()


fig.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0.01, 0.96), fontsize=12.8,
           ncol=3, labelspacing=0.3, handletextpad=0.3, markerscale=2,
           columnspacing=0.4, borderpad=0.2, frameon=True)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0)


plt.tight_layout()
plt.savefig("compas_compare.png", bbox_inches='tight')
plt.show()


####################### smoothness: standard deviation of the first derative of the curve ############################

smooth_FPR = {}
smooth_CR = {}



# open a new file for FPR
with open("roughness_compas_FPR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness"])
    writer.writerow(["black_time_decay", np.std(np.diff(col_data_FPR["black_time_decay"]) / np.diff(x_list))])
    smooth_FPR["black_time_decay"] = np.std(np.diff(col_data_FPR["black_time_decay"]) / np.diff(x_list))
    # remove nan value in list
    fpr = col_data_FPR["black_traditional"]
    fpr = [x for x in fpr if str(x) != 'nan']
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    writer.writerow(["black_traditional", sm])
    smooth_FPR["black_traditional"] = sm
    writer.writerow(["white_time_decay", np.std(np.diff(col_data_FPR["white_time_decay"]) / np.diff(x_list))])
    smooth_FPR["white_time_decay"] = np.std(np.diff(col_data_FPR["white_time_decay"]) / np.diff(x_list))
    fpr = col_data_FPR["white_traditional"]
    fpr = [x for x in fpr if str(x) != 'nan']
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    writer.writerow(["white_traditional", sm])
    smooth_FPR["white_traditional"] = sm
    writer.writerow(["hispanic_time_decay", np.std(np.diff(col_data_FPR["hispanic_time_decay"]) / np.diff(x_list))])
    smooth_FPR["hispanic_time_decay"] = np.std(np.diff(col_data_FPR["hispanic_time_decay"]) / np.diff(x_list))
    fpr = col_data_FPR["hispanic_traditional"]
    fpr = [x for x in fpr if str(x) != 'nan']
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    writer.writerow(["hispanic_traditional", sm])
    smooth_FPR["hispanic_traditional"] = sm

smooth_CR["black_time_decay"] = np.std(np.diff(col_data_CR["black_time_decay"]) / np.diff(x_list))
smooth_CR["black_traditional"] = np.std(np.diff(col_data_CR["black_traditional"]) / np.diff(x_list))
smooth_CR["white_time_decay"] = np.std(np.diff(col_data_CR["white_time_decay"]) / np.diff(x_list))
smooth_CR["white_traditional"] = np.std(np.diff(col_data_CR["white_traditional"]) / np.diff(x_list))
smooth_CR["hispanic_time_decay"] = np.std(np.diff(col_data_CR["hispanic_time_decay"]) / np.diff(x_list))
smooth_CR["hispanic_traditional"] = np.std(np.diff(col_data_CR["hispanic_traditional"]) / np.diff(x_list))

# another file for CR
with open("roughness_compas_CR.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness"])
    writer.writerow(["black_time_decay", np.std(np.diff(col_data_CR["black_time_decay"]) / np.diff(x_list))])
    writer.writerow(["black_traditional", np.std(np.diff(col_data_CR["black_traditional"]) / np.diff(x_list))])
    writer.writerow(["white_time_decay", np.std(np.diff(col_data_CR["white_time_decay"]) / np.diff(x_list))])
    writer.writerow(["white_traditional", np.std(np.diff(col_data_CR["white_traditional"]) / np.diff(x_list))])
    writer.writerow(["hispanic_time_decay", np.std(np.diff(col_data_CR["hispanic_time_decay"]) / np.diff(x_list))])
    writer.writerow(["hispanic_traditional", np.std(np.diff(col_data_CR["hispanic_traditional"]) / np.diff(x_list))])



from sklearn.preprocessing import MinMaxScaler


roughness_scores_FPR = [1 / sd for sd in list(smooth_FPR.values())]
roughness_scores_CR = [1 / sd for sd in list(smooth_CR.values())]

# Assuming roughness_scores is a 2D array where each row is a feature to be normalized
roughness_scores = np.array(roughness_scores_FPR).reshape(-1, 1)  # Reshape for scaler if it's a single feature

scaler = MinMaxScaler(feature_range=(0, 1))
smoothness_scores_normalized_FPR = scaler.fit_transform(roughness_scores).flatten()


smoothness_scores = [1/sd for sd in list(smooth_FPR.values())]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores with Scikit-learn:", smoothness_scores_normalized_FPR)



roughness_scores = np.array(roughness_scores_CR).reshape(-1, 1)  # Reshape for scaler if it's a single feature
smoothness_scores_normalized_CR = scaler.fit_transform(roughness_scores).flatten()


smoothness_scores = [1/sd for sd in list(smooth_CR.values())]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_CR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores with Scikit-learn:", smoothness_scores_normalized_CR)



# draw two bar charts using FPR and CR smoothness
fig, axs = plt.subplots(1, 2, figsize=(5.8, 2))

legend_labels = ["Black time decay", "Black traditional", "White time decay", "White traditional", "Hispanic time decay", "Hispanic traditional"]

axs[0].bar(smooth_FPR.keys(), smoothness_scores_normalized_FPR, color=pair_colors, label=legend_labels)
axs[0].set_ylabel('smoothness', fontsize=17, labelpad=-1, fontweight='bold')
axs[0].set_title('(a) FPR', y=-0.13, pad=0, fontweight='bold')
axs[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
axs[0].grid(True)
# remove x ticks
axs[0].set_xticks([], [], rotation=0, fontsize=20)
axs[0].bar_label(axs[0].containers[0], labels=["{:0.2f}".format(score) for score in smoothness_scores_normalized_FPR],
                 fontsize=14, padding=-1.2)


axs[1].bar(smooth_CR.keys(), smoothness_scores_normalized_CR, color=pair_colors, label=legend_labels)
axs[1].set_ylabel('smoothness', fontsize=17, labelpad=-1, fontweight='bold')
axs[1].set_title('(b) CR', y=-0.13, pad=0, fontweight='bold')
axs[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
axs[1].grid(True)
axs[1].set_xticks([], [], rotation=0, fontsize=20)
axs[1].bar_label(axs[1].containers[0], labels=["{:0.2f}".format(score) for score in smoothness_scores_normalized_CR],
                 fontsize=14, padding=-1.2)


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(-0.07, 0.95), fontsize=13,
              ncol=3, labelspacing=0.3, handletextpad=0.3, markerscale=2, handlelength=1.9,
              columnspacing=0.5, borderpad=0.2, frameon=True)
# fig.legend()
plt.tight_layout()
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0)
fig.savefig("compas_smoothness_normalized.png", bbox_inches='tight')
plt.show()