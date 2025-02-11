import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import colorsys
import csv

from anyio import value
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





fig, axs = plt.subplots(2, 2, figsize=(4.7, 2.7), gridspec_kw={'width_ratios': [3.8, 2]})
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0, wspace=0.25, hspace=0.5)



checking_interval = "1year"
curve_colors = ["blue", "red", "green",  "hotpink", "DarkGrey", "darkorchid",  "Lime", "cyan", "gold",]



######################################## Dynamic fixed window ########################################



df_fixed = pd.read_csv(f"fixed_window_Accuracy_{checking_interval}.csv")

x_list = np.arange(len(df_fixed))
axs[0][0].plot(x_list, df_fixed['male_time_decay'], linewidth=1, markersize=1, label = 'male fixed window',
            linestyle='-', marker='o',
            color=curve_colors[0])
axs[1][0].plot(x_list, df_fixed['female_time_decay'], linewidth=1, markersize=1, label = 'female fixed window',
            linestyle='-', marker='o',
            color=curve_colors[1])



######################################## Dynamic adaptive window ########################################

Tb = 24*7
Tin = 24

df_adaptive = pd.read_csv(f"adaptive_window_Accuracy_{checking_interval}.csv")
x_list = np.arange(len(df_adaptive))
axs[0][0].plot(x_list, df_adaptive["male_time_decay"], linewidth=1, markersize=1, label='male adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[3])
axs[1][0].plot(x_list, df_adaptive["female_time_decay"], linewidth=1, markersize=1, label='female adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[4])



######################################## traditional ########################################

df_traditional = pd.read_csv(f"traditional_Accuracy_{checking_interval}.csv")
x_list = np.arange(len(df_traditional))
male_tra_df = df_traditional["male_traditional"]
female_tra_df = df_traditional["female_traditional"]

axs[0][0].plot(x_list, df_traditional["male_traditional"], linewidth=1, markersize=1, label='male traditional',
            linestyle='--', marker='s', color=curve_colors[6])
axs[1][0].plot(x_list, df_traditional["female_traditional"], linewidth=1, markersize=1, label='female traditional',
            linestyle='--', marker='s', color=curve_colors[7])


axs[0][0].set_ylim(0.86, 0.98)
# axs[0][0].set_yscale('symlog', linthresh=8)
axs[0][0].set_title('(a) Male Accuracy', y=-0.15, pad=-10, fontsize=13)
axs[0][0].set_yticks([0.86, 0.9, 0.94, 0.97], [0.86, 0.9, 0.94, 0.97], fontsize=13)
axs[0][0].set_xticks(range(len(male_tra_df)), [""]*len(male_tra_df), rotation=0, fontsize=13)

axs[0][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][0].grid(True, axis="y", linestyle='--', alpha=0.6)
# axs[0][0].text(12, -0.21, 'screening date', ha='center', va='center', fontsize=12)


# axs[1][0].set_ylabel('Accuracy', fontsize=14, labelpad=-1)
axs[1][0].set_ylim(0.9, 0.97)
# axs[1][0].set_yscale('symlog', linthresh=8)
# axs[1][0].set_yscale('log')
axs[1][0].set_title('(c) female Accuracy', y=-0.15, pad=-10, fontsize=13)
axs[1][0].set_yticks([0.9, 0.94, 0.97], [0.9, 0.94, 0.97], fontsize=13)
axs[1][0].set_xticks([], [], rotation=0, fontsize=13)
axs[1][0].set_xticks(range(len(female_tra_df)), [""]*len(female_tra_df), rotation=0, fontsize=13)
axs[1][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][0].grid(True, axis="x", linestyle='--', alpha=0.6)
# axs[1][0].text(12, -0.17, 'screening date', ha='center', va='center', fontsize=12)
#

######################################## smoothness ########################################

def roughness2smoothness(roughness_Accuracy):
    print("Roughness: ", roughness_Accuracy)

    # Scale roughness to smoothness
    max_roughness = max(roughness_Accuracy.values())
    smoothness_scores = {key: 1 - (value / max_roughness) for key, value in roughness_Accuracy.items()}
    print("Smoothness Scores (Before Normalization):", smoothness_scores)

    # # Normalize smoothness scores to [0, 1]
    # max_smoothness = max(smoothness_scores.values())
    # min_smoothness = min(smoothness_scores.values())
    # normalized_smoothness = {key: score/max_smoothness
    #                          for key, score in smoothness_scores.items()}
    # print("Normalized Smoothness Scores:", normalized_smoothness)
    # print("\n")
    return smoothness_scores




df_adaptive = pd.read_csv(f"adaptive_window_Accuracy_{checking_interval}.csv")
df_fixed = pd.read_csv(f"fixed_window_Accuracy_{checking_interval}.csv")
df_traditional = pd.read_csv(f"traditional_Accuracy_{checking_interval}.csv")

min_length = min(len(df_adaptive), len(df_adaptive))
min_length = min(min_length, len(df_traditional["male_traditional"].dropna().tolist()))
df_adaptive = df_adaptive[:min_length]
df_fixed = df_fixed[:min_length]
df_traditional = df_traditional[:min_length]

print("length = ", len(df_adaptive))
print("df_adaptive: ", df_adaptive)
print("df_fixed: ", df_fixed)
print("df_traditional: ", df_traditional)




roughness_Accuracy = {}
roughness_Accuracy["male_fixed_window_roughness"] = np.std(np.diff(df_fixed["male_time_decay"])/np.diff(np.arange(len(df_fixed))))
roughness_Accuracy["male_adaptive_window_roughness"] = np.std(np.diff(df_adaptive["male_time_decay"])/np.diff(np.arange(len(df_adaptive))))
roughness_Accuracy["male_traditional_roughness"] = np.std(np.diff(df_traditional["male_traditional"])/np.diff(np.arange(len(df_traditional))))

smoothness_scores_normalized_Accuracy_male = roughness2smoothness(roughness_Accuracy)


roughness_Accuracy = {}
fixed_diff = np.diff(df_fixed["female_time_decay"])
adaptive_diff = np.diff(df_adaptive["female_time_decay"])
traditional_diff = np.diff(df_traditional["female_traditional"])

# Compute standard deviation of the differences
adaptive_std = np.std(adaptive_diff)
traditional_std = np.std(traditional_diff)
fixed_std = np.std(fixed_diff)

print("diff ", fixed_diff, adaptive_diff, traditional_diff)
print("std ", fixed_std, adaptive_std, traditional_std)

roughness_Accuracy["female_fixed_window_roughness"] = np.std(np.diff(df_fixed["female_time_decay"]))
roughness_Accuracy["female_adaptive_window_roughness"] = np.std(np.diff(df_adaptive["female_time_decay"]))
roughness_Accuracy["female_traditional_roughness"] = np.std(np.diff(df_traditional["female_traditional"]))

smoothness_scores_normalized_Accuracy_female = roughness2smoothness(roughness_Accuracy)




bar_colors = ["blue", "hotpink",  "Lime",  "red",  "DarkGrey", "cyan",  "green", "darkorchid","gold", ]
axs[0][1].bar(range(len(smoothness_scores_normalized_Accuracy_male)), smoothness_scores_normalized_Accuracy_male.values(), width=0.4,
              label = ["male fixed", "male adaptive", "male traditional"],
              color=bar_colors[:3])
axs[1][1].bar(range(len(smoothness_scores_normalized_Accuracy_female)), smoothness_scores_normalized_Accuracy_female.values(), width=0.4,
color=bar_colors[3:6],
              label = ["female fixed", "female adaptive", "female traditional"])

axs[0][1].set_yscale("log")
axs[0][1].set_title('(b) male Smoothness', y=-0.15, pad=-10, fontsize=13)
axs[0][1].set_yticks([0.1, 1.0], [0.1, 1.0], fontsize=13)
axs[0][1].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][1].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[0][1].set_xticks([], [], rotation=0, fontsize=12)
axs[0][1].tick_params(axis='x', which='major', pad=1)

axs[1][1].set_title('(d) female Smoothness', y=-0.15, pad=-10, fontsize=13)
axs[1][1].set_yticks([0.9, 0.95, 1.0], [0.9, 0.95, 1.0], fontsize=13)
axs[1][1].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[1][1].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][1].set_xticks([], [], rotation=0, fontsize=12)
# axs[1][1].tick_params(axis='x', which='major', pad=1)







handles, labels = axs[0][1].get_legend_handles_labels()
handles1, labels1 = axs[1][1].get_legend_handles_labels()
handles = handles + handles1
labels = labels + labels1


desired_order = [0, 3, 1, 4, 2, 5]
# Reorder handles and labels
reordered_handles = [handles[i] for i in desired_order]
reordered_labels = [labels[i] for i in desired_order]

axs[0][0].legend(reordered_handles, reordered_labels, title_fontsize=14, loc='upper left',
                 bbox_to_anchor=(-0.15, 2.15),handlelength=1.5,
                 fontsize=13, ncol=3, labelspacing=0.2, handletextpad=0.3, markerscale=1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)


fig.show()

fig.savefig("compas_method_curve_comparison_Accuracy.png", bbox_inches='tight')

