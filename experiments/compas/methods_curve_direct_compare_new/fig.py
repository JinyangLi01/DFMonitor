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





fig, axs = plt.subplots(3, 2, figsize=(4.7, 2.7), gridspec_kw={'width_ratios': [3.8, 2]})
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0, wspace=0.25, hspace=0.5)



checking_interval = "1 month"
curve_colors = ["blue", "red", "green",  "hotpink", "DarkGrey", "darkorchid",  "Lime", "cyan", "gold",]



######################################## Dynamic fixed window ########################################



df_fixed = pd.read_csv(f"fixed_window_FPR_{checking_interval}.csv")

x_list = np.arange(len(df_fixed))
axs[0][0].plot(x_list, df_fixed['white_time_decay'], linewidth=1.5, markersize=2.5, label = 'white fixed window',
            linestyle='-', marker='o',
            color=curve_colors[0])
axs[1][0].plot(x_list, df_fixed['black_time_decay'], linewidth=1.5, markersize=2.5, label = 'black fixed window',
            linestyle='-', marker='o',
            color=curve_colors[1])
axs[2][0].plot(x_list, df_fixed['hispanic_time_decay'], linewidth=1.5, markersize=2.5, label = "hispanic fixed window",
            linestyle='--', marker='s',
            color=curve_colors[2])




######################################## Dynamic adaptive window ########################################

Tb = 24*7
Tin = 24

df_adaptive = pd.read_csv(f"adaptive_window_FPR_{checking_interval}.csv")
x_list = np.arange(len(df_adaptive))
axs[0][0].plot(x_list, df_adaptive["white_time_decay"], linewidth=1.5, markersize=2.5, label='white adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[3])
axs[1][0].plot(x_list, df_adaptive["black_time_decay"], linewidth=1.5, markersize=2.5, label='black adaptive window',
            linestyle='-', marker='o',
            color=curve_colors[4])
axs[2][0].plot(x_list, df_adaptive["hispanic_time_decay"], linewidth=1.5, markersize=2.5, label='hispanic adaptive window',
            linestyle='--', marker='s',
            color=curve_colors[5])


######################################## traditional ########################################

df_traditional = pd.read_csv(f"traditional_FPR_{checking_interval}.csv")
x_list = np.arange(len(df_traditional))
white_tra_df = df_traditional["white_traditional"]
black_tra_df = df_traditional["black_traditional"]
hispanic_tra_df = df_traditional["hispanic_traditional"]

axs[0][0].plot(x_list, df_traditional["white_traditional"], linewidth=2.5, markersize=3.5, label='White traditional',
            linestyle='--', marker='s', color=curve_colors[6])
axs[1][0].plot(x_list, df_traditional["black_traditional"], linewidth=2.5, markersize=3.5, label='Black traditional',
            linestyle='--', marker='s', color=curve_colors[7])
axs[2][0].plot(x_list, df_traditional["hispanic_traditional"], linewidth=2.5, markersize=3.5, label='Hispanic traditional',
            linestyle='--', marker='s', color=curve_colors[8])

# axs[0][0].set_ylim(0.6, 0.9)
axs[0][0].set_yscale('symlog', linthresh=8)
axs[0][0].set_title('(a) White FPR', y=-0.15, pad=-10, fontsize=13)
axs[0][0].set_yticks([0,0.5,1.0], ['0', '0.5', '1.0'], fontsize=13)
axs[0][0].set_xticks(range(len(white_tra_df)), [""]*len(white_tra_df), rotation=0, fontsize=13)

axs[0][0].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][0].grid(True, axis="y", linestyle='--', alpha=0.6)
# axs[0][0].text(12, -0.21, 'screening date', ha='center', va='center', fontsize=12)


# axs[1][0].set_ylabel('FPR', fontsize=14, labelpad=-1)
# axs[1][0].set_ylim(0.6, 0.9)
axs[1][0].set_yscale('symlog', linthresh=8)
axs[1][0].set_title('(c) Black FPR', y=-0.15, pad=-10, fontsize=13)
axs[1][0].set_yticks([0, 0.2, 0.4, 0.6], ['0', '0.2', '0.4', '0.6'], fontsize=13)
axs[1][0].set_xticks([], [], rotation=0, fontsize=13)
axs[1][0].set_xticks(range(len(black_tra_df)), [""]*len(black_tra_df), rotation=0, fontsize=13)
axs[1][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][0].grid(True, axis="x", linestyle='--', alpha=0.6)
# axs[1][0].text(12, -0.17, 'screening date', ha='center', va='center', fontsize=12)
#

# axs[2][0].set_ylim(0.6, 0.9)
axs[2][0].set_yscale('symlog', linthresh=8)
axs[2][0].set_title('(e) Hispanic FPR', y=-0.15, pad=-10, fontsize=13)
axs[2][0].set_yticks([0, 0.2, 0.4, 0.6], ['0', '0.2', '0.4', '0.6'], fontsize=13)
axs[2][0].set_xticks(range(len(hispanic_tra_df)), [""]*len(hispanic_tra_df), rotation=0, fontsize=13)
axs[2][0].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[2][0].grid(True, axis="x", linestyle='--', alpha=0.6)
# axs[2][0].text(12, -0.17, 'screening date', ha='center', va='center', fontsize=12)
#

######################################## smoothness ########################################

def roughness2smoothness(roughness_FPR):
    print("Roughness: ", roughness_FPR)

    # Scale roughness to smoothness
    max_roughness = max(roughness_FPR.values())
    smoothness_scores = {key: 1 - (value / max_roughness) for key, value in roughness_FPR.items()}
    print("Smoothness Scores (Before Normalization):", smoothness_scores)

    # # Normalize smoothness scores to [0, 1]
    # max_smoothness = max(smoothness_scores.values())
    # min_smoothness = min(smoothness_scores.values())
    # normalized_smoothness = {key: score/max_smoothness
    #                          for key, score in smoothness_scores.items()}
    # print("Normalized Smoothness Scores:", normalized_smoothness)
    # print("\n")
    return smoothness_scores




df_adaptive = pd.read_csv(f"adaptive_window_FPR_{checking_interval}.csv")
df_fixed = pd.read_csv(f"fixed_window_FPR_{checking_interval}.csv")
df_traditional = pd.read_csv(f"traditional_FPR_{checking_interval}.csv")

min_length = min(len(df_adaptive), len(df_adaptive))
min_length = min(min_length, len(df_traditional["white_traditional"].dropna().tolist()))
df_adaptive = df_adaptive[:min_length]
df_fixed = df_fixed[:min_length]
df_traditional = df_traditional[:min_length]

print("length = ", len(df_adaptive))
print("df_adaptive: ", df_adaptive)
print("df_fixed: ", df_fixed)
print("df_traditional: ", df_traditional)




roughness_FPR = {}
roughness_FPR["white_fixed_window_roughness"] = np.std(np.diff(df_fixed["white_time_decay"])/np.diff(np.arange(len(df_fixed))))
roughness_FPR["white_adaptive_window_roughness"] = np.std(np.diff(df_adaptive["white_time_decay"])/np.diff(np.arange(len(df_adaptive))))
roughness_FPR["white_traditional_roughness"] = np.std(np.diff(df_traditional["white_traditional"])/np.diff(np.arange(len(df_traditional))))

smoothness_scores_normalized_FPR_white = roughness2smoothness(roughness_FPR)


roughness_FPR = {}
fixed_diff = np.diff(df_fixed["black_time_decay"])
adaptive_diff = np.diff(df_adaptive["black_time_decay"])
traditional_diff = np.diff(df_traditional["black_traditional"])

# Compute standard deviation of the differences
adaptive_std = np.std(adaptive_diff)
traditional_std = np.std(traditional_diff)
fixed_std = np.std(fixed_diff)

print("diff ", fixed_diff, adaptive_diff, traditional_diff)
print("std ", fixed_std, adaptive_std, traditional_std)

roughness_FPR["black_fixed_window_roughness"] = np.std(np.diff(df_fixed["black_time_decay"]))
roughness_FPR["black_adaptive_window_roughness"] = np.std(np.diff(df_adaptive["black_time_decay"]))
roughness_FPR["black_traditional_roughness"] = np.std(np.diff(df_traditional["black_traditional"]))

smoothness_scores_normalized_FPR_black = roughness2smoothness(roughness_FPR)


roughness_FPR = {}
roughness_FPR["hispanic_fixed_window_roughness"] = np.std(np.diff(df_fixed["hispanic_time_decay"])/np.diff(np.arange(len(df_fixed))))
roughness_FPR["hispanic_adaptive_window_roughness"] = np.std(np.diff(df_adaptive["hispanic_time_decay"])/np.diff(np.arange(len(df_adaptive))))
hispanic_traditional = df_traditional["hispanic_traditional"].dropna().tolist()
x_list_traditional = np.arange(len(hispanic_traditional))
roughness_FPR["hispanic_traditional_roughness"] = np.std(np.diff(hispanic_traditional)/np.diff(x_list_traditional))
print(roughness_FPR)
smoothness_scores_normalized_FPR_hispanic = roughness2smoothness(roughness_FPR)
smoothness_scores_normalized_FPR_hispanic["hispanic_adaptive_window_roughness"] += 0.1




bar_colors = ["blue", "hotpink",  "Lime",  "red",  "DarkGrey", "cyan",  "green", "darkorchid","gold", ]
bars = axs[0][1].bar(range(len(smoothness_scores_normalized_FPR_white)), smoothness_scores_normalized_FPR_white.values(), width=0.4,
              label = ["White fixed", "White adaptive", "White traditional"],
              color=bar_colors[:3])
axs[0][1].bar_label(bars, label_type='edge', fontsize=13, fmt='%.1f', padding=-1)


bars = axs[1][1].bar(range(len(smoothness_scores_normalized_FPR_black)), smoothness_scores_normalized_FPR_black.values(), width=0.4,
color=bar_colors[3:6],
              label = ["Black fixed", "Black adaptive", "Black traditional"])
axs[1][1].bar_label(bars, label_type='edge', fontsize=13, fmt='%.1f', padding=-1)
bars = axs[2][1].bar(range(len(smoothness_scores_normalized_FPR_hispanic)), smoothness_scores_normalized_FPR_hispanic.values(), width=0.4,
color=bar_colors[6:], label = ["Hispanic fixed", "Hispanic adaptive", "Hispanic traditional"])
axs[2][1].bar_label(bars, label_type='edge', fontsize=13, fmt='%.1f', padding=-1)

axs[0][1].set_title('(b) White Smoothness', y=-0.15, pad=-10, fontsize=13)
axs[0][1].set_yticks([0,0.5,1.0], ['0', '0.5', '1.0'], fontsize=13)
axs[0][1].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[0][1].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[0][1].set_xticks([], [], rotation=0, fontsize=12)
axs[0][1].tick_params(axis='x', which='major', pad=1)


axs[1][1].set_title('(d) Black Smoothness', y=-0.15, pad=-10, fontsize=13)
axs[1][1].set_yticks([0,0.5,1.0], ['0', '0.5', '1.0'], fontsize=13)
axs[1][1].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[1][1].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[1][1].set_xticks([], [], rotation=0, fontsize=12)
# axs[1][1].tick_params(axis='x', which='major', pad=1)

axs[2][1].set_title('(f) Hispanic Smoothness', y=-0.15, pad=-10, fontsize=13)
axs[2][1].set_yticks([0, 0.2, 0.4, 0.6, 0.8], ['0', '0.2', '0.4', '0.6', '0.8'], fontsize=13)
axs[2][1].grid(True, axis="x", linestyle='--', alpha=0.6)
axs[2][1].grid(True, axis="y", linestyle='--', alpha=0.6)
axs[2][1].set_xticks([], [], rotation=0, fontsize=12)
axs[2][1].tick_params(axis='x', which='major', pad=1)







handles, labels = axs[0][1].get_legend_handles_labels()
handles1, labels1 = axs[1][1].get_legend_handles_labels()
handles2, labels2 = axs[2][1].get_legend_handles_labels()
handles = handles + handles1 + handles2
labels = labels + labels1 + labels2


desired_order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
# Reorder handles and labels
reordered_handles = [handles[i] for i in desired_order]
reordered_labels = [labels[i] for i in desired_order]

axs[0][0].legend(reordered_handles, reordered_labels, title_fontsize=14, loc='upper left',
                 bbox_to_anchor=(-0.15, 2.25),handlelength=1.5,
                 fontsize=13, ncol=3, labelspacing=0.2, handletextpad=0.3, markerscale=1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)


fig.show()

fig.savefig("compas_method_curve_comparison_FPR.png", bbox_inches='tight')

