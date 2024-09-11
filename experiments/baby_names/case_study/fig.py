import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
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
df_Accuracy = pd.read_csv("baby_names_compare_Accuracy.csv", sep=",")

# print(df_Accuracy.columns)

col_data_Accuracy = {}

for col_name in df_Accuracy.columns:
    col_data_Accuracy[col_name] = df_Accuracy[col_name].tolist()

# # read CR
# df_CR = pd.read_csv("baby_names_compare_CR.csv", sep=",")
# # print(df_CR.head)
# col_data_Accuracy = {}
# for col_name in df_CR.columns:
#     col_data_Accuracy[col_name] = df_CR[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(df_Accuracy))

# print(len(x_list))

fig, axs = plt.subplots(1, 1, figsize=(4, 1.9))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])


pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']



# Plot the first curve (y1_values)
axs.plot(x_list, col_data_Accuracy["male_time_decay"], linewidth=3.2, markersize=5.1, label='time decay',
            linestyle='-', marker='o', color=pair_colors[0])

axs.plot(x_list, col_data_Accuracy["male_traditional"], linewidth=3, markersize=5.1, label='traditional',
            linestyle='--', marker='s', color=pair_colors[1])

# Plot the second curve (y2_values)
axs.plot(x_list, col_data_Accuracy["female_time_decay"], linewidth=3.2, markersize=5.1, label='time decay',
            linestyle='-', marker='o', color=pair_colors[2])

axs.plot(x_list, col_data_Accuracy["female_traditional"], linewidth=3, markersize=5.1, label='traditional',
            linestyle='--', marker='s',
            color=pair_colors[3])




# add a common x-axis label
fig.text(0.5, -0.04, 'baby names collected year, from\n 1880 to 2020, time window = 10 year',
            ha='center', va='center', fontsize=16, fontweight='bold')

axs.set_ylabel('accuracy', fontsize=19, labelpad=0, fontweight='bold', y=0.5)
# axs.set_yticks([0.85, 0.9, 0.95, 1.0], [0.85, '0.90', 0.95, '1.00'], fontsize=16)


axs.grid(True)

axs.set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=15)
axs.set_yticks([0.875, 0.900, 0.925, 0.950, 0.975], ['0.875', '0.900', '0.925', '0.950', '0.975'], fontsize=16)

# create a common legend
handles, labels = axs.get_legend_handles_labels()


fig.legend(title='male                  female', title_fontsize=15,
           handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0.09, 0.90), fontsize=13,
           ncol=2, labelspacing=0.17, handletextpad=0.4, markerscale=2.0, handlelength=3,
           columnspacing=0.8, borderpad=0.2, frameon=True)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0)


plt.tight_layout()
plt.savefig("baby_names_compare.png", bbox_inches='tight')
plt.show()



###################### smoothness of the curve ############################

smooth_Accuracy = {}
smooth_Accuracy["male_time_decay"] = np.std(np.diff(col_data_Accuracy["male_time_decay"]) / np.diff(x_list))
smooth_Accuracy["male_traditional"] = np.std(np.diff(col_data_Accuracy["male_traditional"]) / np.diff(x_list))
smooth_Accuracy["female_time_decay"] = np.std(np.diff(col_data_Accuracy["female_time_decay"]) / np.diff(x_list))
smooth_Accuracy["female_traditional"] = np.std(np.diff(col_data_Accuracy["female_traditional"]) / np.diff(x_list))

# another file for CR
with open("roughness_compas_Accuracy.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["smoothness"])
    writer.writerow(["male_time_decay", np.std(np.diff(col_data_Accuracy["male_time_decay"]) / np.diff(x_list))])
    writer.writerow(["male_traditional", np.std(np.diff(col_data_Accuracy["male_traditional"]) / np.diff(x_list))])
    writer.writerow(["female_time_decay", np.std(np.diff(col_data_Accuracy["female_time_decay"]) / np.diff(x_list))])
    writer.writerow(["female_traditional", np.std(np.diff(col_data_Accuracy["female_traditional"]) / np.diff(x_list))])


smoothness_scores = [1/sd for sd in list(smooth_Accuracy.values())]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_Accuracy = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores with Scikit-learn:", smoothness_scores_normalized_Accuracy)




# draw two bar charts using FPR and CR smoothness
fig, axs = plt.subplots(1, 1, figsize=(3, 1.7))

legend_labels = ["time decay", "traditional", "time decay", "traditional"]

axs.bar(smooth_Accuracy.keys(), smoothness_scores_normalized_Accuracy, color=pair_colors,
        label=legend_labels, width=0.4)
axs.set_ylabel('smoothness', fontsize=17, labelpad=-1, fontweight='bold')
# axs.set_title('accuracy', y=-0.13, pad=0, fontweight='bold')
axs.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
axs.grid(True)
# remove x ticks
axs.set_xticks([], [], rotation=0, fontsize=20)
# axs.set_xticks(np.arange(0, len(legend_labels)), legend_labels, rotation=0, fontsize=20)

axs.bar_label(axs.containers[0], labels=["{:0.2f}".format(score) for score in smoothness_scores_normalized_Accuracy],
              fontsize=15, padding=-0.9)

handles, labels = axs.get_legend_handles_labels()
fig.legend(title='male                  female', title_fontsize=15,
           handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(-0.07, 0.97), fontsize=13,
              ncol=2, labelspacing=0.25, handletextpad=0.4, markerscale=2, handlelength=1.9,
              columnspacing=0.8, borderpad=0.2, frameon=True)
# fig.legend()
plt.tight_layout()
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0)
fig.savefig("baby_names_smoothness_normalized.png", bbox_inches='tight')
plt.show()