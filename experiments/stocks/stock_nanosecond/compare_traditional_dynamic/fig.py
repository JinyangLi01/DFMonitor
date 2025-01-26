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





fig, axs = plt.subplots(1, 2, figsize=(8, 2.55))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#730a47', '#d810ef'])


pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), '#8ef1e8',
               '#287c37', '#cccc00',
               '#730a47', '#9966cc']



curve_names = ['Technology', 'Consumer Cyclical', 'Communication Services', 'Consumer Defensive', 'Energy',
                          'Healthcare', 'Financial Services']
curve_colors = sns.color_palette(palette=['blue', 'limegreen', '#ffb400', 'darkviolet', 'cyan', 'black',
                                              "red", 'magenta'])


df_fixed = pd.read_csv("accuracy_fixed_window.csv")
df_fixed["check_points"] = pd.to_datetime(df_fixed["check_points"])
df_adaptive = pd.read_csv("accuracy_adaptive_window.csv")
df_adaptive["check_points"] = pd.to_datetime(df_adaptive["check_points"])
df_traditional = pd.read_csv("traditional_accuracy_time_window_1s.csv")


# Set up a row of 8 subplots with a shared y-axis
fig, axes = plt.subplots(1, 2, figsize=(4, 1.7))
plt.subplots_adjust(top=0.82, bottom=0.32, hspace=0, wspace=0.34, left=0.04, right=0.99)

warm_up_time = pd.Timestamp('2024-10-15 14:00:06.7', tz='UTC')
df_fixed = df_fixed[df_fixed["check_points"] >= warm_up_time]
x_list = np.arange(0, len(df_fixed))
for i in range(7):
    axes[0].plot(x_list, df_fixed[curve_names[i]].tolist(),
                 linewidth=0.5, markersize=0.1,
                 label=curve_names[i], linestyle='-',
                 marker='o', color=curve_colors[i], alpha=1)

# df_adaptive = df_adaptive[df_adaptive["check_points"] >= warm_up_time]
# for i in range(7):
#     axes[0].plot(x_list, df_adaptive[curve_names[i]].tolist(),
#                     linewidth=0.5, markersize=0.1,
#                     label=curve_names[i], linestyle='-',
#                     marker='o', color=curve_colors[i], alpha=1)

#
# for i in range(7):
#     df_part = df_traditional[df_traditional["sector"] == curve_names[i]]
#     axes[0].plot(range(len(df_part["accuracy"])), df_part["accuracy"].tolist(),
#                     linewidth=0.5, markersize=0.1,
#                     label=curve_names[i], linestyle='-',
#                     marker='o', color=curve_colors[i], alpha=1)
#






#
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(-0.07, 0.95), fontsize=13,
#               ncol=3, labelspacing=0.3, handletextpad=0.3, markerscale=2, handlelength=1.9,
#               columnspacing=0.5, borderpad=0.2, frameon=True)
# fig.legend()


fig.savefig("compas.png", bbox_inches='tight')
plt.show()