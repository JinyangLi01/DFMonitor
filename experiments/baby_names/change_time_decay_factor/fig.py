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

# sns.set_theme(font='CMU Serif', style='darkgrid')

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


# read Accuracy
df_Accuracy = pd.read_csv("Accuracy_time_decay_factor.csv", sep=",")
# print(df_CR.head)
col_data_Accuracy = {}
for col_name in df_Accuracy.columns:
    col_data_Accuracy[col_name] = df_Accuracy[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(ast.literal_eval(col_data_Accuracy['male_time_decay'][0])))
alpha_list = col_data_Accuracy['alpha']
fig, axs = plt.subplots(1, 2, figsize=(6, 2.4))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#41ab5d', '#006837'])
curve_colors.append('magenta')
# print("x_list", x_list)
# print(type(col_data_FPR['black_time_decay'][0]), col_data_FPR['black_time_decay'][0])


for i in range(len(alpha_list) - 1):
    axs[0].plot(x_list, ast.literal_eval(col_data_Accuracy['male_time_decay'][i]), linewidth=2, markersize=4,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[0, 0].plot(x_list, (col_data_FPR['black_traditional']), linewidth=2, markersize=4,
    #             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    axs[1].plot(x_list, ast.literal_eval(col_data_Accuracy['female_time_decay'][i]), linewidth=2, markersize=4,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[0, 1].plot(x_list, (col_data_FPR['white_traditional']), linewidth=2, markersize=4,
    #             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])


i = len(alpha_list) - 1
axs[0].plot(x_list, ast.literal_eval(df_Accuracy['male_time_decay'][i]), linewidth=2, markersize=3,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[1].plot(x_list, ast.literal_eval(df_Accuracy['female_time_decay'][i]), linewidth=2, markersize=3,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])

axs[0].set_title('(a) male', y=-0.13, pad=-0.5, fontweight='bold')
axs[1].set_title('(b) female', y=-0.13, pad=-0.5, fontweight='bold')
axs[0].set_yticks(np.arange(0.86, 0.99, 0.04), [0.86, '0.90', 0.94, 0.98], fontsize=15)
axs[1].set_yticks([0.87, 0.90, 0.93, 0.95], [0.87, '0.90', 0.93, 0.95], fontsize=15)



axs[0].grid(True)
axs[1].grid(True)
axs[0].set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
axs[1].set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)



axs[0].set_ylabel('accuracy', fontsize=17, labelpad=0, fontweight='bold')

axs[1].set_ylabel('accuracy', fontsize=17, labelpad=0, fontweight='bold')


# Add a common x-axis label
fig.text(0.5, -0.03, 'baby names collected year, from\n 1880 to 2020, time window = 10 year',
         ha='center', va='center', fontsize=16, fontweight='bold')
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=15, rotation='vertical')

handles, labels = axs[0].get_legend_handles_labels()
# put the last legend as the first
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]

fig.legend(title='Alpha Values', title_fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 1.19), fontsize=14,
           ncol=7, labelspacing=0.3, handletextpad=0.24, markerscale=1.6, handlelength=1.9,
           columnspacing=0.28, borderpad=0.2, frameon=True, handles=handles, labels=labels)

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.37, hspace=0.28)
plt.tight_layout()
plt.savefig("baby_names_time_decay_factor.png", bbox_inches='tight')
plt.show()
