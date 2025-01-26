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


# read FPR
df_FPR = pd.read_csv("FPR_time_decay_factor.csv", sep=",")

# print(df_FPR.columns)

col_data_FPR = {}

for col_name in df_FPR.columns:
    col_data_FPR[col_name] = df_FPR[col_name].tolist()

# read CR
df_CR = pd.read_csv("CR_time_decay_factor.csv", sep=",")
# print(df_CR.head)
col_data_CR = {}
for col_name in df_CR.columns:
    col_data_CR[col_name] = df_CR[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(ast.literal_eval(col_data_FPR['black_time_decay'][0])))
alpha_list = col_data_FPR['alpha']
fig, axs = plt.subplots(2, 3, figsize=(8, 2.8))
plt.subplots_adjust(left=0.0, right=1, top=0.98, bottom=0.05, wspace=0.3, hspace=0.3)
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#41ab5d', '#006837'])
curve_colors.append('magenta')
# print("x_list", x_list)
# print(type(col_data_FPR['black_time_decay'][0]), col_data_FPR['black_time_decay'][0])


for i in range(1, len(alpha_list)):
    axs[0, 0].plot(x_list, ast.literal_eval(col_data_FPR['black_time_decay'][i]), linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])

    axs[1, 0].plot(x_list, ast.literal_eval(col_data_CR['black_time_decay'][i]), linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])

    y = ast.literal_eval(col_data_FPR['white_time_decay'][i])
    axs[0, 1].plot(x_list, y, linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[1, 0].axhline(linear_threshold, color='red', linestyle='--', label='Linear to Log Transition')
    # axs[1, 0].axhline(log_linear_transition, color='green', linestyle='--', label='Log to Linear Transition')

    axs[1, 1].plot(x_list, ast.literal_eval(col_data_CR['white_time_decay'][i]), linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[1, 1].plot(x_list, (col_data_CR['white_traditional']), linewidth=2, markersize=3,
    #             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    axs[0, 2].plot(x_list, ast.literal_eval(col_data_FPR['hispanic_time_decay'][i]), linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[2, 0].plot(x_list, (col_data_CR['hispanic_traditional']), linewidth=2, markersize=3,
    #             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    axs[1, 2].plot(x_list, ast.literal_eval(col_data_CR['hispanic_time_decay'][i]), linewidth=2, markersize=3,
                   label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    # axs[2, 1].plot(x_list, (col_data_FPR['hispanic_traditional']), linewidth=2, markersize=3,
    #             label='{}'.format(alpha_list[i]), linestyle='-', marker='o', color=curve_colors[i])

i = 0
axs[0, 0].plot(x_list, ast.literal_eval(col_data_FPR['black_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[1, 0].plot(x_list, ast.literal_eval(col_data_CR['black_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[0, 1].plot(x_list, ast.literal_eval(col_data_FPR['white_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[1, 1].plot(x_list, ast.literal_eval(col_data_CR['white_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[0, 2].plot(x_list, ast.literal_eval(col_data_FPR['hispanic_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])
axs[1, 2].plot(x_list, ast.literal_eval(col_data_CR['hispanic_time_decay'][i]), linewidth=2, markersize=2,
               label='{}'.format(alpha_list[i]), linestyle=':', marker='s', color=curve_colors[i])

axs[0, 0].set_title('(a) FPR, Black', y=-0.22, pad=-1.0,)
axs[1, 0].set_title('(d) CR, Black', y=-0.22, pad=-1.0,)
axs[0, 1].set_title('(b) FPR, White', y=-0.22, pad=-1.0,)
axs[1, 1].set_title('(e) CR, White', y=-0.22, pad=-1.0,)
axs[0, 2].set_title('(c) FPR, Hispanic', y=-0.22, pad=-1.0,)
axs[1, 2].set_title('(f) CR, Hispanic', y=-0.22, pad=-1.0,)

for ax in axs.flat:
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=14)

axs[0, 0].set_yscale('symlog', linthresh=0.29, linscale=0.15, base=2)
axs.flat[0].set_yticks([0, 0.3, 0.4, 0.5], ['0', '0.3', '0.4', '0.5'])
#
axs.flat[2].set_yscale('loglinearlog', A=0.08, linthresh=0.4)
# axs.flat[2].set_ylim(0.08, 1.15)
axs.flat[2].set_yticks([0.1, 0.2, 0.3, 1], ['0.1', '0.2', '0.3', '1.0'])
# axs.flat[2].set_yscale('symlog', linthresh=0.08, linscale=0.1, base=2)
# axs.flat[2].set_ylabel('false positive rate (FPR)', fontsize=14, labelpad=1,)

axs.flat[4].set_yscale('symlog', linthresh=0.08, linscale=0.08, base=2)
axs.flat[4].set_yticks([0.2, 0.3, 0.5], ['0.2', '0.3', '0.5'])

axs.flat[1].set_yscale('loglinearlog', A=0.05, linthresh=0.28)
axs.flat[1].set_ylim(0.1, 1.05)
axs.flat[1].set_yticks([0.1, 0.2, 0.3, 1], ['0.1', '0.2', '0.3', '1.0'])

axs.flat[3].set_yscale('linear')
axs.flat[3].set_yticks([0.3, 0.4, 0.5, 0.6], ['0.3', '0.4', '0.5', '0.6'])


axs.flat[5].set_yscale('log')
axs.flat[5].set_yticks([0.04, 0.06, 0.1, 0.2, 0.3], ['0.04', '0.06', '0.1', '0.2', '0.3'])

# Add a common x-axis label
fig.text(0.48, -0.1, 'compas screening date, from 01/01/2013 to 12/31/2014, time window = 1 month',
         ha='center', va='center', fontsize=14,)
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=14, rotation='vertical')

# create a common legend
handles, labels = axs[-1, -1].get_legend_handles_labels()
# # put the last legend as the first
# handles = [handles[-1]] + handles[:-1]
# labels = [labels[-1]] + labels[:-1]

plt.legend(title='Alpha Values', title_fontsize=14, loc='upper left', bbox_to_anchor=(-2.7, 3), fontsize=14,
           ncol=7, labelspacing=0.3, handletextpad=0.3, markerscale=1.6,
           columnspacing=0.4, borderpad=0.2, frameon=True, handles=handles, labels=labels)


plt.savefig("compas_time_decay_factor.png", bbox_inches='tight')
plt.show()



##########################################################################################################
