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


# read Accuracy
df_Accuracy = pd.read_csv("Accuracy_window_size.csv", sep=",")
# print(df_CR.head)
col_data_Accuracy = {}
for col_name in df_Accuracy.columns:
    col_data_Accuracy[col_name] = df_Accuracy[col_name].tolist()

# draw the plot

# draw the plot

x_list = np.arange(0, len(ast.literal_eval(col_data_Accuracy['male_time_decay'][0])))
window_size_list = col_data_Accuracy['window_size']
fig, axs = plt.subplots(1, 2, figsize=(6, 2.3))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['black', '#09339e', '#5788ee', '#00b9bc', '#7fffff', '#81db46',
                                          '#41ab5d', '#006837'])
# curve_colors = sns.color_palette(palette=['firebrick', 'darkorange', '#004b00', 'blueviolet', '#0000ff', '#57d357',
#                                           'magenta', 'cyan'])
curve_colors = sns.color_palette(palette=[ '#0000ff', 'cyan', '#57d357', '#004b00', 'darkorange', 'firebrick',
                                           'blueviolet', 'magenta'])


# curve_colors = curve_colors[::-1]
# curve_colors = sns.color_palette(palette=["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56",
#                                           "#e14b31", "#c23728", "#8c2d1c"])
# print(type(col_data_Accuracy['male_time_decay'][0]), col_data_Accuracy['male_time_decay'][0])


max_length = max(len(ast.literal_eval(curve)) for curve in col_data_Accuracy['male_time_decay'])
print("max_length", max_length)

for i in range(len(window_size_list)):
    # print(ast.literal_eval(col_data_FPR['black_time_decay'][i]))
    curve = ast.literal_eval(col_data_Accuracy['male_time_decay'][i])
    print(curve)
    proportional_x = np.linspace(0, max_length - 1, len(curve))
    print(len(curve), len(proportional_x))
    axs[0].plot(proportional_x, curve, linewidth=1.4, markersize=3.5,
                   label='{}'.format(window_size_list[i]), linestyle='-', marker='o', color=curve_colors[i])
    curve = ast.literal_eval(col_data_Accuracy['female_time_decay'][i])
    proportional_x = np.linspace(0, max_length - 1, len(curve))
    axs[1].plot(proportional_x, curve, linewidth=1.4, markersize=3.5,
                   label='{}'.format(window_size_list[i]), linestyle='-', marker='o', color=curve_colors[i])




axs[0].set_title('(a) male', y=-0.19, pad=-0.5, fontweight='bold')
axs[1].set_title('(d) female', y=-0.19, pad=-0.5, fontweight='bold')

for ax in axs.flat:
    ax.grid(True)
    ax.set_xticks(np.arange(0, max_length), [], rotation=0, fontsize=20)
    ax.set_ylabel('accuracy', fontsize=16, fontweight='bold')

axs[0].set_yticks([0.86, 0.90, 0.94, 0.98], [0.86, '0.90', 0.94, 0.98], fontsize=15)
axs[1].set_yticks([0.88, 0.90, 0.93, 0.95], [0.88, '0.90', 0.93, 0.95], fontsize=15)



# Add a common x-axis label
fig.text(0.5, 0.04, 'normalized measuring time',
         ha='center', va='center', fontsize=16, fontweight='bold')
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=15, rotation='vertical')

# create a common legend
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(title='time window', title_fontsize=14, loc='lower left', bbox_to_anchor=(0.1, 0.92), fontsize=14,
           ncol=4, labelspacing=0.2, handletextpad=0.5, markerscale=2, handlelength=2,
           columnspacing=0.6, borderpad=0.2, frameon=True, handles=handles, labels=labels)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
plt.tight_layout()
plt.savefig("baby_names_window_size.png", bbox_inches='tight')
plt.show()
