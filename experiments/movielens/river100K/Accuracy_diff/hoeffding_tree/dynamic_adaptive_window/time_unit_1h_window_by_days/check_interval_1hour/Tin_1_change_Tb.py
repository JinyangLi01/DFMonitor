import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
import math

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



def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)



time_unit = "1 hour"
alpha = 0.996
T_in = 1
checking_interval = "1 hour"
method_name = "hoeffding_classifier"
data = pd.read_csv('../../../../../result_' + method_name + '.csv', dtype={"zip_code": str})
print(data["gender"].unique())
date_column = "datetime"

fig, axs = plt.subplots(1, 1, figsize=(3.5, 2))
plt.rcParams['font.size'] = 10
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#41ab5d', '#006837'])
curve_colors.insert(0, 'magenta')
# print("x_list", x_list)
# print(type(col_data_FPR['black_time_decay'][0]), col_data_FPR['black_time_decay'][0])

# I tried Tb = 2, 4, 8, 12, 24, 48, 72 with Tin = 1
# the results are all the same
Tb_list = [2]
idx = 0
for T_b in Tb_list:
    df = pd.read_csv(f"movielens_compare_Accuracy_{method_name}_gender_dynamic_Tb_{T_b}_Tin_{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.csv")
    print(df)
    x_list = np.arange(0, len(df))
    male_time_decay = df["male_time_decay_dynamic"].tolist()
    female_time_decay = df["female_time_decay_dynamic"].tolist()
    axs.plot(x_list, male_time_decay, linewidth=0.5, markersize=1,
                label='{}'.format(T_b), linestyle='-', marker='o', color="blue")
    axs.plot(x_list, female_time_decay, linewidth=0.5, markersize=1,
             label='{}'.format(T_b), linestyle='-', marker='o', color="darkorange")
    idx += 1
    print(male_time_decay[20:30])


axs.set_ylim(0.6, 1.0)
axs.grid(True)

axs.set_xticks([] , [], rotation=0, fontsize=20)




axs.set_ylabel('accuracy', fontsize=17, labelpad=0, fontweight='bold')



# Add a common x-axis label
# fig.text(0.5, -0.03, 'baby names collected year, from\n 1880 to 2020, time window = 10 year',
#          ha='center', va='center', fontsize=16, fontweight='bold')
# # Add a common y-axis label
# fig.text(-0.07, 0.55, 'y-axis: value of false positive rate (FPR) or coverage ratio (CR)', ha='center', va='center', fontsize=15, rotation='vertical')


fig.legend(title='Alpha Values', title_fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 1.19), fontsize=14,
           ncol=7, labelspacing=0.3, handletextpad=0.24, markerscale=1.6, handlelength=1.9,
           columnspacing=0.28, borderpad=0.2, frameon=True)

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.37, hspace=0.28)
plt.tight_layout()
plt.savefig(f"merged_fig_change_Tb_Tin{T_in}_alpha_{str(get_integer(alpha))}_time_unit_{time_unit}_check_interval_{checking_interval}.png", bbox_inches='tight')
plt.show()


