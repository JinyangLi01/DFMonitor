import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import FNR_workload as workload
import seaborn as sns
import colorsys

sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


method_name = "hoeffding_adaptive_classifier_age_2groups"
df = pd.read_csv('movielens_compare_Accuracy_' + method_name + '.csv', dtype={"zip_code": str})

print(df)

x_list = np.arange(0, len(df))
age_7_30_time_decay = df["age_7_30_time_decay"].tolist()
age_7_30_traditional = df["age_7_30_traditional"].tolist()
age_31_73_time_decay = df["age_31_73_time_decay"].tolist()
age_31_73_traditional = df["age_31_73_traditional"].tolist()

pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
                '#287c37', '#cccc00']






fig, ax = plt.subplots(figsize=(6, 3.5))
# Get the "Paired" color palette
paired_palette = sns.color_palette("Paired")
# Rearrange the colors within each pair
# pair_colors = [paired_palette[i + 1] if i % 2 == 0 else paired_palette[i - 1] for i in range(len(paired_palette))]
pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
               '#287c37', '#cccc00',
               'indianred', '#fe01b1',
               'black', 'red']

ax.plot(x_list, age_7_30_time_decay, linewidth=3, markersize=6, label='7-30 time decay', linestyle='-', marker='o',
        color=pair_colors[0])
ax.plot(x_list, age_7_30_traditional, linewidth=3, markersize=6, label='7-30 traditional', linestyle='--', marker='s',
        color=pair_colors[1])
ax.plot(x_list, age_31_73_time_decay, linewidth=3, markersize=6, label='31-73 time decay', linestyle='-', marker='o',
        color=pair_colors[2])
ax.plot(x_list, age_31_73_traditional, linewidth=3, markersize=6, label='31-73 traditional', linestyle='--', marker='s',
        color=pair_colors[3])



plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yticks([0.0, 0.2, 0.4, 0.6], fontsize=20)
plt.xlabel('year',
           fontsize=20, labelpad=-2).set_position((0.47, -0.1))
plt.ylabel('Accuracy', fontsize=20, labelpad=-1)
plt.grid(True)
plt.tight_layout()
plt.legend(loc='lower left', bbox_to_anchor=(-0.142, 1), fontsize=15,
           ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
           columnspacing=0.2, borderpad=0.2, frameon=True)
filename = "Acc_movielens_" + method_name + ".png"
plt.savefig(filename, bbox_inches='tight')
plt.show()
