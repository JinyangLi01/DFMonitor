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



df = pd.read_csv("movielens_compare_Accuracy_hoeffding_classifier_age_group.csv")
print(df)

x_list = np.arange(0, len(df))
age_7_25_time_decay = df["age_7_25_time_decay"].tolist()
age_7_25_traditional = df["age_7_25_traditional"].tolist()
age_26_31_time_decay = df["age_26_31_time_decay"].tolist()
age_26_31_traditional = df["age_26_31_traditional"].tolist()
age_32_40_time_decay = df["age_32_40_time_decay"].tolist()
age_32_40_traditional = df["age_32_40_traditional"].tolist()
age_41_73_time_decay = df["age_41_73_time_decay"].tolist()
age_41_73_traditional = df["age_41_73_traditional"].tolist()




fig, ax = plt.subplots(figsize=(6, 3.5))
# Get the "Paired" color palette
paired_palette = sns.color_palette("Paired")
# Rearrange the colors within each pair
# pair_colors = [paired_palette[i + 1] if i % 2 == 0 else paired_palette[i - 1] for i in range(len(paired_palette))]
pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
               '#287c37', '#cccc00',
               'indianred', '#fe01b1',
               'black', 'red']

ax.plot(x_list, age_7_25_time_decay, linewidth=3, markersize=6, label='7-25 time decay', linestyle='-', marker='o',
        color=pair_colors[0])
ax.plot(x_list, age_7_25_traditional, linewidth=3, markersize=6, label='7-25 traditional', linestyle='--', marker='s',
        color=pair_colors[1])
ax.plot(x_list, age_26_31_time_decay, linewidth=3, markersize=6, label='26-31 time decay', linestyle='-', marker='o',
        color=pair_colors[2])
ax.plot(x_list, age_26_31_traditional, linewidth=3, markersize=6, label='26-31 traditional', linestyle='--', marker='s',
        color=pair_colors[3])
ax.plot(x_list, age_32_40_time_decay, linewidth=3, markersize=6, label='32-40 time decay', linestyle='-', marker='o',
        color=pair_colors[4])
ax.plot(x_list, age_32_40_traditional, linewidth=3, markersize=6, label='32-40 traditional', linestyle='--', marker='s',
        color=pair_colors[5])
ax.plot(x_list, age_41_73_time_decay, linewidth=3, markersize=6, label='41-73 time decay', linestyle='-', marker='o',
        color=pair_colors[6])
ax.plot(x_list, age_41_73_traditional, linewidth=3, markersize=6, label='41-73 traditional', linestyle='--', marker='s',
        color=pair_colors[7])




plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
# plt.yticks([0.0, 0.2, 0.4, 0.6], fontsize=20)
plt.xlabel('year',
           fontsize=20, labelpad=-2).set_position((0.47, -0.1))
plt.ylabel('False Negative Rate', fontsize=20, labelpad=-1)
plt.grid(True)
plt.tight_layout()
plt.legend(loc='lower left', bbox_to_anchor=(-0.142, 1), fontsize=15,
           ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
           columnspacing=0.2, borderpad=0.2, frameon=True)
plt.savefig("Acc_movielens_age_group_hoeffding_classofier.png", bbox_inches='tight')
plt.show()
