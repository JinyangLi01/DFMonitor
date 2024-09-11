import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fixed_window import FPR_workload as workload
import seaborn as sns
import colorsys

sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

def scale_lightness(rgb, scale_l):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

data = pd.read_csv('../result_hoeffding_classifier.csv')
print(data["gender"].unique())
date_column = "datetime"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
date_time_format = True
time_window_str = "1 month"
monitored_groups = [{"gender": "M"}, {"gender": "F"}]

alpha = 0.5
threshold = 0.3
label_prediction = "prediction_binary"
label_ground_truth = "rating_binary"

DFMonitor, uf_list_DF, fnr_list_DF, counter_list_TP_DF, counter_list_FN_DF \
    = workload.traverse_data_DFMonitor(data,
                                       date_column,
                                       time_window_str,
                                       monitored_groups,
                                       threshold,
                                       alpha, label_prediction, label_ground_truth)

uf_list_trad, fnr_list_trad, counter_list_TP_trad, counter_list_FN_trad = workload.FPR_traditional(data, date_column,
                                                                                                   time_window_str,
                                                                                                   monitored_groups,
                                                                                                   threshold,
                                                                                                   label_prediction,
                                                                                                   label_ground_truth)

print(fnr_list_DF, fnr_list_trad)

male_time_decay = [x[0] for x in fnr_list_DF]
male_traditional = [x[0] for x in fnr_list_trad]
female_time_decay = [x[1] for x in fnr_list_DF]
female_traditional = [x[1] for x in fnr_list_trad]

x_list = np.arange(0, len(fnr_list_DF))

with open("movielens_compare_FNR_gender.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["male_time_decay", "male_traditional", "female_time_decay", "female_traditional"])
    for i in range(len(male_time_decay)):
        writer.writerow([male_time_decay[i], male_traditional[i], female_time_decay[i], female_traditional[i]])

pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
               '#287c37', '#cccc00']

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(x_list, male_time_decay, linewidth=3, markersize=6, label='Male time decay', linestyle='-', marker='o', color=pair_colors[0])
ax.plot(x_list, male_traditional, linewidth=3, markersize=6, label='Male traditional', linestyle='--', marker='s', color=pair_colors[1])
ax.plot(x_list, female_time_decay, linewidth=3, markersize=6, label='Female time decay', linestyle='-', marker='o', color=pair_colors[2])
ax.plot(x_list, female_traditional, linewidth=3, markersize=6, label='Female traditional', linestyle='--', marker='s', color=pair_colors[3])

plt.xticks(np.arange(0, len(x_list)), [], rotation=0, fontsize=20)
plt.xlabel('compas screening date starting\n from 01/01/2013, 1m interval',
           fontsize=20, labelpad=-2).set_position((0.47, -0.1))
plt.ylabel('false negative rate (FNR)', fontsize=20, labelpad=-1)
plt.grid(True)
plt.tight_layout()
plt.legend(loc='lower left', bbox_to_anchor=(-0.145, 1), fontsize=15,
               ncol=2, labelspacing=0.2, handletextpad=0.2, markerscale=1.5,
               columnspacing=0.2, borderpad=0.2, frameon=True)
plt.savefig("FPR_timedecay_traditional_gender.png", bbox_inches='tight')
plt.show()