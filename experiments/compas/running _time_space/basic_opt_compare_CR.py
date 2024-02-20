import colorsys
import csv

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithm import CR_workload as workload
import seaborn as sns
from matplotlib import rc
from pympler import asizeof


# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20


data = pd.read_csv('../../../data/compas/preprocessed/cox-parsed_7214rows_with_labels_sorted_by_dates.csv')
print(data["race"].unique())
# get distribution of compas_screening_date
data['compas_screening_date'] = pd.to_datetime(data['compas_screening_date'])
# data['compas_screening_date'].hist()
date_column = "compas_screening_date"
time_window_str = "1 month"
monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Asian"}, {"race": "Hispanic"},
                    {"race": "Other"}, {"race": "Native American"}]
date_time_format = True
alpha = 0.5
threshold = 0.3


DFMonitor_baseline, uf_list_baseline, counter_list_baseline, cr_list_baseline \
    = workload.traverse_data_DFMonitor_baseline(data, date_column,
                                                time_window_str, date_time_format,
                                                monitored_groups,
                                                threshold,
                                                alpha)

# use CR for compas dataset, a time window = 1 month, record the result of each uf in each month and draw a plot
DFMonitor, uf_list_DF, cr_list_DF, counter_list_DF = workload.traverse_data_DFMonitor(data, date_column,
                                                                                      time_window_str,
                                                                                      date_time_format,
                                                                                      monitored_groups,
                                                                                      threshold,
                                                                                      alpha)

size_base = asizeof.asizeof(DFMonitor_baseline)
size_opt = asizeof.asizeof(DFMonitor)
print("size of base: ", size_base)
print("size of opt: ", size_opt)