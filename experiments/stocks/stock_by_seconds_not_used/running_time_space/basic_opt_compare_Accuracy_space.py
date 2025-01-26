import csv

import pandas as pd
import matplotlib.pyplot as plt
from algorithm.fixed_window import Accuracy_workload as workload
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20

data = pd.read_csv('../../../data/name_gender/baby_names_1880_2020_US_predicted.csv')
print(data["sex"].unique())
date_column = "year"
date_time_format = False
time_window_str = "10"

monitored_groups = [{"sector": 'Technology'}, {"sector": 'Consumer Cyclical'}, {'sector': 'Communication Services'}]
print(data[:5])
alpha = 0.997


window_size_str_list = ["2 year", "5 year", "10 year", "20 year", "40 year", "60 year", "80 year", "100 year"]
window_size_str_list_brev = ["2", "5", "10", "20", "40", "60", "80", "100"]
threshold = 0.3
alpha = 0.5
label_prediction = "predicted_gender"
label_ground_truth = "sex"

########################################## start space comparison ##########################################

# use all different combinations of race, sex and age_cat as different monitored groups
# for example, the first group is {"race": 'Caucasian'} and the second group is {"race": "Afrian-American"}


monitored_groups_set = [
    [{"sex": "male"}], [{"sex": "female"}],
    [{"sex": "female"}, {"sex": "male"}],

]

file = open('space_comparison_Accuracy.csv', mode='w', newline='')
writer = csv.writer(file)
writer.writerow(["monitored_group", "size_base", "size_opt", "(size_base - size_opt) / size_base"])

num_repeat = 1

total_elapsed_time_opt = 0
total_elapsed_time_base = 0

for monitored_group in monitored_groups_set:
    # print("monitored_group: ", monitored_group)

    (DFMonitor_baseline, elapsed_time_base, total_time_new_window_base,
     num_items_after_first_window_base, num_new_windows_base) \
        = workload.traverse_data_DF_baseline_only(data,
                                                  date_column,
                                                  time_window_str,
                                                  date_time_format,
                                                  monitored_group,
                                                  threshold,
                                                  alpha, label_prediction, label_ground_truth)

    DFMonitor, elapsed_time_opt, total_time_new_window_opt, num_items_after_first_window_opt, num_new_windows_opt \
        = workload.traverse_data_DFMonitor_only(data,
                                                date_column,
                                                time_window_str,
                                                date_time_format,
                                                monitored_group,
                                                threshold,
                                                alpha, label_prediction, label_ground_truth)

    total_elapsed_time_opt += elapsed_time_opt
    total_elapsed_time_base += elapsed_time_base

    # size_base = asizeof.asizeof(DFMonitor_baseline)
    # size_opt = asizeof.asizeof(DFMonitor)

    size_base = DFMonitor_baseline.get_size()
    size_opt = DFMonitor.get_size()

    print("\nmonitored_group: ", monitored_group)
    print("size of base: ", size_base)
    print("size of opt: ", size_opt)
    print("(size of base - size of opt) / size of base: ", (size_base - size_opt) / size_base)
    writer.writerow([monitored_group, size_base, size_opt, (size_base - size_opt) / size_base])

    # ================================================
    print(DFMonitor.get_size())
