import csv

import pandas as pd
import matplotlib.pyplot as plt
from algorithm.fixed_window import FPR_workload as workload
import seaborn as sns

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

date_time_format = True
alpha = 0.5
threshold = 0.3

########################################## start space comparison ##########################################

# use all different combinations of race, sex and age_cat as different monitored groups
# for example, the first group is {"race": 'Caucasian'} and the second group is {"race": "Afrian-American"}


monitored_groups_set = [[{"race": 'Caucasian'}], [{"race": 'African-American'}], [{"race": "Hispanic"}],
                        [{"sex": "Male"}], [{"sex": "Female"}],

                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}],
                        # [{"race": 'African-American'}, {"race": "Hispanic"}],
                        # [{"race": 'Caucasian'}, {"race": 'Caucasian', "sex": "Male"}],
                        # [{"race": 'Hispanic'}, {"race": 'Caucasian', "sex": "Female"}],
                        # [{"race": 'African-American'}, {"sex": "Male"}],

                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"}, {"race": "Hispanic"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"}, {"race": 'Caucasian', "sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'African-American', "sex": "Female"}],

                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American'}, {"race": "Hispanic"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"}, {"race": 'African-American', "sex": "Male"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'African-American', "sex": "Female"}],


                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"},
                        #  {"race": "Hispanic"}, {"race": 'African-American', "sex": "Female"}, {"sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"}, {"sex": "Female"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"}, {"race": 'Hispanic', "sex": "Male"}],


                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": "Hispanic"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                        #  {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"}, {"race": "Hispanic"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}],

                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"race": 'African-American', "sex": "Male"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}, {"race": "African-American"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                        #  {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American', "sex": "Male"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": "Hispanic"}, {"race": 'Caucasian', "sex": "Male"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"}],

                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"race": 'African-American', "sex": "Male"}, {"sex": "Female"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"},
                        #  {"race": "African-American"}, {"sex": "Female"}],
                        # [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                        #  {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}],
                        # [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                        #  {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": "Hispanic"}, {"race": 'Caucasian', "sex": "Male"}, {"sex": "Male"}],
                        # [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        #  {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        #  {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"}, {"sex": "Male"}],

                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}],



                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],


                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],


                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],

                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}]
                        ]



file = open('space_comparison_FPR_nparray.csv', mode='w', newline='')
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
                                                  monitored_group,
                                                  threshold,
                                                  alpha)

    DFMonitor, elapsed_time_opt, total_time_new_window_opt, num_items_after_first_window_opt, num_new_windows_opt \
        = workload.traverse_data_DFMonitor_only(data,
                                                date_column,
                                                time_window_str,
                                                monitored_group,
                                                threshold,
                                                alpha)

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
