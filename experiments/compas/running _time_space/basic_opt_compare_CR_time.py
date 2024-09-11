import csv

import pandas as pd
import matplotlib.pyplot as plt
from algorithm.fixed_window import CR_workload as workload
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
# monitored_groups = [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}]
date_time_format = True
alpha = 0.5
threshold = 0.3


########################################## start running time ##########################################


def one_run(monitored_groups, num_repeat=10):

    total_elapsed_time_opt = 0
    total_elapsed_time_base = 0
    total_elasped_time_new_window_opt = 0
    total_elasped_time_new_window_base = 0
    total_elapsed_time_insertion_opt = 0
    total_elapsed_time_insertion_base = 0
    num_items_after_first_window_base = 0
    num_items_after_first_window_opt = 0
    total_time_new_window_base = 0
    total_time_new_window_opt = 0
    num_new_windows_base = 0
    num_new_windows_opt = 0

    for i in range(num_repeat):
        (DFMonitor_baseline, elapsed_time_base, total_time_new_window_base,
         num_items_after_first_window_base, num_new_windows_base) \
            = workload.traverse_data_DFMonitor_baseline_only(data, date_column,
                                                             time_window_str, date_time_format,
                                                             monitored_groups, threshold, alpha)

        DFMonitor, elapsed_time_opt, total_time_new_window_opt, num_items_after_first_window_opt, num_new_windows_opt \
            = workload.traverse_data_DFMonitor_only(data, date_column,
                                                    time_window_str, date_time_format,
                                                    monitored_groups, threshold, alpha)

        total_elapsed_time_opt += elapsed_time_opt
        total_elapsed_time_base += elapsed_time_base
    print("monitored groups: ", monitored_groups)
    # print("avg elapsed_time_base out of {} executions: {}".format(num_repeat, total_elapsed_time_base / num_repeat))
    # print("avg elapsed_time_opt out of {} executions: {}".format(num_repeat, total_elapsed_time_opt / num_repeat))
    # print("num_items_after_first_window_base: ", num_items_after_first_window_base)
    # print("avg time of an operation base: ", total_elapsed_time_base / (num_items_after_first_window_base * num_repeat))
    # print("avg time of an operation opt: ", total_elapsed_time_opt / (num_items_after_first_window_base * num_repeat))

    avg_newwindow_base = total_time_new_window_base / (num_repeat * num_new_windows_base)
    avg_newwindow_opt = total_time_new_window_opt / (num_new_windows_opt * num_repeat)
    avg_insertion_base = (total_elapsed_time_base - total_time_new_window_base) / \
                            (num_items_after_first_window_base * num_repeat - (num_repeat * num_new_windows_base))
    avg_insertion_opt = (total_elapsed_time_opt - total_time_new_window_opt) / \
                            (num_items_after_first_window_opt * num_repeat - (num_new_windows_opt * num_repeat))

    return avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt




# 1-13
monitored_groups_set = [[{"race": 'Caucasian'}], [{"race": 'African-American'}], [{"race": "Hispanic"}],
                        [{"race": 'Caucasian', "sex": "Male"}], [{"race": 'Caucasian', "sex": "Female"}],

                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}],
                        [{"race": 'African-American'}, {"race": "Hispanic"}],
                        [{"race": 'Caucasian'}, {"race": 'Caucasian', "sex": "Male"}],
                        [{"race": 'Hispanic'}, {"race": 'Caucasian', "sex": "Female"}],
                        [{"race": 'African-American'}, {"sex": "Female"}],

                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"}, {"race": "Hispanic"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"}, {"race": 'Caucasian', "sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'African-American', "sex": "Female"}],

                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American'}, {"race": "Hispanic"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"}, {"race": 'African-American', "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}],


                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"race": 'African-American', "sex": "Female"}, {"sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"}, {"sex": "Female"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"}, {"race": 'Hispanic', "sex": "Male"}],


                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": "Hispanic"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                         {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"}, {"race": "Hispanic"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}],

                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"race": 'African-American', "sex": "Male"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}, {"race": "African-American"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                         {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American', "sex": "Male"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic"}, {"race": 'Caucasian', "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"}],
        # 8
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Female"}, {"race": 'Caucasian', "sex": "Male"},
                         {"race": "Hispanic"}, {"race": 'African-American', "sex": "Male"}, {"sex": "Female"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"},
                         {"race": "African-American"}, {"sex": "Female"}],
                        [{"race": 'Caucasian'}, {"race": 'African-American'}, {"race": "Hispanic"}, {"sex": "Male"},
                         {"sex": "Female"}, {"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}],
                        [{"race": 'Caucasian', "sex": "Male"}, {"race": 'African-American'}, {"sex": "Female"},
                         {"race": 'African-American', "sex": "Male"}, {"race": 'African-American', "sex": "Female"},
                         {"race": "Hispanic"}, {"race": 'Caucasian', "sex": "Male"}, {"sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"}, {"sex": "Male"}],
        # 9
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                        {"sex": "Male"}, {"race": "Hispanic"}],
[{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                        {"sex": "Male"}, {"race": "Hispanic"}],
[{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                        {"sex": "Male"}, {"race": "Hispanic"}],
[{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                        {"sex": "Male"}, {"race": "Hispanic"}],
[{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                        {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                        {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                        {"sex": "Male"}, {"race": "Hispanic"}],
        # 10
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}],

        # 11
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'}],
        # 12
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}],
        # 13
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}],
                        [{"race": 'African-American', "sex": "Male"}, {"race": 'Caucasian', "sex": "Female"},
                         {"race": 'Caucasian', "sex": "Female"}, {"race": 'African-American', "sex": "Female"},
                         {"race": 'Hispanic', "sex": "Male"}, {"race": "African-American"}, {"sex": "Female"},
                         {"sex": "Male"}, {"race": "Hispanic"}, {"race": 'Caucasian'}, {"race": 'African-American'},
                         {"race": "Hispanic", "sex": "Male"}, {"race": "Hispanic", "sex": "Female"}]


                        ]
avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt = 0, 0, 0, 0
result_insertion_base = []
result_insertion_opt = []
result_newwindow_base = []
result_newwindow_opt = []
repeat = 10
for i, group in enumerate(monitored_groups_set):
    print("group: ", group, "i: ", i)
    a, b, c, d = one_run(group, repeat)
    avg_insertion_base += a
    avg_insertion_opt += b
    avg_newwindow_base += c
    avg_newwindow_opt += d
    if (i+1) % 5 == 0:
        result_insertion_base.append(avg_insertion_base / 5)
        result_insertion_opt.append(avg_insertion_opt / 5)
        result_newwindow_base.append(avg_newwindow_base / 5)
        result_newwindow_opt.append(avg_newwindow_opt / 5)
        avg_insertion_base, avg_insertion_opt, avg_newwindow_base, avg_newwindow_opt = 0, 0, 0, 0
with open("running_time_CR.csv", "a", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["avg_insertion_base", "avg_insertion_opt", "avg_newwindow_base", "avg_newwindow_opt"])
    writer.writerow([result_insertion_base, result_insertion_opt, result_newwindow_base, result_newwindow_opt])
    # the difference between two results by percentage
    writer.writerow([(result_insertion_base[i] - result_insertion_opt[i]) / result_insertion_base[i] for i in range(len(result_insertion_base))])
    writer.writerow([(result_newwindow_base[i] - result_newwindow_opt[i]) / result_newwindow_base[i] for i in range(len(result_newwindow_base))])

print("avg_insertion_base: ", result_insertion_base)
print("avg_insertion_opt: ", result_insertion_opt)
print("avg_newwindow_base: ", result_newwindow_base)
print("avg_newwindow_opt: ", result_newwindow_opt)
print("the difference between two results by percentage insertion: ",
      [(result_insertion_base[i] - result_insertion_opt[i]) / result_insertion_base[i] for i in range(len(result_insertion_base))])
print("the difference between two results by percentage new window: ",
      [(result_newwindow_base[i] - result_newwindow_opt[i]) / result_newwindow_base[i] for i in range(len(result_newwindow_base))])



########################################## end running time ##########################################
# repeat 1000
# avg elapsed_time_base out of 1000 executions: 0.13568266344070434
# avg elapsed_time_opt out of 1000 executions: 0.13563484144210816
