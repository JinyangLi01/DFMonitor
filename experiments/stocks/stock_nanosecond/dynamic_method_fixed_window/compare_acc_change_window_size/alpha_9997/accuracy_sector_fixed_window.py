import argparse
import colorsys
import csv

import pandas as pd
import matplotlib.pyplot as plt
from numpy import dtype
import math
import sys
sys.path.append("../../../../../../")
from algorithm.per_item import Accuracy_workload as workload
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np

# sns.set_style("whitegrid")
sns.set_palette("Paired")
sns.set_context("paper", font_scale=2)

# Set the font size for labels, legends, and titles

plt.figure(figsize=(6, 3.5))
plt.rcParams['font.size'] = 20


# # activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)

def get_integer(alpha):
    while not math.isclose(alpha, round(alpha), rel_tol=1e-9):  # Use a small tolerance
        alpha *= 10
    return int(alpha)


time_period = "14-00--14-10"
date = "20241015"
data_file_name = f"xnas-itch-{date}_{time_period}"
len_chunk = 1
# Prepare the result file for writing
data_file_name = f"../../../predict_results/prediction_result_{data_file_name}_chunk_size_{len_chunk}_v3.csv"
data = pd.read_csv(data_file_name)
#
time_start = pd.Timestamp('2024-10-15 14:00:08.00', tz='UTC')
time_end = pd.Timestamp('2024-10-15 14:00:14.00', tz='UTC')


data["ts_event"] = pd.to_datetime(data["ts_event"])
# get data between time_start and time_end
data = data[(data["ts_event"] >= time_start) & (data["ts_event"] <= time_end)]

# reset the index
data = data.reset_index(drop=True)
print("len of selected data", len(data))




print(data["sector"].unique())
date_column = "ts_event"
# get distribution of compas_screening_date
data[date_column] = pd.to_datetime(data[date_column])
print(data[date_column].min(), data[date_column].max())
date_time_format = True
# time_window_str = "1 month"
monitored_groups = [{"sector": 'Technology'}, {'sector': 'Communication Services'}, {"sector": 'Energy'},
                    {"sector": 'Consumer Defensive'}, {"sector": 'Consumer Cyclical'}, {"sector": 'Financial Services'},
                    {"sector": 'Healthcare'}, {"sector": 'Industrials'}, {"sector": "Basic Materials"}]
print(data[:5])
alpha = 0.9997


threshold = 0.4
label_prediction = "predicted_direction"
label_ground_truth = "next_price_direction"
correctness_column = "prediction_binary_correctness"
use_two_counters = True
time_unit = "10000 nanosecond"
# window_size_units = "10"
checking_interval = "100000 nanosecond"
use_nanosecond = True


parser = argparse.ArgumentParser(description="Sample script")
parser.add_argument("window_size_units", type=int)

args = parser.parse_args()
window_size_units = args.window_size_units
print(window_size_units)
window_size_units = str(window_size_units)


DFMonitor_bit, DFMonitor_counter, uf_list, accuracy_list, counter_list_correct, counter_list_incorrect, \
    window_reset_times, check_points, elapsed_time_DFMonitor_bit, elapsed_time_DFMonitor_counter, \
            total_time_insertion_bit, total_time_insertion_counter, \
            total_time_new_window_bit, total_time_new_window_counter, total_num_accuracy_check \
    = workload.traverse_data_DFMonitor_and_baseline(data, date_column,
                                                    date_time_format,
                                                    monitored_groups,
                                                    threshold, alpha, time_unit, eval(window_size_units),
                                                    checking_interval, label_prediction,
                                                    label_ground_truth, correctness_column, use_nanosecond,
                                                    use_two_counters)
# already check the correctness of the accuracy finally got
final_accuracy = accuracy_list[-1]
counter_list_correct_final = counter_list_correct[-1]
counter_list_incorrect_final = counter_list_incorrect[-1]
uf_list_final = uf_list[-1]
# print("final_accuracy", final_accuracy)
# save the result to a file
Technology_time_decay = [x[0] for x in accuracy_list]

print("number of window reset times", len(window_reset_times))

df = pd.DataFrame(accuracy_list, columns=["Technology_time_decay", "CommunicationServices_time_decay", "Energy_time_decay",
                     "ConsumerDefensive_time_decay", "ConsumerCyclical_time_decay",
                     "FinancialServices_time_decay", "Healthcare_time_decay", "Industrials",
                     "BasicMaterials_time_decay" ]               )

df["check_points"] = check_points

# Create a DataFrame from window_reset_times if they are not already in the df
window_reset_df = pd.DataFrame({"check_points": pd.to_datetime(window_reset_times)})
window_reset_df["window_reset"] = "reset"
# Set the existing DataFrame's reset flag to 'not_reset'
df["window_reset"] = "not_reset"

print(len(df), len(window_reset_df))
matches = df["check_points"].isin(window_reset_df["check_points"])
if not matches.all():
    print(f"Missing window reset times for some check_points\n")
matches2 = window_reset_df["check_points"].isin(df["check_points"])
if not matches2.all():
    print(f"Missing check_points for some window reset times\n")

# Merge with outer join, using suffixes to distinguish columns
merged_df = pd.merge(df, window_reset_df[['check_points', 'window_reset']], on="check_points", how="outer", suffixes=('_df', '_reset_df'))

# Combine the `window_reset_df` and `window_reset_reset_df` columns into a single `window_reset` column
merged_df["window_reset"] = merged_df["window_reset_df"].combine_first(merged_df["window_reset_reset_df"])
# Now, ensure that if `window_reset_reset_df` is "reset", `window_reset_df` will also be "reset"
merged_df.loc[merged_df["window_reset_reset_df"] == "reset", "window_reset"] = "reset"

print(merged_df)
print(merged_df[merged_df["window_reset"] == "reset"])
print(merged_df[merged_df["window_reset"] == "not_reset"])



# Drop the temporary `_df` and `_reset_df` columns created during the merge
merged_df.drop(columns=["window_reset_df", "window_reset_reset_df"], inplace=True)

# Sort by 'check_points' to maintain time order if needed
merged_df.sort_values(by="check_points", inplace=True)

# Ensure other columns are preserved correctly, fill accuracy values for added reset times with NaN
accuracy_columns = [col for col in df.columns if col.endswith("_time_decay")]
for col in accuracy_columns:
    merged_df[col] = merged_df[col].fillna(np.nan)

print(merged_df[:5])

# Write the updated DataFrame to a CSV
filename = (f"stocks_compare_Accuracy_sector_{date}_{time_period}_alpha_{str(get_integer(alpha))}"
            f"_time_unit_{time_unit}*{window_size_units}_check_interval_{checking_interval}_start_time_"
            f"{time_start}_end_time_{time_end}.csv")

merged_df.to_csv(filename, index=False)



with open(filename, 'r') as f:
    contents = f.read()
if "]" in contents:
    updated_contents = contents.replace("]\"", "")
    updated_contents = updated_contents.replace("\"[", "")
    with open(filename, 'w') as f:
        f.write(updated_contents)


#
# ================================== draw the figure ===========================================

df = pd.read_csv(filename)


df["check_points"] = pd.to_datetime(df["check_points"])
print(len(df))
print(df[:2])




draw_figure_start_time = pd.Timestamp('2024-10-15 14:00:09.00', tz='UTC')
draw_figure_end_time = pd.Timestamp('2024-10-15 14:00:13.00', tz='UTC')


df = df[(df["check_points"] >= draw_figure_start_time) & (df["check_points"] <= draw_figure_end_time)]

print(len(df))

# df["check_points"] = df["check_points"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y"))
check_points = df["check_points"].tolist()
x_list = np.arange(0, len(df))
curve_names = df.columns.tolist()[:-1]

curve_names = ['Technology', 'ConsumerCyclical', 'CommunicationServices', 'ConsumerDefensive',
               'Healthcare']

# pair_colors = [scale_lightness(matplotlib.colors.ColorConverter.to_rgb("navy"), 2.2), 'cyan',
#                '#287c37', '#cccc00']
pair_colors = ["blue", "darkorange", "green", "red", "cyan", "black", "magenta"]
#
#
# num_lines = len(x_list)
# pair_colors = cmaps.set1.colors

fig, ax = plt.subplots(figsize=(3.5, 1.8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

for i in range(len(curve_names)):
    ax.plot(x_list, df[curve_names[i]+"_time_decay"].tolist(), linewidth=1, markersize=1, label=curve_names[i], linestyle='-',
            marker='o', color=pair_colors[i], alpha=0.5)






plt.xlabel('',
           fontsize=14, labelpad=5).set_position((0.47, 0))
plt.ylabel('Accuracy', fontsize=13, labelpad=-1)
plt.xticks([], [])
plt.yticks([0.4, 0.6, 0.8], fontsize=13)

# # Manually place the label for 0.6 with a slight adjustment
# plt.text(-845, 0.58, '0.6', fontsize=17, va='bottom')  # Adjust the 0.6 label higher


plt.grid(True, axis='y')
plt.tight_layout()
# plt.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.4), fontsize=11,
#                ncol=2, labelspacing=0.5, handletextpad=0.2, markerscale=4, handlelength=1.5,
#                columnspacing=0.6, borderpad=0.2, frameon=True)
plt.savefig(f"StockAcc_{date}_{time_period}_a{str(get_integer(alpha))}_tu{time_unit}*{window_size_units}"
            f"_ci{checking_interval}_"
            f"start_{time_start}_end_{time_end}_fig_{draw_figure_start_time}_to_{draw_figure_end_time}.png",
            bbox_inches='tight')
plt.show()
