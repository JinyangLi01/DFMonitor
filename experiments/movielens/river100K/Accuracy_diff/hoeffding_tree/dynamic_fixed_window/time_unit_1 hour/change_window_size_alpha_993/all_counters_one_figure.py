import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define your window size units and alpha
window_size_unit_list = ['1', '24', '24*7', '24*7*4', '24*7*4*2', '24*7*4*3']
result_file_list = dict()
alpha = 993

# Check if files exist, run the script if they don't, and read the CSV files
for window_size in window_size_unit_list:
    file_name = f"movielens_compare_Counters_hoeffding_classifier_gender_per_item_alpha_{alpha}_time_unit_1 hour*{window_size}_check_interval_1 hour.csv"

    if not os.path.exists(file_name):
        subprocess.run(["python3", "accuracy_gender_per_item_1hour.py", window_size])
        # Check again after running the script
        if not os.path.exists(file_name):
            raise Exception(f"Error: {file_name} was not created by the script.")

    df = pd.read_csv(file_name)
    result_file_list[window_size] = df

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
plt.rcParams['font.size'] = 10

# Define your custom curve colors
curve_colors = sns.color_palette(
    ['#0000ff', 'cyan', '#57d357', '#004b00', 'darkorange', 'firebrick', 'blueviolet', 'magenta'])

# Plot each DataFrame with a different color
for i, window_size in enumerate(window_size_unit_list):
    df = result_file_list[window_size]

    # Ensure there's data to plot
    if not df.empty:
        ax.plot(range(len(df)), df['counter_list_correct_Male'], label=f'Window: {window_size}', color=curve_colors[i])
        ax.plot(range(len(df)), df['counter_list_correct_Female'], color=curve_colors[i], linestyle='dashed')
        ax.plot(range(len(df)), df['counter_list_incorrect_Male'], color=curve_colors[i], linestyle='dashdot')
        ax.plot(range(len(df)), df['counter_list_incorrect_Female'], color=curve_colors[i], linestyle='dotted')

# Add labels, title, and legend
ax.set_xlabel("X-Axis Label")
ax.set_ylabel("Y-Axis Label")
ax.set_title("Comparison of Accuracy Over Different Window Sizes")
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
