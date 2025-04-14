# Retry generating the modified bar chart with annotations

import matplotlib.pyplot as plt
import pandas as pd
from plotly import tools

# Manually create the DataFrame
data = {
    "Quarter": ["2010 Q1", "2010 Q2", "2010 Q3", "2010 Q4",
                "2011 Q1", "2011 Q2", "2011 Q3", "2011 Q4"],
    "Female Correct": [8, 8, 8, 7, 7, 5, 5, 4],
    "Female Incorrect": [2, 2, 2, 3, 3, 5, 5, 6],
    "Male Correct": [9, 7, 8, 9, 8, 4, 8, 7],
    "Male Incorrect": [1, 3, 2, 1, 2, 6, 2, 3]
}

# Create DataFrame
df_correct_incorrect = pd.DataFrame(data)

# Compute quarterly accuracy
df_correct_incorrect["Female Quarterly Accuracy"] = df_correct_incorrect["Female Correct"] / (
    df_correct_incorrect["Female Correct"] + df_correct_incorrect["Female Incorrect"]
)
df_correct_incorrect["Male Quarterly Accuracy"] = df_correct_incorrect["Male Correct"] / (
    df_correct_incorrect["Male Correct"] + df_correct_incorrect["Male Incorrect"]
)


# Compute cumulative sums for aggregated accuracy
df_correct_incorrect["Female Correct Cumulative"] = df_correct_incorrect["Female Correct"].cumsum()
df_correct_incorrect["Female Incorrect Cumulative"] = df_correct_incorrect["Female Incorrect"].cumsum()
df_correct_incorrect["Male Correct Cumulative"] = df_correct_incorrect["Male Correct"].cumsum()
df_correct_incorrect["Male Incorrect Cumulative"] = df_correct_incorrect["Male Incorrect"].cumsum()

# Compute aggregated accuracy
df_correct_incorrect["Female Aggregated Accuracy"] = df_correct_incorrect["Female Correct Cumulative"] / (
    df_correct_incorrect["Female Correct Cumulative"] + df_correct_incorrect["Female Incorrect Cumulative"]
)
df_correct_incorrect["Male Aggregated Accuracy"] = df_correct_incorrect["Male Correct Cumulative"] / (
    df_correct_incorrect["Male Correct Cumulative"] + df_correct_incorrect["Male Incorrect Cumulative"]
)
pd.set_option('display.max_rows', None)  # `None` means displaying all rows
pd.set_option('display.max_columns', None)  # `None` means displaying all columns
# Display DataFrame
print(df_correct_incorrect)





# Replot the bar chart with larger fonts for better readability

plt.figure(figsize=(4, 3))
bar_width = 0.35
x = range(len(df_correct_incorrect))

# Plot bars
plt.bar([p - bar_width/2 for p in x], df_correct_incorrect['Female Quarterly Accuracy'], width=bar_width, label='Female', color='cornflowerblue')
plt.bar([p + bar_width/2 for p in x], df_correct_incorrect['Male Quarterly Accuracy'], width=bar_width, label='Male', color='lightgreen')

# Highlight key trends from the provided table
plt.axhline(y=0.60, color='red', linestyle='--', label='Fairness Threshold', linewidth=4)

# Increase font sizes
plt.xticks(ticks=x, labels=df_correct_incorrect['Quarter'], rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Quarter', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Quarterly Accuracy by Gender (2010–2011)', fontsize=17, y=1.3, pad=-14)

# Annotate key insights with larger font sizes
# plt.text(5.2, 0.52, "Catching decreasing trend", color='blue', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(5.2, 0.75, "Large window: Missing trend", color='red', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(1.5, 0.5, "Small window: false alarm", color='green', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(6.5, 0.78, "No false alarm", color='darkorange', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(3.5, 0.3, "Dynamic fairness problem:\nCatch trends + Ignore noise", color='black', fontsize=16, fontweight='bold', bbox=dict(facecolor='gold', edgecolor='none', alpha=0.7))

# Final plot settings
plt.legend(fontsize=16, loc='upper left',
                 bbox_to_anchor=(-0.2, 1.17),
                 ncol=3, labelspacing=0.1, handletextpad=0.2, markerscale=0.1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig("hiring_example_quarterly.png", bbox_inches='tight')

plt.show()


# ============================================================================= average accuracy ======================================================

# Sample quarterly accuracy data for female and male applicants (from example)
quarters = ['2010 Q1', '2010 Q2', '2010 Q3', '2010 Q4',
            '2011 Q1', '2011 Q2', '2011 Q3', '2011 Q4']
female_acc = [0.83, 0.85, 0.80, 0.75, 0.72, 0.75, 0.73, 0.65]
male_acc =   [0.85, 0.85, 0.83, 0.83, 0.80, 0.69, 0.74, 0.78]

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Quarter': quarters,
    'Female Accuracy': female_acc,
    'Male Accuracy': male_acc
})


# Replot the bar chart with larger fonts for better readability

plt.figure(figsize=(6, 4))
bar_width = 0.3
x = range(len(df))

# Plot bars
plt.bar([p - bar_width/2 for p in x], df_correct_incorrect['Female Aggregated Accuracy'], width=bar_width, label='Female', color='cornflowerblue')
plt.bar([p + bar_width/2 for p in x], df_correct_incorrect['Male Aggregated Accuracy'], width=bar_width, label='Male', color='lightgreen')

# Highlight key trends from the provided table
plt.axhline(y=0.60, color='red', linestyle='--', label='Fairness Threshold', linewidth=4)

# Increase font sizes
plt.xticks(ticks=x, labels=df['Quarter'], rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Quarter', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Aggregated Average Accuracy by Gender (2010–2011)', fontsize=17, y=1.2, pad=-14)

# Annotate key insights with larger font sizes
# plt.text(5.2, 0.52, "Catching decreasing trend", color='blue', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(5.2, 0.75, "Large window: Missing trend", color='red', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(1.5, 0.5, "Small window: false alarm", color='green', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# plt.text(6.5, 0.78, "No false alarm", color='darkorange', fontsize=14, fontweight='bold', bbox=dict(facecolor='lightyellow', edgecolor='none', alpha=0.7))
# # plt.text(3.5, 0.3, "Dynamic fairness problem:\nCatch trends + Ignore noise", color='black', fontsize=16, fontweight='bold', bbox=dict(facecolor='gold', edgecolor='none', alpha=0.7))

# Final plot settings
plt.legend(fontsize=16, loc='upper left',
                 bbox_to_anchor=(-0.1, 1.13),
                 ncol=3, labelspacing=0.1, handletextpad=0.2, markerscale=0.1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig("hiring_example_average.png", bbox_inches='tight')

plt.show()



# Compute exponential time-decay accuracy with alpha = 0.5
alpha = 0.6
female_td_accuracy = []
male_td_accuracy = []

for i in range(len(df_correct_incorrect)):
    weights = [alpha ** (i - j) for j in range(i + 1)]

    f_correct = df_correct_incorrect.loc[:i, "Female Correct"].values
    f_incorrect = df_correct_incorrect.loc[:i, "Female Incorrect"].values
    m_correct = df_correct_incorrect.loc[:i, "Male Correct"].values
    m_incorrect = df_correct_incorrect.loc[:i, "Male Incorrect"].values

    f_td_correct = sum(f_correct[j] * weights[j] for j in range(len(weights)))
    f_td_total = sum((f_correct[j] + f_incorrect[j]) * weights[j] for j in range(len(weights)))
    m_td_correct = sum(m_correct[j] * weights[j] for j in range(len(weights)))
    m_td_total = sum((m_correct[j] + m_incorrect[j]) * weights[j] for j in range(len(weights)))

    female_td_accuracy.append(f_td_correct / f_td_total)
    male_td_accuracy.append(m_td_correct / m_td_total)

df_correct_incorrect["Female Time-Decay Accuracy"] = female_td_accuracy
df_correct_incorrect["Male Time-Decay Accuracy"] = male_td_accuracy




plt.figure(figsize=(6, 4))
bar_width = 0.3
x = range(len(df))

# Plot bars
plt.bar([p - bar_width/2 for p in x], df_correct_incorrect['Female Time-Decay Accuracy'], width=bar_width, label='Female', color='cornflowerblue')
plt.bar([p + bar_width/2 for p in x], df_correct_incorrect['Male Time-Decay Accuracy'], width=bar_width, label='Male', color='lightgreen')

# Highlight key trends from the provided table
plt.axhline(y=0.60, color='red', linestyle='--', label='Fairness Threshold', linewidth=4)

# Increase font sizes
plt.xticks(ticks=x, labels=df['Quarter'], rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Quarter', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Time-Decay Average Accuracy by Gender (2010–2011)', fontsize=17, y=1.2, pad=-14)


# Final plot settings
plt.legend(fontsize=16, loc='upper left',
                 bbox_to_anchor=(-0.1, 1.13),
                 ncol=3, labelspacing=0.1, handletextpad=0.2, markerscale=0.1,
                 columnspacing=0.5, borderpad=0.2, frameon=False)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig("hiring_example_time_decay.png", bbox_inches='tight')

plt.show()
