import matplotlib.pyplot as plt
import pandas as pd

# Recreate the data
data = {
    "Quarter": ["Q1", "Q2", "Q3", "Q4",
                "Q1", "Q2", "Q3", "Q4"],
    "Female Correct": [8, 8, 8, 7, 7, 5, 5, 4],
    "Female Incorrect": [2, 2, 2, 3, 3, 5, 5, 6],
    "Male Correct": [9, 7, 8, 9, 8, 4, 8, 7],
    "Male Incorrect": [1, 3, 2, 1, 2, 6, 2, 3]
}

# Create DataFrame
df = pd.DataFrame(data)

# Compute quarterly accuracy
df["Female Quarterly Accuracy"] = df["Female Correct"] / (df["Female Correct"] + df["Female Incorrect"])
df["Male Quarterly Accuracy"] = df["Male Correct"] / (df["Male Correct"] + df["Male Incorrect"])

# Compute aggregated accuracy
df["Female Correct Cumulative"] = df["Female Correct"].cumsum()
df["Female Incorrect Cumulative"] = df["Female Incorrect"].cumsum()
df["Male Correct Cumulative"] = df["Male Correct"].cumsum()
df["Male Incorrect Cumulative"] = df["Male Incorrect"].cumsum()

df["Female Aggregated Accuracy"] = df["Female Correct Cumulative"] / (
    df["Female Correct Cumulative"] + df["Female Incorrect Cumulative"]
)
df["Male Aggregated Accuracy"] = df["Male Correct Cumulative"] / (
    df["Male Correct Cumulative"] + df["Male Incorrect Cumulative"]
)

# Compute exponential time-decay accuracy
alpha = 0.6
female_td_accuracy = []
male_td_accuracy = []

for i in range(len(df)):
    weights = [alpha ** (i - j) for j in range(i + 1)]
    f_correct = df.loc[:i, "Female Correct"].values
    f_incorrect = df.loc[:i, "Female Incorrect"].values
    m_correct = df.loc[:i, "Male Correct"].values
    m_incorrect = df.loc[:i, "Male Incorrect"].values

    f_td_correct = sum(f_correct[j] * weights[j] for j in range(len(weights)))
    f_td_total = sum((f_correct[j] + f_incorrect[j]) * weights[j] for j in range(len(weights)))
    m_td_correct = sum(m_correct[j] * weights[j] for j in range(len(weights)))
    m_td_total = sum((m_correct[j] + m_incorrect[j]) * weights[j] for j in range(len(weights)))

    female_td_accuracy.append(f_td_correct / f_td_total)
    male_td_accuracy.append(m_td_correct / m_td_total)

df["Female Time-Decay Accuracy"] = female_td_accuracy
df["Male Time-Decay Accuracy"] = male_td_accuracy

# Plot all three subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 2.1), sharey=True)
bar_width = 0.3
x = range(len(df))

# Common settings
quarters = df['Quarter']
fairness_threshold = 0.60
colors = {'Female': 'cornflowerblue', 'Male': 'lightgreen'}

# Quarterly Accuracy
axes[0].bar([p - bar_width/2 for p in x], df['Female Quarterly Accuracy'], width=bar_width, label='Female', color=colors['Female'])
axes[0].bar([p + bar_width/2 for p in x], df['Male Quarterly Accuracy'], width=bar_width, label='Male', color=colors['Male'])
axes[0].axhline(y=fairness_threshold, color='red', linestyle='--', linewidth=3)
axes[0].set_title('Quarterly Accuracy', fontsize=16)
axes[0].set_xticks(x)
axes[0].set_xticklabels(quarters, rotation=0, fontsize=16)
axes[0].tick_params(labelsize=16)
# axes[0].set_yticklabels([0, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=16)
axes[0].set_yticks([0, 0.2, 0.4,  0.6, 0.8, 1.0])

# Aggregated Accuracy
axes[1].bar([p - bar_width/2 for p in x], df['Female Aggregated Accuracy'], width=bar_width, color=colors['Female'])
axes[1].bar([p + bar_width/2 for p in x], df['Male Aggregated Accuracy'], width=bar_width, color=colors['Male'])
axes[1].axhline(y=fairness_threshold, color='red', linestyle='--', linewidth=3)
axes[1].set_title('Aggregated Accuracy', fontsize=16)
axes[1].set_xticks(x)
axes[1].set_xticklabels(quarters, rotation=0, fontsize=16)
axes[1].tick_params(labelsize=16)
# axes[1].set_yticklabels([0, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=16)
axes[1].set_yticks([0, 0.2, 0.4,  0.6, 0.8, 1.0])

# Time-Decay Accuracy
axes[2].bar([p - bar_width/2 for p in x], df['Female Time-Decay Accuracy'], width=bar_width, color=colors['Female'])
axes[2].bar([p + bar_width/2 for p in x], df['Male Time-Decay Accuracy'], width=bar_width, color=colors['Male'])
axes[2].axhline(y=fairness_threshold, color='red', linestyle='--', linewidth=3)
axes[2].set_title('Time-Decay Accuracy', fontsize=16)
axes[2].set_xticks(x)
axes[2].set_xticklabels(quarters, rotation=0, fontsize=16)
axes[2].tick_params(labelsize=16)
# axes[2].set_yticklabels([0, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=16)
for ax in axes:
    ax.tick_params(labelleft=True)  # force y-axis tick labels to show
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])  # optional: set consistent ticks


plt.subplots_adjust(wspace=0.2)

# Shared y-label
fig.text(0.07, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=18)

fig.text(0.17, -0.15, '2010', rotation='horizontal', fontsize=18)
fig.text(0.27, -0.15, '2011', rotation='horizontal', fontsize=18)


fig.text(0.44, -0.15, '2010', rotation='horizontal', fontsize=18)
fig.text(0.54, -0.15, '2011', rotation='horizontal', fontsize=18)


fig.text(0.71, -0.15, '2010', rotation='horizontal', fontsize=18)
fig.text(0.82, -0.15, '2011', rotation='horizontal', fontsize=18)




# Shared legend
# fig.legend(['Female', 'Male', 'Fairness Threshold'], loc='upper center',
#            bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=16, frameon=False)

axes[0].legend(['Fairness Threshold', 'Female', 'Male'],
               loc='upper center',
               bbox_to_anchor=(1, 1.4),
               ncol=3,
               fontsize=14,
               frameon=False)

plt.savefig("hiring_example_all_views.png", bbox_inches='tight')
plt.show()
