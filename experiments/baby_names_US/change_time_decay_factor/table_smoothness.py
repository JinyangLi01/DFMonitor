import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'CMU.Serif'
# plt.rcParams['font.serif'] = 'Computer Modern'

# Set the global font family to serif
plt.rcParams['font.family'] = 'arial'

sns.set_palette("Paired")
sns.set_context("paper", font_scale=1.6)

# read FPR
df_FPR = pd.read_csv("Accuracy_time_decay_factor.csv", sep=",")

# print(df_FPR.columns)

col_data_FPR = {}

for col_name in df_FPR.columns:
    col_data_FPR[col_name] = df_FPR[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(ast.literal_eval(col_data_FPR['male_time_decay'][0])))
alpha_list = col_data_FPR['alpha']
plt.rcParams['font.size'] = 17
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#41ab5d', '#006837'])
curve_colors.append('magenta')

####################### smoothness: standard deviation of the first derative of the curve ############################


# smoothness: standard deviation of the first derative of the curve
smooth_score = {'male_acc': [], 'female_acc': []}
# print(col_data_FPR["black_time_decay"][0])
for i in range(len(alpha_list)):
    print("i = {}, alpha = {}".format(i, alpha_list[i]))
    fpr = ast.literal_eval(col_data_FPR["male_time_decay"][i])
    fpr = [x for x in fpr if x != None]
    print(fpr)
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    print("male acc smooth ", sm)
    smooth_score["male_acc"].append(sm)
    print(smooth_score)
    fpr = ast.literal_eval(col_data_FPR["female_time_decay"][i])
    fpr = [x for x in fpr if x != None]
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    smooth_score["female_acc"].append(sm)



smooth_score_results = {}
smoothness_scores = [1 / sd for sd in list(smooth_score['male_acc'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['male'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['female_acc'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['female'] = smoothness_scores_normalized_FPR




formatted_data = {key: [format(val, '.2f') for val in values] for key, values in smooth_score_results.items()}
print(formatted_data)
df = pd.DataFrame(formatted_data)


headers = list(smooth_score_results.keys())
row_names = [str(x) for x in alpha_list]
# put the last item as the first
row_names = ['t'] + row_names
row_names = row_names[:-1]
rows = zip(*formatted_data.values())
rows = list(rows)
print(rows)

print(headers, row_names)

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(8, 6))



# Create table with row labels
table = plt.table(cellText=list(rows), colLabels=['male', 'female'],
                  rowColours=[curve_colors[-1]]+curve_colors[:-1], colWidths=[0.1] * (len(headers) + 1),
                  cellLoc='center', rowLoc='center',
                  rowLabels=['   '] * len(row_names), loc='center')
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# table.auto_set_column_width(col=list(range(len(headers) + 1)))
table.PAD = 0.2
table.scale(1, 1.2)


fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.savefig("baby_names_time_decay_factor_smoothness_2f.png", bbox_inches='tight', dpi=250)
plt.show()

img = plt.imread("baby_names_time_decay_factor_smoothness_2f.png")
print(img.shape)
img_cropped = img[520:img.shape[0] - 580, 830:img.shape[1] - 840]
plt.axis('off')
plt.imshow(img_cropped)
plt.savefig("baby_names_time_decay_factor_smoothness_2f.png", bbox_inches='tight', dpi=250)



