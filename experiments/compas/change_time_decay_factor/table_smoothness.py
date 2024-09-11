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
df_FPR = pd.read_csv("FPR_time_decay_factor.csv", sep=",")

# print(df_FPR.columns)

col_data_FPR = {}

for col_name in df_FPR.columns:
    col_data_FPR[col_name] = df_FPR[col_name].tolist()

# read CR
df_CR = pd.read_csv("CR_time_decay_factor.csv", sep=",")
# print(df_CR.head)
col_data_CR = {}
for col_name in df_CR.columns:
    col_data_CR[col_name] = df_CR[col_name].tolist()

# draw the plot

x_list = np.arange(0, len(ast.literal_eval(col_data_FPR['black_time_decay'][0])))
alpha_list = col_data_FPR['alpha']
plt.rcParams['font.size'] = 17
curve_colors = sns.color_palette(palette=['navy', 'blue', 'cornflowerblue', 'lightgreen', '#41ab5d', '#006837'])
curve_colors.append('magenta')

####################### smoothness: standard deviation of the first derative of the curve ############################


# smoothness: standard deviation of the first derative of the curve
smooth_score = {'black_FPR': [], 'white_FPR': [], 'hispanic_FPR': [], 'black_CR': [], 'white_CR': [], 'hispanic_CR': []}
# print(col_data_FPR["black_time_decay"][0])
for i in range(len(alpha_list)):
    print("i = {}, alpha = {}".format(i, alpha_list[i]))
    fpr = ast.literal_eval(col_data_FPR["black_time_decay"][i])
    fpr = [x for x in fpr if x != None]
    print(fpr)
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    print("black_time_decay FPR smooth ", sm)
    smooth_score["black_FPR"].append(sm)
    print(smooth_score)
    smooth_score["black_CR"].append(
        np.std(np.diff(ast.literal_eval(col_data_CR["black_time_decay"][i])) / np.diff(x_list)))
    fpr = ast.literal_eval(col_data_FPR["white_time_decay"][i])
    fpr = [x for x in fpr if x != None]
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    smooth_score["white_FPR"].append(sm)
    smooth_score["white_CR"].append(
        np.std(np.diff(ast.literal_eval(col_data_CR["white_time_decay"][i])) / np.diff(x_list)))
    fpr = ast.literal_eval(col_data_FPR["hispanic_time_decay"][i])
    fpr = [x for x in fpr if x != None]
    dydx = np.diff(fpr) / np.diff(np.arange(0, len(fpr)))
    sm = np.std(dydx)
    smooth_score["hispanic_FPR"].append(sm)
    smooth_score["hispanic_CR"].append(
        np.std(np.diff(ast.literal_eval(col_data_CR["hispanic_time_decay"][i])) / np.diff(x_list)))


smooth_score_results = {}
smoothness_scores = [1 / sd for sd in list(smooth_score['black_FPR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['B, F'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['white_FPR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['W, F'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['hispanic_FPR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['H, F'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['black_CR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['B, C'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['white_CR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['W, C'] = smoothness_scores_normalized_FPR

smoothness_scores = [1 / sd for sd in list(smooth_score['hispanic_CR'])]  # Example data
# Normalize based on the maximum value observed
max_smoothness = max(smoothness_scores)
smoothness_scores_normalized_FPR = [score / max_smoothness for score in smoothness_scores]
print("Normalized Smoothness Scores :", smoothness_scores_normalized_FPR)
smooth_score_results['H, C'] = smoothness_scores_normalized_FPR


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

plt.axis('off')

# Create table with row labels
table = plt.table(cellText=list(rows), colLabels=['B, F', 'W, F', 'H, F', 'B, C', 'W, C', 'H, C'],
                  rowColours=[curve_colors[-1]]+curve_colors[:-1], colWidths=[0.09] * (len(headers) + 1),
                  cellLoc='center', rowLoc='center',
                  rowLabels=['   '] * len(row_names), loc='center')
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# table.auto_set_column_width(col=list(range(len(headers) + 1)))
table.PAD = 0.2
table.scale(1, 1.2)
# table.auto_set_font_size(True)

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

plt.savefig("compas_time_decay_factor_smoothness_2f.png", bbox_inches='tight', dpi=250)
plt.show()

img = plt.imread("compas_time_decay_factor_smoothness_2f.png")
print(img.shape)
img_cropped = img[500:img.shape[0] - 500, 410:img.shape[1] - 470]
plt.axis('off')
plt.imshow(img_cropped)
plt.savefig("compas_time_decay_factor_smoothness_2f.png", bbox_inches='tight', dpi=250)



