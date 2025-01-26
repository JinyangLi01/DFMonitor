# I want to draw a chart comparing two lists of docts.  the y-axis is the value of the dots, and x-axis is the timestamp of each dot. I have around 1k dots so 1k timestamps in x-axis so even with a very. wide figure, the chart still seems crowded. Also the two lists have similar values and the values have some turbulance so the two colors of dots overlap with each other and it is hard to visiualize the overall trend
# When dealing with large datasets and overlapping points, there are several techniques you can use to make your chart clearer and the trends more discernible:
# in this script I'm trying different ways to make the chart more readable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("../movielens_compare_Accuracy_hoeffding_classifier_gender_per_item_50.csv")
print(df)


timestamps = np.arange(0, len(df))
male_time_decay = df["male_time_decay"].tolist()
female_time_decay = df["female_time_decay"].tolist()

values_list1 = male_time_decay
values_list2 = female_time_decay

# Create a DataFrame for easier handling
df = pd.DataFrame({
    'Timestamp': pd.to_datetime(timestamps),
    'Value1': values_list1,
    'Value2': values_list2
})
df.set_index('Timestamp', inplace=True)

# Option 1: Smoothing with Moving Average
window_size = 50
df['Value1_Smooth'] = df['Value1'].rolling(window=window_size).mean()
df['Value2_Smooth'] = df['Value2'].rolling(window=window_size).mean()

plt.figure(figsize=(15, 7))
plt.plot(df.index, df['Value1_Smooth'], label='Series 1 Smoothed')
plt.plot(df.index, df['Value2_Smooth'], label='Series 2 Smoothed')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Smoothed Time Series')
plt.legend()
plt.show()

# Option 2: Interactive Plot with Plotly
import plotly.express as px
fig = px.line(df, x=df.index, y=['Value1_Smooth', 'Value2_Smooth'], labels={'value':'Value', 'variable':'Series'})
fig.update_layout(title='Interactive Smoothed Time Series')
fig.show()
