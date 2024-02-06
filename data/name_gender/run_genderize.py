from __future__ import annotations


import get_gender
import pandas as pd


sub = pd.read_csv("baby_names_2009_2020.csv")
sub.head()


import pandas as pd
from genderize import Genderize

# Your Genderize API key
api_key = '1c13aa6f943ac21b16232d1691bb6e53'

# Initialize Genderize with your API key
genderize = Genderize(api_key=api_key)

# Use list comprehension to query Genderize for each name and get the gender
# It's efficient to send names in batches if you have a lot
genders = [genderize.get([na])[0]['gender'] for na in sub['name']]

# Add the genders as a new column to the dataframe
sub['predicted_gender'] = genders
print(sub.head())



sub.to_csv("baby_names_2009_2020_predicted.csv", index=False)

print(sub.head())


