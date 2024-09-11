import river
from river import datasets

# Load the dataset
dataset = datasets.CreditCard()
print(dataset)

for x, y in dataset:
    print(x, y)
    break

#%%

dataset = datasets.MovieLens100K()
dataset

#%%

