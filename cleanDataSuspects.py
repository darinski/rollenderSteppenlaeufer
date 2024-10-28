import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("\n SUSPECTS DATASET:")
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

suspectsdf.columns = suspectsdf.columns.str.lower()
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s)

# Drop NaN values
suspectsdf = suspectsdf.dropna()
suspectsdf = suspectsdf.describe()
# Check for Duplicates
duplicates = suspectsdf.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

print(suspectsdf.head())