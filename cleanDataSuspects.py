import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("\n SUSPECTS DATASET:")
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

suspectsdf.head()

suspectsdf.columns = suspectsdf.columns.str.lower()
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s)

# Drop NaN values
suspectsdf = suspectsdf.dropna()
# Check for Duplicates
duplicates = suspectsdf.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

suspectsdf.info()

# Column criminal record: check unique values
suspectsdf[suspectsdf['criminal record'] == 'yes'] = 1
print(suspectsdf.head())