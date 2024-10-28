import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n CRIMES DATASET:")
crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)

crimesdf.columns = crimesdf.columns.str.lower()
crimesdf = crimesdf.applymap(lambda s: s.lower() if type(s) == str else s)

print("\n Null Values:")
print(crimesdf.isnull().sum())

print(crimesdf.head())

print("\n Monsters:" + str(crimesdf['monster involved'].unique()))
print("\n Regions:" + str(crimesdf['region'].unique()))
print("\n Crime Types:" + str(crimesdf['crime type'].unique()))
print("\n Crime Weapons:" + str(crimesdf['crime weapon'].unique()))
nan_count = crimesdf['crime weapon'].isna().sum()
# Count 'n/a' values (assuming 'n/a' is a string)
n_a_count = (crimesdf['crime weapon'] == 'n/a').sum()
print(f"NaN count: {nan_count}")
print(f"'n/a' count: {n_a_count}")
print("\n Time of Day:" + str(crimesdf['time of day'].unique()))
print("\n Evidence Found:" + str(crimesdf['evidence found'].unique()))

# Replace 'n/a' with NaN and fill NaN with 'unclear'
crimesdf['crime weapon'] = crimesdf['crime weapon'].replace('n/a', np.nan).fillna('unclear')
uncleardf = crimesdf[crimesdf['crime weapon'] == 'unclear']
unique_monsters_count = uncleardf['monster involved'].value_counts()
print(unique_monsters_count)

crimesdf.info()
# Drop rows with 'unclear' crime weapon (formerly 'n/a' & NaN)
crimesdf = crimesdf[crimesdf['crime weapon'] != 'unclear']
# Drop NaN values
crimesdf = crimesdf.dropna()
# Date column in date object
crimesdf['date'] = pd.to_datetime(crimesdf['date'])
# Check for Duplicates
duplicates = crimesdf.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

crimesdf.info()

fig, axis = plt.subplots(4, 2, figsize=(15, 15))

# Evidence Found per Monster
evidence = crimesdf.groupby('monster involved')['evidence found'].value_counts()
print(evidence.describe())
print("\nEvidence Found per Monster:")

# Mean of Days of Investigation By Monster
mean_days = crimesdf.groupby('monster involved')['days of investigation'].mean()
print("\nMean Days of Investigation By Monster:")
print(mean_days)

# Crime Types per Monster
crime_types = crimesdf.groupby('monster involved')['crime type'].value_counts()
print("\nCrime Types per Monster:")
print(crime_types)

# Regions per Monster
regions = crimesdf.groupby('monster involved')['region'].value_counts()
print("\nRegions per Monster:")
print(regions)

# Time of Day per Monster
time_of_day = crimesdf.groupby('monster involved')['time of day'].value_counts()
print("\nTime of Day per Monster:")
print(time_of_day)






print("\n SUSPECTS DATASET:")
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

suspectsdf.columns = suspectsdf.columns.str.lower()
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s)

# Drop NaN values
suspectsdf = suspectsdf.dropna()
suspectsdf = suspectsdf.describe()

print(suspectsdf.head())