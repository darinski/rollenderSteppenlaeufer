# This file reads the csv files and creates dataframes from them
# Studying the data and the structure of the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import mpld3

print("\n CRIMES DATASET:")
crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)

# Shape of the Dataframe
print("\n Shape:")
print(crimesdf.shape)

# Convert column names to lowercase
crimesdf.columns = crimesdf.columns.str.lower()
print("\n Columns:")
print(crimesdf.columns)


print("\n Info:")
print(crimesdf.info())
print("\n Describe:")
print(crimesdf.describe())
print("\n Null Values:")
print(crimesdf.isnull().sum())

# Convert all entries to lowercase
crimesdf = crimesdf.applymap(lambda s: s.lower() if type(s) == str else s)
print(crimesdf.head())

# Create subplots with adjustable spacing
fig, axis = plt.subplots(4, 2, figsize=(15, 15))
gs = gridspec.GridSpec(2, 2, hspace=1, wspace=0.3)  # Adjust hspace and wspace for distances
fig.suptitle('Crime Data Analysis', fontweight='bold')

# Subplot 1: Count Crime Types
ax1 = sns.countplot(x='crime type', data=crimesdf, color='darkseagreen', ax=axis[0, 0])
axis[0, 0].set_title('Crime Type', fontweight='bold')
axis[0, 0].set_xlabel('Crime Type')
axis[0, 0].set_ylabel('Count')
# Add count labels
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=7, color='black', xytext=(0, 5),
                textcoords='offset points')
axis[0, 0].tick_params(axis='x', rotation=90)

# Subplot 2: Date Crime Counts
crimesdf['date'] = pd.to_datetime(crimesdf['date'])
# Group by month
crimesdf['year'] = crimesdf['date'].dt.year
crimesdf['month'] = crimesdf['date'].dt.month
# Plot Subplot 2
crimes_per_month_year = crimesdf.groupby(['year', 'month']).size().unstack(level=0)
for year in crimes_per_month_year.columns:
    axis[0, 1].plot(crimes_per_month_year.index, crimes_per_month_year[year], marker='o', label=year)
axis[0, 1].set_title('Crimes Over Time', fontweight='bold')
axis[0, 1].set_xlabel('Month')
axis[0, 1].set_ylabel('Number of Crimes')
axis[0, 1].legend(title='Year')
axis[0, 1].set_xticks(range(1, 13))
axis[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axis[0, 1].grid(True)

# Subplot 3: Monster in Relation to Days of Investigation
sns.boxplot(x='monster involved', y='days of investigation', data=crimesdf, color='thistle', ax=axis[1, 0])
axis[1, 0].set_title('Monster in Relation to Days of Investigation', fontweight='bold')
axis[1, 0].set_xlabel('Monster')
axis[1, 0].set_ylabel('Days to Solve')
axis[1, 0].grid(True)

plt.tight_layout()
plt.show()
mpld3.display()


print("\n SUSPECTS DATASET:")
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

# Shape of the Dataframe
print("\n Shape:")
print(suspectsdf.shape)

# Convert column names to lowercase
suspectsdf.columns = suspectsdf.columns.str.lower()
print("\n Columns:")
print(suspectsdf.columns)


print("\n Info:")
print(suspectsdf.info())
print("\n Describe:")
print(suspectsdf.describe())
print("\n Null Values:")
print(suspectsdf.isnull().sum())


# Convert all entries to lowercase
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s)
print(suspectsdf.head())



