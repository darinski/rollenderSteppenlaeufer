# This file reads the csv files and creates dataframes from them
# Studying the data and the structure of the data

import pandas as pd
import numpy as np

print("Crimes Dataset:")
crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)

print(crimesdf.head())
print("\n Shape:")
print(crimesdf.shape)
print("\n Columns:")
print(crimesdf.columns)
print("\n Info:")
print(crimesdf.info())
print("\n Describe:")
print(crimesdf.describe())
print("\n Null Values:")
print(crimesdf.isnull().sum())


print("\nSuspects Dataset \n")
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

print(suspectsdf.head())
print("\n Shape:")
print(suspectsdf.shape)
print("\n Columns:")
print(suspectsdf.columns)
print("\n Info:")
print(suspectsdf.info())
print("\n Describe:")
print(suspectsdf.describe())
print("\n Null Values:")
print(suspectsdf.isnull().sum())

