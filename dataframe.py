# This files reads the csv files and creates dataframes from them

import pandas as pd


crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)
print(crimesdf.head())

suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)
print(suspectsdf.head())

