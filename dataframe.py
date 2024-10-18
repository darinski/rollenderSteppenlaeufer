# This files reads the csv files and creates dataframes from them

import pandas as pd


crimes = pd.read_csv('/Users/dariashevyrev/git/rollenderSteppenlaeufer/rollenderSteppenlaeufer/resources/Crimes_Dataset.csv')
crimesdf = pd.DataFrame(crimes)
print(crimesdf.head())

suspects = pd.read_csv('/Users/dariashevyrev/git/rollenderSteppenlaeufer/rollenderSteppenlaeufer/resources/Suspects_Dataset.csv')
suspectsdf = pd.DataFrame(suspects)
print(suspectsdf.head())

