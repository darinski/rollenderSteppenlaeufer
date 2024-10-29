import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)




crimesdf.columns = crimesdf.columns.str.lower() # lowercase column names
crimesdf = crimesdf.applymap(lambda s: s.lower() if type(s) == str else s) # lowercase entries
crimesdf['crime weapon'] = crimesdf['crime weapon'].replace('n/a', np.nan).fillna('unclear') # replace 'n/a' with NaN and fill NaN with 'unclear'
crimesdf = crimesdf[crimesdf['crime weapon'] != 'unclear'] # drop rows with 'unclear' crime weapon
crimesdf = crimesdf.dropna() # drop NaN values
crimesdf['date'] = pd.to_datetime(crimesdf['date']) # date column in date object
    


suspectsdf.columns = suspectsdf.columns.str.lower() # lowercase column names
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s) # lowercase entries
suspectsdf = suspectsdf.dropna() # drop NaN values
suspectsdf['criminal record'] = suspectsdf['criminal record'].replace({'yes': 1, 'no': 0}) # column criminal record: check
  


print(crimesdf.head())
print(suspectsdf.head())


filtered_crimes = crimesdf[
    (crimesdf['region'] == 'village') & 
    (crimesdf['time of day'] == 'day') & 
    (crimesdf['evidence found'] == 'bones') & 
    (crimesdf['crime weapon'] == 'knife')
]
monsters_involved = filtered_crimes['monster involved'].unique()
suspectsdf = suspectsdf[suspectsdf['monster'].isin(monsters_involved)]

filtered_suspects = suspectsdf[
    (suspectsdf['allergy'] != 'silver')
]

# Assuming you want to predict which monster is involved
if filtered_suspects.empty:
    print("No suspects found for the filtered crimes.")
else:
    X = filtered_suspects[['age', 'speed level', 'strength level']]  # Feature columns
    y = filtered_suspects['monster']  # Target column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predictions = np.array(predictions)
    unique, counts = np.unique(predictions, return_counts=True)
    summary = dict(zip(unique, counts))
    print(summary)



