import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV




# Load the data
crimes = pd.read_csv('rollenderSteppenlaeufer/resources/Crimes_Dataset_fixed.csv')
crimesdf = pd.DataFrame(crimes)
suspects = pd.read_csv('rollenderSteppenlaeufer/resources/Suspects_Dataset_fixed.csv')
suspectsdf = pd.DataFrame(suspects)

# Lowercase column names
crimesdf.columns = crimesdf.columns.str.lower()
suspectsdf.columns = suspectsdf.columns.str.lower()

# Lowercase entries
crimesdf = crimesdf.applymap(lambda s: s.lower() if type(s) == str else s)
suspectsdf = suspectsdf.applymap(lambda s: s.lower() if type(s) == str else s)

# Merge the datasets
merged = pd.merge(crimesdf, suspectsdf, left_on='index_crimes', right_on='index_monster', how='inner')#

# Clean entries & columns
merged.drop(['index_monster', 'monster involved'], axis=1, inplace=True) # drop duplicate columns
merged['criminal record'] = merged['criminal record'].replace({'yes': 1, 'no': 0}) # column criminal record change
merged['gender'] = merged['gender'].replace({'f': 1, 'm': 0}) # replace f = 1 & m = 0 to have numerical values
merged['date'] = pd.to_datetime(merged['date']) # date column in date object

# Handling missing values
merged['crime weapon'] = merged['crime weapon'].replace('n/a', np.nan).fillna('none') # crime weapon not mentioned replaced with none
merged.dropna(inplace=True) # drop NaN values

print(merged.head(20))
print(merged.info())

# 1. Cross-tabulation: Crime Type by Region
crime_region_crosstab = pd.crosstab(merged['crime type'], merged['region'])
print("Crime Type by Region:\n", crime_region_crosstab)

# 2. Cross-tabulation: Crime Weapon by Time of Day
weapon_time_crosstab = pd.crosstab(merged['crime weapon'], merged['time of day'])
print("\nCrime Weapon by Time of Day:\n", weapon_time_crosstab)

# 3. Cross-tabulation: Crime Weapon by Allergy Involved
weapon_allergy_crosstab = pd.crosstab(merged['crime weapon'], merged['allergy'])
print("\nCrime Weapon by Allergy:\n", weapon_allergy_crosstab)

# 4. Cross-tabulation: Monster by Evidence Found
monster_evidence_crosstab = pd.crosstab(merged['monster'], merged['evidence found'])
print("\nMonster by Evidence Found:\n", monster_evidence_crosstab)
# Visualize with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(monster_evidence_crosstab, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Monster by Evidence Found")
plt.xlabel("Evidence Found")
plt.ylabel("Monster")
plt.show()

# One-hot encoding (convert categorical variables to numerical)
one_hot_encoded = pd.get_dummies(merged[['allergy', 'favorite food', 'region', 'time of day', 'evidence found', 'crime weapon']])
merged = pd.concat([merged, one_hot_encoded], axis=1)
print(one_hot_encoded.head())
print(merged.columns)
merged = merged.drop(columns = ['index_crimes', 'date', 'region', 'crime type', 'crime weapon', 'time of day', 'evidence found', 'allergy', 'favorite food'])

# Feature and target variables
X = merged.drop(columns=['monster'])  # feature variables
y = merged['monster']  # target variable

# Split the datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Predict the monster involved in our crime
mayor_murder_data = {'region': 'village', 'time of day': 'day', 'evidence found': 'bones', 'crime weapon': 'knife'}
mayor_murder_df = pd.DataFrame(mayor_murder_data, index=[0])
mayor_murder_ohe = pd.get_dummies(mayor_murder_df[['region', 'time of day', 'evidence found', 'crime weapon']])
mayor_murder_df = pd.concat([mayor_murder_df, mayor_murder_ohe], axis=1)


mayor_murder_df = mayor_murder_df.drop(columns=['region', 'time of day', 'evidence found', 'crime weapon'])
print(mayor_murder_df.columns)
print(mayor_murder_df.head())

# Ensure mayor_murder_df has the same features as X
mayor_murder_df = mayor_murder_df.reindex(columns=X.columns, fill_value=0)

predicted_monster = rf_model.predict(mayor_murder_df)
print(f'\nPredicted monster: {predicted_monster[0]}')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")


# Example: Grid search for best parameters
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("\nBest parameters found:", grid_search.best_params_)