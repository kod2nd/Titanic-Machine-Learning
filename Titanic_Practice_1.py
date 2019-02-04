# %%
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

# %%
# Read the Titanic_train.csv file here
titanicData = pd.read_csv('titanic_train.csv')
titanicData.head()


# %%
'''
For this example, we are only extracting 2 things: Class and sex. 

Do that below
'''
titanicData[['pclass', 'sex']]


# %%
# Extract the pclass and sex into a new Dataframe

features = titanicData[['pclass', 'sex']]
features.head()
# %%
# Convert pclass to pure numbers
features['pclass'].replace(to_replace="1st", value=1, inplace=True)
features['pclass'].replace(to_replace="2nd", value=2, inplace=True)
features['pclass'].replace(to_replace="3rd", value=3, inplace=True)

features.head()
# %%
# Replace the sex with 0 for female, 1 for male
features['sex'].replace(to_replace="female", value=0, inplace=True)
features['sex'].replace(to_replace="male", value=1, inplace=True)
features.head()
# %%
# Create the expected result dataframe.
expectedResult = titanicData[['survived']]
expectedResult.head()
# %%
# Create test/train split
X_train, X_test, y_train, y_test = train_test_split(
    features, expectedResult, test_size=0.33, random_state=50)

print(X_train.head())
print(y_train.head())

# %%
# Create the random forest instance, and train it with training data
randomForest = RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train, y_train)

# %%
# Get the accuracy of your model
accuracy = randomForest.score(X_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))


# %%
# Write the model to a file called "titanic_model2"
joblib.dump(randomForest, 'titanic_model2', compress=9)
# %%
