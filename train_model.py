import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the data
print("loading data...")
df = pd.read_csv('titanic.csv')

# just pick the columns we need
features = ['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# cleaning up some stuff
# fill missing ages with median
df['Age'] = df['Age'].fillna(df['Age'].median())
# fill embarked with most common one
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# make gender and embarked numbers
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# split into X and y
X = df[features]
y = df[target]

# splitting for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
print("training model...")
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
clf.fit(X_train, y_train)

# see how it did
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Accuracy: {acc*100:.2f}%")
print("done!")
