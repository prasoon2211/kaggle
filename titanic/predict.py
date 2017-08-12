import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

tune = True
np.random.seed(4313)

def fix_columns(d, columns):  
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0
    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[columns]
    return d

def predict_age(d, cols):
    d = d[cols]
    d = pd.get_dummies(d, columns=["Pclass", "Title", "Sex"])
    X_train, y_train = d[d.Age.notnull()].drop("Age", axis=1), d[d.Age.notnull()].Age
    clf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    X_test = d[d.Age.isnull()].drop("Age", axis=1)
    y = clf.predict(X_test)
    return y

def classify_age(row):
    if row.Age < 10 or row.Title == "Master":
        return "Child"
    if row.Age >=10 and row.Age < 16:
        return "Adolescent"
    if row.Age >=16 and row.Age < 50:
        return "Adult"
    if row.Age >= 50:
        return "Old"


def fill_age(df):
    cols = ['Pclass', "SibSp", "Parch", "Fare", "Title", "Sex", "Age", "Survived"]
    dftrain = df.iloc[:train_size, :]
    y1 = predict_age(dftrain, cols)

    cols.pop()
    dftest = df.iloc[train_size:, :]
    y2 = predict_age(dftest, cols)
    
    ages = np.concatenate([y1, y2])
    
    df.loc[df.Age.isnull(), "Age"] = ages
    

    df["AgeGroup"] = "Others"
    df["AgeGroup"] = df.apply(classify_age, axis=1)
    return df
    
def preproc(df):
    df["Title"] = df["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df["Title"][df["Title"].isin(['Mlle', 'Miss', 'Ms'])] = "Miss"
    df["Title"][df["Title"].isin(['Mme', 'Mrs'])] = "Mrs"
    df["Title"][df["Title"].isin(['Don', 'Dr', 'Sir', 'Rev'])] = "Sir"
    df["Title"][df["Title"].isin(['Capt', 'Major', 'Col'])] = "Capt"
    df["Title"][df["Title"].isin(['Dona', 'Lady', 'the Countess', 'Jonkheer'])] = "Lady"
    
    df["Family"] = df["SibSp"] + df["Parch"] + 1
    
    df["Cabin"] = df["Cabin"].fillna("X")
    df["Cabin"] = df["Cabin"].apply(lambda x: x[0])
    
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    df["Fare"] = df["Fare"].fillna(np.nanmedian(df["Fare"]))
    
    df["Pclass"] = df["Pclass"].astype('category')
    df = fill_age(df)
    cols = ['Pclass', "Family", "Fare", "Title", "Sex", "Age", "AgeGroup", "Survived"]
    return df[cols]

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['Survived'] = 0
comb = pd.concat([train, test])
train_size = train.shape[0]
comb = preproc(comb)
comb_ohe = pd.get_dummies(comb.drop("Survived", axis=1))
X_train, X_test = comb_ohe.iloc[:train_size, :], comb_ohe.iloc[train_size:, :]
y_train = train.Survived

clf1 = ExtraTreesClassifier(n_estimators=150, max_depth=10, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1)
clf3 = AdaBoostClassifier(n_estimators=100)
# clf4 = GaussianNB()
clf5 = XGBClassifier(n_estimators=150, max_depth=15)

clf = VotingClassifier(estimators=[('et', clf1), ('rf', clf2), ('abc', clf3), ('xgb', clf5)],
                                   voting='soft')
print("Score: ", cross_val_score(clf, X_train, y_train, cv=5))

clf.fit(X_train, y_train)

y_test = clf.predict(X_test)

test['Survived'] = y_test
result = test[["PassengerId", "Survived"]]
result.to_csv('predict-Python.csv', index=False)