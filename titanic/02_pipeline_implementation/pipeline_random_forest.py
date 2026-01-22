#Import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#Load data
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

re_train_df = train_df.copy()

#feature engineering
re_train_df['Sex_num'] = re_train_df['Sex'].map({'female': 1, 'male': 0})

re_train_df['Has_cabin'] = re_train_df['Cabin'].notna().astype(int)

re_train_df['FamilySize'] = (
    re_train_df['SibSp'] + re_train_df['Parch'] + 1
)

re_train_df['FarePerPerson'] = (
    re_train_df['Fare'] / re_train_df['FamilySize']
)

re_train_df['IsChild'] = (re_train_df['Age'] < 12).astype(int)

#re_train_df['IsElderly'] = (re_train_df['Age'] > 60).astype(int)

#target (what to predict)
y = re_train_df['Survived']
#y = train_df['Survived']

#freatures 
features = [
    "Pclass", "Age", "SibSp", "Parch", 
    "FarePerPerson", "Sex_num", "Has_cabin", "IsChild"
]


X = re_train_df[features]
#X = train_df[features]

#pipeline for better process
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        #n_estimators=274,
        #random_state=23
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=23
    ))
    #("model", DecisionTreeClassifier(
     #    max_leaf_nodes = 200,
      #   random_state=23
    #))
    
])

"""scores = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)
"""
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


#print("MAE:", -scores.mean())
print("Accuracy:", scores.mean())


# Train final model on full training data
pipeline.fit(X, y)

# Prepare test features
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# Apply same feature engineering to test data
test_df['Sex_num'] = test_df['Sex'].map({'female': 1, 'male': 0})
test_df['Has_cabin'] = test_df['Cabin'].notna().astype(int)
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['FarePerPerson'] = test_df['Fare'] / test_df['FamilySize']
test_df['IsChild'] = (test_df['Age'] < 12).astype(int)
test_df['IsElderly'] = (test_df['Age'] > 60).astype(int)

X_test = test_df[features]

# Predict
test_preds = pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_preds.astype(int)
})

submission.to_csv("submission.csv", index=False)
