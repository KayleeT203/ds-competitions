#Import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from xgboost import XGBClassifier

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

re_train_df['Age'] = re_train_df['Age'].fillna(re_train_df['Age'].median())

#re_train_df['IsElderly'] = (re_train_df['Age'] > 60).astype(int)

#target (what to predict)
y = re_train_df['Survived']
#y = train_df['Survived']

#freatures 
features = [
    "Pclass", "Age", "SibSp", "Parch", "FarePerPerson", 
    "Sex_num", "Has_cabin", "IsChild"
]

X = re_train_df[features]
#X = train_df[features]

xgb_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])
scores = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


#print("MAE:", -scores.mean())
print("Accuracy:", scores.mean())

# Train final model on full training data
xgb_pipeline.fit(X, y)

# Prepare test features
test_xgb = pd.read_csv('/kaggle/input/titanic/test.csv')

# Apply same feature engineering to test data
test_xgb['Sex_num'] = test_xgb['Sex'].map({'female': 1, 'male': 0})
test_xgb['Has_cabin'] = test_xgb['Cabin'].notna().astype(int)
test_xgb['FamilySize'] = test_xgb['SibSp'] + test_xgb['Parch'] + 1
test_xgb['FarePerPerson'] = test_xgb['Fare'] / test_xgb['FamilySize']
test_xgb['IsChild'] = (test_xgb['Age'] < 12).astype(int)
#test_xgb['IsElderly'] = (test_xgb['Age'] > 60).astype(int)

X_test = test_xgb[features]

# Predict
test_preds = xgb_pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_xgb["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)
