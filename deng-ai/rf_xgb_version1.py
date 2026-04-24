# Import
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from utils import get_high_correlation, get_low_correlation, cap_outliers, drop_null_rows, add_corr_rank_feature, apply_all, add_mosquito_risk_features

# Load data
train_features_df = pd.read_csv('../../dengue_features_train.csv')
train_label_df = pd.read_csv('../../dengue_labels_train.csv')
train_df = pd.merge(
    train_features_df.copy(),
    train_label_df.copy(),
    on=['city', 'year', 'weekofyear']
)

# test
test_features_df = pd.read_csv('../../dengue_features_test.csv')

test_df_sj = test_features_df[test_features_df['city'] == 'sj']
test_df_iq = test_features_df[test_features_df['city'] == 'iq']

#print(train_df['city'].unique())
# Split df into two (sj and iq)
train_df_sj = train_df[train_df['city'] == 'sj']
train_df_iq = train_df[train_df['city'] == 'iq']

train_df_sj_cleaned = apply_all(train_df_sj, is_train=True)
train_df_iq_cleaned = apply_all(train_df_iq, is_train=True)

train_df_sj_cleaned, scaler_sj = add_mosquito_risk_features(train_df_sj_cleaned, use_rolling=True)
train_df_iq_cleaned, scaler_iq = add_mosquito_risk_features(train_df_iq_cleaned, use_rolling=False)

# Features
# Columns to exclude
exclude_cols = ['city', 'year', 'weekofyear', 'week_start_date', 'total_cases']

# Automatically grab all remaining columns as features
features_sj = [col for col in train_df_sj_cleaned.columns if col not in exclude_cols]
features_iq = [col for col in train_df_iq_cleaned.columns if col not in exclude_cols]

X_sj = train_df_sj_cleaned[features_sj]
y_sj = train_df_sj_cleaned["total_cases"]

X_iq = train_df_iq_cleaned[features_iq]
y_iq = train_df_iq_cleaned["total_cases"]

xgb_pipeline_sj = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        n_estimators = 1000,
        learning_rate =0.005,
        max_depth=3,
        # 5
        min_child_weight=5,
        # 1
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])
xgb_pipeline_iq = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBRegressor(
        n_estimators = 100,
        learning_rate =0.01,
        max_depth=3,
        #3
        #0.05
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

rf_pipeline_sj = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        #n_estimators=274,
        #random_state=23
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=5,
        random_state=42
    ))
])

rf_pipeline_iq = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        #n_estimators=274,
        #random_state=23
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        min_samples_split=5,
        random_state=42
    ))
])

tscv = TimeSeriesSplit(n_splits=5)
scores_sj_xgb = cross_val_score(
    xgb_pipeline_sj,
    X_sj,
    y_sj,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)
scores_iq_xgb = cross_val_score(
    xgb_pipeline_iq,
    X_iq,
    y_iq,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)

scores_sj_rf = cross_val_score(
    rf_pipeline_sj,
    X_sj,
    y_sj,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)
scores_iq_rf = cross_val_score(
    rf_pipeline_iq,
    X_iq,
    y_iq,
    cv=tscv,
    scoring="neg_mean_absolute_error"
)


print(f"XGB SJ MAE: {-scores_sj_xgb.mean():.2f}")
print(f"XGB IQ MAE: {-scores_iq_xgb.mean():.2f}")
print(f"RF SJ MAE: {-scores_sj_rf.mean():.2f}")
print(f"RF IQ MAE: {-scores_iq_rf.mean():.2f}")

test_df_sj_cleaned = apply_all(test_df_sj, is_train=False)
test_df_iq_cleaned = apply_all(test_df_iq, is_train=False)

test_df_sj_cleaned, _ = add_mosquito_risk_features(test_df_sj_cleaned, scaler=scaler_sj, use_rolling=True)
test_df_iq_cleaned, _ = add_mosquito_risk_features(test_df_iq_cleaned, scaler=scaler_iq, use_rolling=False)

X_test_sj = test_df_sj_cleaned[features_sj]
X_test_iq = test_df_iq_cleaned[features_iq]

xgb_pipeline_sj.fit(X_sj, y_sj)
xgb_pipeline_iq.fit(X_iq, y_iq)
rf_pipeline_sj.fit(X_sj, y_sj)
rf_pipeline_iq.fit(X_iq, y_iq)

preds_sj_xgb = xgb_pipeline_sj.predict(X_test_sj).astype(int)
preds_iq_xgb = xgb_pipeline_iq.predict(X_test_iq).astype(int)
preds_sj_rf = rf_pipeline_sj.predict(X_test_sj).astype(int)
preds_iq_rf = rf_pipeline_iq.predict(X_test_iq).astype(int)

final_preds_sj = (preds_sj_xgb * 0.3 + preds_sj_rf * 0.7).astype(int)
final_preds_iq = (preds_iq_xgb * 0.4 + preds_iq_rf * 0.6).astype(int)

submission = pd.read_csv('../../submission_format.csv')
submission.loc[submission['city'] == 'sj', 'total_cases'] = final_preds_sj
submission.loc[submission['city'] == 'iq', 'total_cases'] = final_preds_iq

print(f"Shape: {submission.shape}")
print(f"Nulls: {submission.isnull().sum().sum()}")
print(f"Negatives: {(submission['total_cases'] < 0).sum()}")

submission.to_csv('submission_xgb3_rf7.csv', index=False)
print("Done!")