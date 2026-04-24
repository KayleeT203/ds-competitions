import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_high_correlation(df) -> pd.Series:
    corr = df.corr(numeric_only=True)
    corr_pairs = corr.unstack()
    high_corr = corr_pairs[(corr_pairs > 0.75) & (corr_pairs < 1.0)]
    # Remove mirrored duplicates (A,B) and (B,A)
    high_corr = high_corr[high_corr.index.get_level_values(0) < high_corr.index.get_level_values(1)]
    return high_corr.sort_values(ascending=False)

def get_low_correlation(df) -> pd.Series:
    corr = df.corr(numeric_only=True)
    corr_pairs = corr.unstack()
    low_corr = corr_pairs[corr_pairs < -0.75]
    # Remove mirrored duplicates (A,B) and (B,A)
    low_corr = low_corr[low_corr.index.get_level_values(0) < low_corr.index.get_level_values(1)]
    return low_corr.sort_values()

def cap_outliers(df):
    df_capped = df.copy()
    df_numeric = df_capped.select_dtypes(include='number')
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_capped[df_numeric.columns] = df_numeric.clip(lower=lower, upper=upper, axis=1)
    return df_capped

def drop_null_rows(df, threshold=0.5):
    df_cleaned = df.copy()
    thresh = len(df_cleaned.columns) * threshold
    df_cleaned = df_cleaned.dropna(thresh=thresh)
    return df_cleaned

def fill_nulls(df):
    df_filled = df.copy()
    df_numeric = df_filled.select_dtypes(include='number')

    df_filled[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())
    return df_filled

# Utilise the high and low correlation indexes
def add_corr_rank_feature(df):
    df = df.copy()

    high_corr = get_high_correlation(df)
    low_corr = get_low_correlation(df)

    #Use correlation value to score them
    high_scores = {}
    for (col1, col2), corr_value in high_corr.items():
        high_scores[col1] = high_scores.get(col1, 0) + corr_value
        high_scores[col2] = high_scores.get(col2, 0) + corr_value

    low_scores = {}
    for (col1, col2), corr_value in low_corr.items():
        low_scores[col1] = low_scores.get(col1, 0) + corr_value
        low_scores[col2] = low_scores.get(col2, 0) + corr_value

    df["high_corr_rank_score"] = sum(
        df[col] * score for col, score in high_scores.items() if col in df.columns
    )
    df["low_corr_rank_score"] = sum(
        df[col] * score for col, score in low_scores.items() if col in df.columns
    )
    return df

def add_mosquito_risk(df):
    df = df.copy()

    scaler = MinMaxScaler()

    df['precip_norm'] = scaler.fit_transform(df[['precipitation_amt_mm']])
    df['humidity_norm'] = scaler.fit_transform(df[['reanalysis_relative_humidity_percent']])
    df['temp_norm'] = scaler.fit_transform(df[['station_avg_temp_c']])

    df['mosquito_risk_score'] = (
        df['precip_norm'] +
        df['humidity_norm'] +
        df['temp_norm']
    ) / 3

    # Rolling mean for 2-6 weeks ago (mosquito lifecycle)
    # 卵から成虫＋潜伏期間考慮
    df['dengue_risk_2_6wk'] = (
        df['mosquito_risk_score']
        .shift(2)   # start from 2 weeks ago
        .rolling(window=4)  # average over 4 weeks (2-6 weeks ago)
        .mean()
    )

    # Different windows just incase
    df['dengue_risk_3_5wk'] = (
        df['mosquito_risk_score']
        .shift(3)   # start from 3 weeks ago
        .rolling(window=2)  # average over 2 weeks (3-5 weeks ago)
        .mean()
    )

    df['dengue_risk_4_6wk'] = (
        df['mosquito_risk_score']
        .shift(4)   # start from 4 weeks ago
        .rolling(window=2)  # average over 2 weeks (4-6 weeks ago)
        .mean()
    )

    df = df.ffill()

    return df

def apply_all(df, is_train=False):
    df = df.copy()

    # Capped outliers
    if is_train:
        df = cap_outliers(df)

    # ndvi empty issue
    df[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']] = (
        df[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']].ffill())

    # Fill station temps using satellite equivalent (more reliable)
    df['station_avg_temp_c'] = df['station_avg_temp_c'].fillna(
        df['reanalysis_avg_temp_k'] - 273.15)  # Kelvin to Celsius)

    df['station_diur_temp_rng_c'] = df['station_diur_temp_rng_c'].fillna(
        df['reanalysis_tdtr_k'])  # tdtr = diurnal temp range)

    df['station_max_temp_c'] = df['station_max_temp_c'].fillna(
        df['reanalysis_max_air_temp_k'] - 273.15)

    df['station_min_temp_c'] = df['station_min_temp_c'].fillna(
        df['reanalysis_min_air_temp_k'] - 273.15)

    # Rainfall is sequential
    df['precipitation_amt_mm'] = df['precipitation_amt_mm'].ffill()
    df['reanalysis_sat_precip_amt_mm'] = df['reanalysis_sat_precip_amt_mm'].ffill()

    # Drop rows with too much null columns
    if is_train:
        df = drop_null_rows(df)

    # Fill remaining with ffill as last resort
    df = df.ffill()

    # Feature engineering

    # extract the month (first int before /)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['month'] = df['week_start_date'].dt.month
    # wet season
    df["wet_season"] = ((df['month'] >= 5) & (df['month'] <= 11)).astype(int)
    # spike window between 30 to 50 week
    df["spike_window"] = ((df['weekofyear'] >= 30) & (df['weekofyear'] <= 50)).astype(int)

    # Mosquito risk features
    #df = add_mosquito_risk_features(df)

    return df

def add_mosquito_risk_features(df, scaler=None, use_rolling=True):
    df = df.copy()

    cols = ['precipitation_amt_mm',
            'reanalysis_relative_humidity_percent',
            'station_avg_temp_c']

    if scaler is None:
        # Training - fit a new scaler
        scaler = MinMaxScaler()
        df[['precip_norm', 'humidity_norm', 'temp_norm']] = scaler.fit_transform(df[cols])
    else:
        # Test - use existing scaler fitted on train
        df[['precip_norm', 'humidity_norm', 'temp_norm']] = scaler.transform(df[cols])

    df['mosquito_risk_score'] = (
        df['precip_norm'] +
        df['humidity_norm'] +
        df['temp_norm']
    ) / 3

    df['dengue_risk_2_6wk'] = (df['mosquito_risk_score'].shift(2).rolling(window=4).mean())
    df['dengue_risk_3_5wk'] = (df['mosquito_risk_score'].shift(3).rolling(window=2).mean())
    df['dengue_risk_4_6wk'] = (df['mosquito_risk_score'].shift(4).rolling(window=2).mean())

    if use_rolling:
        df['temp_rolling_4wk'] = df['temp_norm'].shift(1).rolling(4).mean()
        df['temp_rolling_7wk'] = df['temp_norm'].shift(1).rolling(7).mean()

        df['humidity_rolling_4wk'] = df['humidity_norm'].shift(1).rolling(4).mean()
        df['humidity_rolling_7wk'] = df['humidity_norm'].shift(1).rolling(7).mean()

        df['precip_rolling_4k'] = df['precip_norm'].shift(1).rolling(4).mean()
        df['precip_rolling_7k'] = df['precip_norm'].shift(1).rolling(7).mean()

    df = df.ffill()
    return df, scaler
