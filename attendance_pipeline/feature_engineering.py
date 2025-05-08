import os
import pandas as pd
import numpy as np
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import BASE_DIR
from .logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_features(df: pd.DataFrame):
    logger.info("Starting feature engineering on %d rows", len(df))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').copy()

    df["dow"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["Date"].min(), end=df["Date"].max())
    df["is_holiday"] = df["Date"].isin(holidays).astype(int)
    logger.debug("Added calendar features (dow, month, is_weekend, is_holiday)")

    df = (
        df
        .groupby(["Place","AssignedRoom"], group_keys=False)
        .apply(lambda g: g.assign(
            lag_1        = g["TotalDurationHours"].shift(1),
            lag_7        = g["TotalDurationHours"].shift(7),
            lag_30       = g["TotalDurationHours"].shift(30),
            roll_mean_7  = g["TotalDurationHours"].shift(1).rolling(7).mean(),
            roll_std_7   = g["TotalDurationHours"].shift(1).rolling(7).std(),
            roll_mean_30 = g["TotalDurationHours"].shift(1).rolling(30).mean(),
            roll_std_30  = g["TotalDurationHours"].shift(1).rolling(30).std()
        ))
        .reset_index(drop=True)
    )
    logger.debug("Computed lag and rolling window features")

    df["level_mean_day"] = df.groupby(["Date","Level"])["TotalDurationHours"].transform("mean")
    df["level_sum_day"]  = df.groupby(["Date","Level"])["TotalDurationHours"].transform("sum")
    logger.debug("Added level_mean_day and level_sum_day features")

    df.fillna(0, inplace=True)
    logger.debug("Filled NaNs with 0")

    cat_df = df[["Place","Level","AssignedRoom"]].astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat = ohe.fit_transform(cat_df)
    logger.info("Fitted OneHotEncoder on categories; output shape %s", cat.shape)

    num_cols = [
        "dow","month","is_weekend","is_holiday",
        "lag_1","lag_7","lag_30",
        "roll_mean_7","roll_std_7","roll_mean_30","roll_std_30",
        "level_mean_day","level_sum_day"
    ]
    scaler = StandardScaler()
    num = scaler.fit_transform(df[num_cols])
    logger.info("Fitted StandardScaler on numeric cols; output shape %s", num.shape)

    X = np.hstack([cat, num])
    y = df["TotalDurationHours"].values
    logger.info("Built feature matrix X shape %s and target y length %d", X.shape, len(y))

    ohe_path    = os.path.join(MODELS_DIR, "ohe.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(ohe, ohe_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved OneHotEncoder to %s and StandardScaler to %s", ohe_path, scaler_path)

    return X, y
