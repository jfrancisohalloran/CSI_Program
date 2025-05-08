#C:\Users\jfran\OneDrive\Documentos\GitHub\CSI_Forecast\attendance_pipeline\forecasting.py

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from .logger import get_logger
from .utils import BASE_DIR, get_level_from_room

logger = get_logger(__name__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def compute_typical_staffing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing typical staffing pivot from %d rows", len(df))
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['dow'] = df['Date'].dt.weekday
    df['dow_name'] = df['Date'].dt.day_name()

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
    biz = df[(df['dow'] < 5) & (~df['Date'].isin(holidays))].copy()

    biz['FTE'] = biz['TotalDurationHours'] / 9.0
    students_per_staff = {"Infant":4, "Multi-Age":4, "Toddler":6, "Preschool":10, "Pre-K":12}

    fte_pivot = (
        biz
        .groupby(['dow', 'dow_name', 'Place', 'Level'])['FTE']
        .mean()
        .reset_index()
        .pivot(
            index='dow_name',
            columns=['Place', 'Level'],
            values='FTE'
        )
        .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    )

    staff_df = fte_pivot.copy()
    for place, level in staff_df.columns:
        ratio = students_per_staff.get(level, 1)
        staff_df[(place, level)] = np.ceil(staff_df[(place, level)] / ratio).astype(int)

    logger.info("Typical staffing pivot built with shape %s", staff_df.shape)
    return staff_df

def forecast_staff_by_group(
    df: pd.DataFrame,
    seq_len: int = 7,
    *,
    start_date: str | pd.Timestamp | None = None,
    horizon: int = 7,
    lr: float = 1e-3,
    epochs: int = 20,
    batch: int = 16
) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    cal = USFederalHolidayCalendar()
    cbd = CustomBusinessDay(calendar=cal)
    holidays = cal.holidays(start=df.Date.min(), end=df.Date.max())
    mask_bd = (df.Date.dt.weekday < 5) & (~df.Date.isin(holidays))
    df_bd = df.loc[mask_bd]

    students_per_staff = {
        "Infant":4, "Multi-Age":4, "Toddler":6,
        "Preschool":10, "Pre-K":12
    }
    df_bd['StaffRequiredHist'] = (
        np.ceil((df_bd['TotalDurationHours'] / 9.0) /
                df_bd['Level'].map(students_per_staff).fillna(1))
        .astype(int)
    )

    if start_date is None:
        train_end = df_bd.Date.max()
    else:
        train_end = max(pd.to_datetime(start_date), df_bd.Date.min())
        if train_end not in df_bd.Date.values:
            train_end = pd.date_range(train_end, train_end + cbd, freq=cbd)[0]

    bd_index = pd.date_range(df_bd.Date.min(), train_end, freq=cbd)
    all_forecasts = []

    for (place, assigned_room), sub in df_bd.groupby(['Place','AssignedRoom']):
        sub_train = sub[sub.Date <= train_end]
        daily_hours = sub_train.groupby('Date')['TotalDurationHours'].sum()

        scaler = StandardScaler()
        scaler.fit(daily_hours.values.reshape(-1,1))

        ts_full = daily_hours.reindex(bd_index, fill_value=0)
        ts_scaled = scaler.transform(ts_full.values.reshape(-1,1)).ravel()

        if len(ts_scaled) < seq_len+1:
            continue

        X, y = [], []
        for i in range(len(ts_scaled)-seq_len):
            X.append(ts_scaled[i:i+seq_len])
            y.append(ts_scaled[i+seq_len])
        X = np.array(X).reshape(-1,seq_len,1)
        y = np.array(y)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len,1)),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='relu')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
        model.fit(X, y, epochs=epochs, batch_size=batch,
                  validation_split=0.1, verbose=0)

        level = get_level_from_room(assigned_room, place)
        fn = f"lstm_{place.replace(' ','_')}_{assigned_room.replace(' ','_')}.h5"
        model.save(os.path.join(MODELS_DIR, fn))
        logger.info("Saved LSTM model for %s – %s (Level=%s) → %s",
                    place, assigned_room, level, fn)

        window = ts_scaled[-seq_len:].copy()
        future_dates = pd.date_range(train_end + cbd, periods=horizon, freq=cbd)
        fc_scaled = []
        for _ in range(horizon):
            nxt = float(model.predict(window.reshape(1,seq_len,1)))
            fc_scaled.append(nxt)
            window = np.roll(window, -1)
            window[-1] = nxt

        fc_hours = scaler.inverse_transform(
            np.array(fc_scaled).reshape(-1,1)
        ).ravel()
        fc_hours = np.clip(fc_hours, 0, None)
        fc_students = fc_hours / 9.0
        ratio = students_per_staff.get(level,1)
        staff_fc = np.ceil(fc_students / ratio).astype(int)

        all_forecasts.append(pd.DataFrame({
            'Place': place,
            'AssignedRoom': assigned_room,
            'Level': level,
            'Date': future_dates,
            'ForecastHours': fc_hours,
            'ForecastStudents': fc_students,
            'StaffRequired': staff_fc
        }))

    if not all_forecasts:
        logger.warning("No groups had enough data to forecast.")
        return pd.DataFrame(columns=[
            'Place','AssignedRoom','Level','Date',
            'ForecastHours','ForecastStudents','StaffRequired'
        ])

    return pd.concat(all_forecasts, ignore_index=True)
