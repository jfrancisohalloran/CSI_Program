# C:\Users\jfran\OneDrive\Documentos\GitHub\CSI_Forecast\attendance_pipeline\modeling.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

import shap

from .utils import BASE_DIR
from .logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def create_dnn(u1=128, u2=64, lr=0.001):
    logger.debug("Building DNN: u1=%d, u2=%d, lr=%f", u1, u2, lr)
    model = Sequential([
        Dense(u1, activation="relu", input_dim=input_dim),
        Dropout(0.2),
        Dense(u2, activation="relu"),
        Dense(1, activation="relu")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mae")
    return model

def hyperparameter_search(X_train, y_train):
    logger.info("Starting hyperparameter search")
    logger.debug("X_train shape: %s, y_train length: %s", X_train.shape, len(y_train))
    for i in range(min(3, X_train.shape[1])):
        logger.debug("  feature[%d] first 5 values: %s", i, X_train[:5, i].tolist())

    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    logger.info("Tuning RandomForest with params %s", rf_params)
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(),
        rf_params,
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    ).fit(X_train, y_train)
    logger.info("RF best MAE=%.3f with %s",
                -rf_search.best_score_, rf_search.best_params_)
    rf_cv = pd.DataFrame(rf_search.cv_results_)
    top3_rf = rf_cv.nsmallest(3, 'mean_test_score')[['params','mean_test_score','std_test_score']]
    logger.debug("RF top 3 CV results:\n%s", top3_rf)
    results['rf'] = rf_search

    xgb_params = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9]
    }
    logger.info("Tuning XGB with params %s", xgb_params)
    xgb_search = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            predictor="cpu_predictor"
        ),
        xgb_params,
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    ).fit(X_train, y_train)
    logger.info("XGB best RMSE=%.3f with %s",
                np.sqrt(-xgb_search.best_score_), xgb_search.best_params_)
    xgb_cv = pd.DataFrame(xgb_search.cv_results_)
    top3_xgb = xgb_cv.nsmallest(3, 'mean_test_score')[['params','mean_test_score','std_test_score']]
    logger.debug("XGB top 3 CV results:\n%s", top3_xgb)
    results['xgb'] = xgb_search

    global input_dim
    input_dim = X_train.shape[1]
    keras_reg = KerasRegressor(
        build_fn=create_dnn,
        epochs=20,
        batch_size=16,
        verbose=0
    )
    dnn_params = {"u1": [64, 128], "u2": [32, 64], "lr": [0.001, 0.01]}
    logger.info("Tuning DNN with params %s", dnn_params)
    dnn_search = RandomizedSearchCV(
        keras_reg,
        dnn_params,
        n_iter=5,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    try:
        dnn_search.fit(X_train, y_train)
        logger.info("DNN best MAE=%.3f with %s",
                    -dnn_search.best_score_, dnn_search.best_params_)
        dnn_cv = pd.DataFrame(dnn_search.cv_results_)
        top3_dnn = dnn_cv.nsmallest(3, 'mean_test_score')[['params','mean_test_score','std_test_score']]
        logger.debug("DNN top 3 CV results:\n%s", top3_dnn)
        results['dnn'] = dnn_search
    except Exception as e:
        logger.warning("Skipping DNN search due to error: %s", e)
        results['dnn'] = None

    return results


def rolling_errors(model, X, y, window=5) -> pd.Series:
    preds = model.predict(X)
    errors = np.abs(y - preds)
    rolling = pd.Series(errors).rolling(window).mean()
    logger.debug("Computed rolling MAE with window=%d", window)
    return rolling


def explain_tree(model, X):
    logger.info("Running SHAP explainer")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    logger.info("SHAP plot generated")


def evaluate_and_diagnose(model, X_test, y_test, model_name="Model"):
    logger.info("Evaluating %s", model_name)
    preds = model.predict(X_test).ravel()
    errors = y_test - preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info("%s MAE=%.3f, RMSE=%.3f", model_name, mae, rmse)
    for i in range(min(5, len(y_test))):
        logger.debug("%s sample %d: pred=%.2f  actual=%.2f", model_name, i, preds[i], y_test[i])

    rolling = pd.Series(np.abs(errors)).rolling(7, min_periods=1).mean()
    logger.info("%s Rolling MAE: last=%.3f, mean=%.3f, max=%.3f",
                model_name, rolling.iloc[-1], rolling.mean(), rolling.max())

    rolling_path = os.path.join(LOGS_DIR, f"{model_name}_rolling_mae.csv")
    rolling.to_csv(rolling_path, index=False)
    logger.info("Saved rolling MAE to %s", rolling_path)

    from statsmodels.tsa.stattools import acf
    acf_vals = acf(errors, nlags=10)
    logger.info("%s ACF[0..10]=%s", model_name, acf_vals.tolist())
