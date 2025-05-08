#\CSI_Forecast\attendance_pipeline\main.py
import os
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

from .logger import configure_logging, get_logger
from .data_processing import parse_and_aggregate_attendance
from .feature_engineering import build_features
from .modeling import hyperparameter_search, evaluate_and_diagnose
from .forecasting import compute_typical_staffing, forecast_staff_by_group
from .visualization import plot_typical_staffing_table, plot_forecast_sequence, prompt_for_date
import attendance_pipeline.utils as utils

def main():
    parser = argparse.ArgumentParser(description="Run attendance forecasting pipeline")
    parser.add_argument(
        '--start-date',
        dest='start_date',
        help='Forecast start date (YYYY-MM-DD). If omitted, prompts via GUI or defaults to today.',
        type=str,
        required=False
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        dest='force_refresh',
        help='Bypass cached attendance output and re-parse all Excel files'
    )
    args = parser.parse_args()

    base_dir = utils.BASE_DIR
    log_path = os.path.join(base_dir, "logs", "pipeline.log")
    configure_logging(log_file=log_path)
    logger = get_logger(__name__)
    logger.info("//////Pipeline started")

    df = parse_and_aggregate_attendance()
    df = parse_and_aggregate_attendance(force_refresh=args.force_refresh)
    logger.info("Loaded attendance data: %d rows", len(df))

    if args.start_date:
        try:
            start = pd.to_datetime(args.start_date).normalize()
            logger.info("Using start date from CLI: %s", start)
        except Exception:
            logger.error("Invalid --start-date '%s'; expected YYYY-MM-DD. Exiting.", args.start_date)
            return
    else:
        try:
            start = prompt_for_date()
            logger.info("Using start date from GUI prompt: %s", start)
        except SystemExit:
            start = pd.to_datetime("today").normalize()
            logger.warning("No date entered; defaulting to %s", start)

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    logger.info("Features built: X=%s, y=%s", X.shape, y.shape)

    search_results = hyperparameter_search(X_train, y_train)
    logger.info("Hyperparameter search complete")

    best_rf  = search_results['rf'].best_estimator_
    best_xgb = search_results['xgb'].best_estimator_

    logger.info("Evaluating RandomForest")
    evaluate_and_diagnose(best_rf, X_test, y_test, model_name="RandomForest")

    logger.info("Evaluating XGBoost")
    evaluate_and_diagnose(best_xgb, X_test, y_test, model_name="XGBoost")

    typical = compute_typical_staffing(df)
    logger.info("Computed typical staffing pivot")
    fig1 = plot_typical_staffing_table(typical, start_date=start)
    fig1_path = os.path.join(base_dir, "plots", "typical_staffing.png")
    os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
    logger.info("Saved typical staffing plot to %s", fig1_path)

    detailed = forecast_staff_by_group(df, seq_len=7, horizon=7)
    fig2 = plot_forecast_sequence(
        df,
        start_date=start,
        seq_len=7,
        horizon=7
    )
    logger.info("Generated next-week staffing forecast: %d rows", len(detailed))
    fig2_path = os.path.join(base_dir, "plots", "forecast_sequence.png")
    logger.info("Saved forecast sequence plot to %s", fig2_path)

    logger.info("///////Pipeline finished at %s", datetime.now().isoformat())

if __name__ == "__main__":
    main()