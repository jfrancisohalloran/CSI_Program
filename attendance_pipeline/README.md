attendance_pipeline/
├── utils.py – Utility functions for parsing and mapping raw attendance events
│   ├─ extract_year_from_filename(filename) → Extract a 4-digit year from a filename
│   ├─ parse_event(cell_text) → Parse time/teacher/room events from a cell
│   ├─ combine_date_time(date_obj, time_str) → Combine a date and time string into a pandas Timestamp
│   └─ get_level_from_room(assigned_room, place) → Map an assigned room name to a staffing level
│
├── logger.py – Configure and retrieve module loggers
│   ├─ configure_logging(log_file: str = None) → Set up logging to console and optional file
│   └─ get_logger(name: str) → Obtain a named logger for use in modules
│
├── visualization.py – Render staffing tables and interactive forecasts
│   ├─ plot_typical_staffing_table(typical_df: pd.DataFrame, start_date: pd.Timestamp|None) → Generate a business‑day table of average required staff per level
│   └─ plot_forecast_sequence(df: pd.DataFrame, start_date: pd.Timestamp, horizon: int) → Produce an interactive table and line chart of forecasted staff requirements
│
├── data_processing.py – Load and aggregate raw Excel attendance logs
│   └─ parse_and_aggregate_attendance() → Read sign‑in/out sheets, compute daily durations, FTE students, and required staff per room
│
├── feature_engineering.py – Create model input features and target arrays
│   └─ build_features(df: pd.DataFrame) → Add calendar, lag/rolling, level aggregate, one‑hot, and scaled numeric features for modeling
│
├── modeling.py – Machine learning training, tuning, and diagnostics
│   ├─ create_dnn(u1=128, u2=64, lr=0.001) → Build a Keras sequential neural network for regression tasks
│   ├─ hyperparameter_search(X_train, y_train) → Perform time‑series cross‑validated tuning for RF, XGB, and DNN models
│   ├─ evaluate_and_diagnose(model, X_test, y_test, model_name) → Calculate MAE/RMSE, rolling MAE, and save diagnostics
│   ├─ rolling_errors(model, X, y, window=5) → Compute rolling MAE series for error analysis
│   └─ explain_tree(model, X) → Generate a SHAP summary plot for tree‑based models
│
├── forecasting.py – Compute typical staffing and generate LSTM forecasts
│   ├─ compute_typical_staffing(df: pd.DataFrame) → Pivot average staff required per weekday and level from historical data
│   └─ forecast_staff_by_group(df: pd.DataFrame, seq_len=7, start_date: Optional[pd.Timestamp], horizon=7) → Train per‑group LSTM models on past hours and forecast future staff requirements
│
└── main.py – Orchestrate the full attendance-to‑forecast pipeline
    └─ main() → Execute data parsing, feature building, modeling, forecasting, and visualization end‑to‑end

