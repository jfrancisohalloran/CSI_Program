from setuptools import setup, find_packages

setup(
    name="attendance_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.3",
        "numpy>=2.1.3",
        "scikit-learn>=1.6.1",
        "xgboost>=3.0.0",
        "tensorflow>=2.19.0",
        "scikeras>=0.13.0",
        "shap>=0.47.2",
        "statsmodels>=0.14.4",
        "plotly>=6.0.1",
        "matplotlib>=3.10.1",
        "joblib>=1.4.2"            
    ],
    entry_points={
        "console_scripts": [
            "attendance-pipeline=attendance_pipeline.main:main"
        ]
    },
    author="UNO BSAD 8310",
    description="An end-to-end attendance forecasting pipeline with interactive HTML visualizations.",
    license="MIT",
)
