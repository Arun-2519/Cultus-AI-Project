# Cultus-AI-Project

README — Advanced Time Series Forecasting with Deep Learning (LSTM Optimization & Interpretability)

Project Type: Deep Learning, Time Series Forecasting, Model Optimization, Interpretability Author: Your Name Dataset: Programmatically Generated Multivariate Time Series (2000 observations) Frameworks: PyTorch, Statsmodels, Scikit-Learn, NumPy, Pandas

Project Overview
This project focuses on building a fully optimized and interpretable LSTM-based deep learning model for multivariate time series forecasting. The workflow covers:

Generating a complex synthetic economic-like dataset with trend, noise, seasonal cycles, autoregressive dependencies, and multiple exogenous variables.

Designing and training an advanced LSTM architecture that incorporates:

AdamW optimizer

Cosine annealing learning rate scheduler

Dropout scheduling (dynamic dropout increase during training)

Weight decay

Benchmarking against baseline forecasting models:

SARIMA (Seasonal ARIMA)

Feedforward Neural Network (FFNN) with flattened lag features

Conducting interpretability analysis using permutation-based feature importance for sequence models.

The final deliverables include dataset, code, metrics, and interpretability reports.

Dataset Description 2.1 Data Generation
The dataset is programmatically generated using:

Daily timestamps for 2000 days

Components such as:

Long-term trend

Yearly seasonal pattern

Weekly seasonal pattern

Autoregressive signals (AR(2))

Exogenous variables (exog1, exog2, exog3)

Gaussian noise

2.2 Features Included Feature Description target Main variable to be forecasted exog1 Seasonal exogenous driver (90-day cycle) exog2 Faster exogenous cycle (30-day cosine) exog3 Random noise-driven exogenous variable trend Linear upward trend seasonal_yearly Long periodic component seasonal_weekly Weekly periodic component lag_1 ... lag_14 Lagged historical observations

Minimum observations: 2000 rows Saved file: lstm_timeseries_dataset.csv

Problem Statement
To build a high-performance forecasting model capable of learning long-sequence temporal patterns while maintaining interpretability.

Key objectives:

Predict future values of a non-stationary, seasonal time series.

Compare deep learning with classical models.

Understand which features influence the forecast using explainable AI (XAI) methods.

Methodology & Approach 4.1 Data Preprocessing
Normalization using StandardScaler.

Creation of sequence windows with sequence length = 30.

Time-based train-validation-test split:

70% Train

10% Validation

20% Test

Model Architectures 5.1 LSTM Forecasting Model (Primary Model)
Key features:

2-layer LSTM architecture

Hidden size = 128

Dropout applied to:

LSTM layers

Fully-connected layers

Final prediction layer: Dense(64 → 1)

5.2 Advanced Optimization Techniques AdamW Optimizer

Decoupled weight decay

More stable and generalizable than Adam

Cosine Annealing LR Scheduler

Smoothly decreases learning rate across training epochs.

Dropout Scheduling

A custom function increases dropout probability as training progresses:

initial_dropout = 0.1 final_dropout = 0.5

This helps:

Prevent overfitting

Encourage more robust temporal pattern learning

Baseline Models 6.1 SARIMA
Captures seasonal components

Implemented using Statsmodels

Used as classical statistical benchmark

6.2 Feedforward Neural Network (FFNN)

Uses flattened lag features

Non-sequential model

Helps compare how sequence models outperform simple networks

Evaluation Metrics
Models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (%)

These metrics allow fair comparison between models with different assumptions.

Interpretability Analysis 8.1 Permutation Importance (Sequence-Specific)
Each feature is permuted across test sequences while maintaining temporal order.

Interpretation:

Increase in RMSE = feature importance

Higher RMSE change → More crucial feature

This helps identify:

Which exogenous variables matter

Whether lags or seasonality components dominate prediction

How the LSTM internally relies on temporal dependencies

Results Summary (Typical Observations)
Expected behavior:

LSTM performs significantly better than FFNN and SARIMA due to sequence modeling ability.

Exog variables and lag features show strong importance in permutation tests.

SARIMA may struggle due to nonlinearities and noise.

FFNN performs reasonably but cannot model long temporal patterns.

Computational Resources
Typical requirements:

CPU: 4–8 cores

RAM: 4–8 GB minimum

GPU (optional but recommended): RTX 1650 or better

Training times:

CPU-only: 10–20 minutes

GPU: 3–8 minutes

File Structure Project/ │ ├── Advanced_LSTM_Forecasting_Project.py # Full project code ├── lstm_timeseries_dataset.csv # Auto-generated dataset ├── README.md # Project documentation └── /outputs (optional)

How to Run the Project Step 1: Install dependencies pip install numpy pandas matplotlib scikit-learn torch statsmodels

Step 2: Run the script python Advanced_LSTM_Forecasting_Project.py

Step 3: Outputs

The script will display:

LSTM performance metrics

Baseline model metrics

Feature importances

Dataset saved to CSV

Conclusion
This project demonstrates a full end-to-end implementation of:

Time series data simulation

Deep learning sequence modeling

Model optimization

Statistical baselines

Modern interpretability techniques

It serves as a strong final-year, research, or professional portfolio project showcasing advanced machine learning and deep learning skills.# Advanced-Time-Series-Forecasting-with-Deep-Learning-LSTM-Optimization-and-Interpretability
