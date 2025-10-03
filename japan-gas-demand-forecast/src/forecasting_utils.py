"""
Utility functions for Japan Gas Demand Forecasting Project
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


def calculate_forecast_metrics(actual, predicted, model_name="Model"):
    """
    Calculate comprehensive forecast accuracy metrics
    
    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like  
        Predicted/forecasted values
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    # Ensure arrays are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {"error": "No valid data points for evaluation"}
    
    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    # Additional metrics
    mean_actual = np.mean(actual)
    normalized_mae = mae / mean_actual * 100
    normalized_rmse = rmse / mean_actual * 100
    
    # Directional accuracy (for trend prediction)
    actual_diff = np.diff(actual)
    pred_diff = np.diff(predicted)
    directional_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
    
    # R-squared
    r2 = r2_score(actual, predicted)
    
    return {
        'model_name': model_name,
        'mae': mae,
        'mse': mse, 
        'rmse': rmse,
        'mape': mape,
        'normalized_mae': normalized_mae,
        'normalized_rmse': normalized_rmse,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'n_observations': len(actual)
    }


def time_series_cv_split(data, n_splits=5, test_size=12):
    """
    Create time series cross-validation splits
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with datetime index
    n_splits : int
        Number of CV splits
    test_size : int
        Number of periods in test set
        
    Returns:
    --------
    list : List of (train_idx, test_idx) tuples
    """
    splits = []
    n_obs = len(data)
    
    # Calculate split positions
    for i in range(n_splits):
        test_end = n_obs - i * test_size
        test_start = test_end - test_size
        train_end = test_start
        
        if train_end < test_size:  # Ensure minimum training size
            break
            
        train_idx = data.index[:train_end]
        test_idx = data.index[test_start:test_end]
        
        splits.append((train_idx, test_idx))
    
    return list(reversed(splits))  # Return chronological order


def create_lagged_features(data, target_col, lags=[1, 2, 3, 12]):
    """
    Create lagged features for machine learning models
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    target_col : str
        Name of target column
    lags : list
        List of lag periods to create
        
    Returns:
    --------
    pandas.DataFrame : Data with lagged features
    """
    df = data.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Add rolling averages
    for window in [3, 6, 12]:
        df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
    
    # Add seasonal indicators
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    return df


def generate_forecast_intervals(predictions, residuals, confidence_level=0.95):
    """
    Generate prediction intervals using residual-based method
    
    Parameters:
    -----------
    predictions : array-like
        Point forecasts
    residuals : array-like
        Historical residuals from model
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% intervals)
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    # Calculate residual standard deviation
    residual_std = np.std(residuals)
    
    # Calculate critical value for normal distribution
    alpha = 1 - confidence_level
    from scipy.stats import norm
    z_critical = norm.ppf(1 - alpha/2)
    
    # Calculate intervals
    margin_error = z_critical * residual_std
    lower_bound = predictions - margin_error
    upper_bound = predictions + margin_error
    
    return lower_bound, upper_bound


class ForecastEvaluator:
    """
    Class for comprehensive forecast evaluation and comparison
    """
    
    def __init__(self):
        self.results = []
    
    def add_model_results(self, actual, predicted, model_name, additional_info=None):
        """Add model results for evaluation"""
        metrics = calculate_forecast_metrics(actual, predicted, model_name)
        if additional_info:
            metrics.update(additional_info)
        self.results.append(metrics)
    
    def get_results_summary(self):
        """Get summary table of all model results"""
        if not self.results:
            return "No results to display"
        
        df = pd.DataFrame(self.results)
        df = df.round(3)
        return df.sort_values('mape')  # Sort by MAPE (lower is better)
    
    def get_best_model(self, metric='mape'):
        """Get the best performing model based on specified metric"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        # For metrics where lower is better
        if metric in ['mae', 'mse', 'rmse', 'mape']:
            best_idx = df[metric].idxmin()
        # For metrics where higher is better  
        elif metric in ['r2', 'directional_accuracy']:
            best_idx = df[metric].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return df.iloc[best_idx].to_dict()


def plot_forecast_results(actual, predicted, dates, model_name, forecast_horizon=None):
    """
    Plot actual vs predicted values with forecast horizon
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    dates : array-like
        Corresponding dates
    model_name : str
        Name of the model
    forecast_horizon : int
        Number of periods that are forecasts (vs fitted values)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(dates, actual, label='Actual', color='blue', linewidth=2)
    
    # Split into fitted and forecasted if horizon specified
    if forecast_horizon:
        fitted_dates = dates[:-forecast_horizon]
        fitted_pred = predicted[:-forecast_horizon]
        forecast_dates = dates[-forecast_horizon:]
        forecast_pred = predicted[-forecast_horizon:]
        
        plt.plot(fitted_dates, fitted_pred, label='Fitted', color='green', linewidth=1.5)
        plt.plot(forecast_dates, forecast_pred, label='Forecast', color='red', 
                linewidth=2, linestyle='--')
    else:
        plt.plot(dates, predicted, label='Predicted', color='red', linewidth=1.5)
    
    plt.title(f'{model_name} - Forecast Results')
    plt.xlabel('Date')
    plt.ylabel('Gas Demand (Million m³)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Data validation utilities
def check_stationarity(series, significance_level=0.05):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(series.dropna())
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    is_stationary = p_value < significance_level
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary,
        'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
    }


def seasonal_strength(series, period=12):
    """
    Calculate seasonal strength measure
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomp = seasonal_decompose(series, model='additive', period=period)
    
    # Calculate seasonal strength as ratio of seasonal variance to total variance
    seasonal_var = np.var(decomp.seasonal.dropna())
    total_var = np.var(series.dropna())
    
    seasonal_strength = seasonal_var / total_var
    
    return seasonal_strength