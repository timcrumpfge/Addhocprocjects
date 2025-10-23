#!/usr/bin/env python3
"""
Gas Import Forecasting Analysis
This script analyzes gas import data from multiple countries and implements various forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Import forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    print("✓ Statsmodels imported successfully")
except ImportError as e:
    print(f"✗ Statsmodels import error: {e}")
    print("Please install: pip install statsmodels")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Scikit-learn import error: {e}")
    print("Please install: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import error: {e}")
    print("Please install: pip install tensorflow")

print("\n" + "="*60)
print("GAS IMPORT FORECASTING ANALYSIS")
print("="*60)

# Create sample data based on the image description
print("\n1. Loading Data...")
data = {
    'Year': [2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
             2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
             2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020,
             2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021,
             2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022,
             2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
             2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024,
             2025, 2025, 2025, 2025, 2025, 2025, 2025],
    'Month': ['Jan-18', 'Feb-18', 'Mar-18', 'Apr-18', 'May-18', 'Jun-18', 
              'Jul-18', 'Aug-18', 'Sep-18', 'Oct-18', 'Nov-18', 'Dec-18',
              'Jan-19', 'Feb-19', 'Mar-19', 'Apr-19', 'May-19', 'Jun-19',
              'Jul-19', 'Aug-19', 'Sep-19', 'Oct-19', 'Nov-19', 'Dec-19',
              'Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20',
              'Jul-20', 'Aug-20', 'Sep-20', 'Oct-20', 'Nov-20', 'Dec-20',
              'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21',
              'Jul-21', 'Aug-21', 'Sep-21', 'Oct-21', 'Nov-21', 'Dec-21',
              'Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22',
              'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22',
              'Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'May-23', 'Jun-23',
              'Jul-23', 'Aug-23', 'Sep-23', 'Oct-23', 'Nov-23', 'Dec-23',
              'Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24',
              'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
              'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25', 'Jul-25'],
    'Algeria': [4239, 4000, 4500, 4200, 4100, 4300, 4400, 4200, 4000, 4100, 4200, 4300,
                4100, 3900, 4200, 4000, 3900, 4100, 4200, 4000, 3800, 3900, 4000, 4100,
                1071, 1200, 1300, 1071, 1100, 1200, 1300, 1200, 1100, 1200, 1300, 1200,
                1400, 1300, 1500, 1400, 1300, 1400, 1500, 1400, 1300, 1400, 1500, 1400,
                1600, 1500, 1700, 1600, 1500, 1600, 1700, 1600, 1500, 1600, 1700, 1600,
                1800, 1700, 1900, 1800, 1700, 1800, 1900, 1800, 1700, 1800, 1900, 1800,
                2000, 1900, 2100, 2000, 1900, 2000, 2100, 2000, 1900, 2000, 2100, 2000,
                2200, 2100, 2300, 2200, 2100, 2200, 2759],
    'Libya': [358, 300, 350, 320, 310, 330, 340, 320, 300, 310, 320, 330,
              300, 280, 320, 300, 280, 300, 320, 300, 280, 290, 300, 310,
              635, 600, 650, 635, 600, 650, 700, 650, 600, 650, 700, 650,
              500, 450, 550, 500, 450, 500, 550, 500, 450, 500, 550, 500,
              400, 350, 450, 400, 350, 400, 450, 400, 350, 400, 450, 400,
              300, 250, 350, 300, 250, 300, 350, 300, 250, 300, 350, 300,
              200, 150, 250, 200, 150, 200, 250, 200, 150, 200, 250, 200,
              100, 50, 150, 100, 50, 100, 49],
    'Iran': [952, 900, 950, 920, 910, 930, 940, 920, 900, 910, 920, 930,
             900, 880, 920, 900, 880, 900, 920, 900, 880, 890, 900, 910,
             800, 750, 850, '-', '-', '-', 900, 850, 800, 850, 900, 850,
             700, 650, 750, 700, 650, 700, 750, 700, 650, 700, 750, 700,
             600, 550, 650, 600, 550, 600, 650, 600, 550, 600, 650, 600,
             500, 450, 550, 500, 450, 500, 550, 500, 450, 500, 550, 500,
             400, 350, 450, 400, 350, 400, 450, 400, 350, 400, 450, 400,
             300, 250, 350, 300, 250, 300, 973],
    'Azerbaijan': [733, 700, 750, 720, 710, 730, 740, 720, 700, 710, 720, 730,
                  700, 680, 720, 700, 680, 700, 720, 700, 680, 690, 700, 710,
                  1083, 1000, 1100, 1083, 1000, 1100, 1200, 1100, 1000, 1100, 1200, 1100,
                  1300, 1200, 1400, 1300, 1200, 1300, 1400, 1300, 1200, 1300, 1400, 1300,
                  1500, 1400, 1600, 1500, 1400, 1500, 1600, 1500, 1400, 1500, 1600, 1500,
                  1700, 1600, 1800, 1700, 1600, 1700, 1800, 1700, 1600, 1700, 1800, 1700,
                  1900, 1800, 2000, 1900, 1800, 1900, 2000, 1900, 1800, 1900, 2000, 1900,
                  2100, 2000, 2200, 2100, 2000, 2100, 2259],
    'Norway': [12729, 12000, 13000, 12500, 12400, 12600, 12800, 12600, 12400, 12500, 12600, 12700,
               12400, 12200, 12600, 12400, 12200, 12400, 12600, 12400, 12200, 12300, 12400, 12500,
               10454, 10000, 11000, 10454, 10000, 11000, 12000, 11000, 10000, 11000, 12000, 11000,
               13000, 12000, 14000, 13000, 12000, 13000, 14000, 13000, 12000, 13000, 14000, 13000,
               15000, 14000, 16000, 15000, 14000, 15000, 16000, 15000, 14000, 15000, 16000, 15000,
               17000, 16000, 18000, 17000, 16000, 17000, 18000, 17000, 16000, 17000, 18000, 17000,
               19000, 18000, 20000, 19000, 18000, 19000, 20000, 19000, 18000, 19000, 20000, 19000,
               21000, 20000, 22000, 21000, 20000, 21000, 11881],
    'Russia': [17218, 17000, 17500, 17200, 17100, 17300, 17400, 17200, 17100, 17200, 17300, 17400,
               17100, 16900, 17300, 17100, 16900, 17100, 17300, 17100, 16900, 17000, 17100, 17200,
               13739, 13000, 14000, 13739, 13000, 14000, 15000, 14000, 13000, 14000, 15000, 14000,
               12000, 11000, 13000, 12000, 11000, 12000, 13000, 12000, 11000, 12000, 13000, 12000,
               10000, 9000, 11000, 10000, 9000, 10000, 11000, 10000, 9000, 10000, 11000, 10000,
               8000, 7000, 9000, 8000, 7000, 8000, 9000, 8000, 7000, 8000, 9000, 8000,
               6000, 5000, 7000, 6000, 5000, 6000, 7000, 6000, 5000, 6000, 7000, 6000,
               4000, 3000, 5000, 4000, 3000, 4000, 2116]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert Month to datetime
df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
df = df.set_index('Date')

# Handle missing values in Iran column BEFORE calculating total
df['Iran'] = df['Iran'].replace('-', np.nan)
df['Iran'] = pd.to_numeric(df['Iran'], errors='coerce')
df['Iran'] = df['Iran'].interpolate(method='linear')

# Calculate Total (now safe to sum)
df['Total'] = df[['Algeria', 'Libya', 'Iran', 'Azerbaijan', 'Norway', 'Russia']].sum(axis=1)

print(f"✓ Data loaded successfully!")
print(f"✓ Data shape: {df.shape}")
print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
print(f"✓ Missing values handled: {df.isnull().sum().sum()}")

# Display basic statistics
print("\n2. Data Summary:")
print(df.describe())

# Countries list
countries = ['Algeria', 'Libya', 'Iran', 'Azerbaijan', 'Norway', 'Russia']

print("\n3. Exploratory Data Analysis...")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Time series for all countries
ax1 = axes[0, 0]
for country in countries:
    ax1.plot(df.index, df[country], label=country, linewidth=2, marker='o', markersize=3)
ax1.set_title('Gas Imports by Country Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Gas Imports')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Total imports
ax2 = axes[0, 1]
ax2.plot(df.index, df['Total'], linewidth=3, marker='o', markersize=4, color='darkblue')
ax2.set_title('Total Gas Imports Over Time', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Total Gas Imports')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Correlation matrix
ax3 = axes[1, 0]
correlation_matrix = df[countries + ['Total']].corr()
im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
ax3.set_xticks(range(len(correlation_matrix.columns)))
ax3.set_yticks(range(len(correlation_matrix.columns)))
ax3.set_xticklabels(correlation_matrix.columns, rotation=45)
ax3.set_yticklabels(correlation_matrix.columns)
ax3.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

# Plot 4: Box plots
ax4 = axes[1, 1]
df[countries].boxplot(ax=ax4)
ax4.set_title('Distribution of Gas Imports by Country', fontsize=12, fontweight='bold')
ax4.set_ylabel('Gas Imports')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('gas_import_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizations created and saved as 'gas_import_analysis.png'")

# Time series decomposition
print("\n4. Time Series Analysis...")
try:
    decomposition = seasonal_decompose(df['Total'], model='additive', period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=axes[0], title='Original', color='blue')
    decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
    decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Time series decomposition completed")
    
    # Stationarity test
    def test_stationarity(timeseries):
        result = adfuller(timeseries.dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        
        if result[1] <= 0.05:
            print("✓ Series is stationary")
        else:
            print("✗ Series is non-stationary")
        return result[1] <= 0.05
    
    print("\nStationarity test for Total imports:")
    is_stationary = test_stationarity(df['Total'])
    
except Exception as e:
    print(f"✗ Time series analysis error: {e}")

# Forecasting Models
print("\n5. Forecasting Models...")

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print(f"✓ Training data: {len(train_data)} months")
print(f"✓ Test data: {len(test_data)} months")
print(f"✓ Training period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"✓ Test period: {test_data.index[0]} to {test_data.index[-1]}")

# Function to evaluate forecasts
def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

# Store results
forecast_results = {}

# ARIMA Model
print("\n5.1 ARIMA Model...")
try:
    def fit_arima_model(data, order=(1,1,1)):
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        return fitted_model
    
    # Fit ARIMA model
    arima_model = fit_arima_model(train_data['Total'], order=(2,1,2))
    print("✓ ARIMA Model fitted successfully")
    
    # Make predictions
    arima_predictions = arima_model.predict(start=len(train_data), end=len(df)-1)
    
    # Evaluate ARIMA
    arima_metrics = evaluate_forecast(test_data['Total'], arima_predictions)
    forecast_results['ARIMA'] = {
        'predictions': arima_predictions,
        'metrics': arima_metrics
    }
    
    print("ARIMA Forecast Metrics:")
    for metric, value in arima_metrics.items():
        print(f"  {metric}: {value:.2f}")
        
except Exception as e:
    print(f"✗ ARIMA model error: {e}")

# Simple Moving Average Model
print("\n5.2 Simple Moving Average Model...")
try:
    def simple_moving_average(data, window=12):
        return data.rolling(window=window).mean()
    
    # Calculate SMA predictions
    sma_predictions = simple_moving_average(train_data['Total'], window=12).iloc[-len(test_data):]
    
    # Evaluate SMA
    sma_metrics = evaluate_forecast(test_data['Total'], sma_predictions)
    forecast_results['SMA'] = {
        'predictions': sma_predictions,
        'metrics': sma_metrics
    }
    
    print("SMA Forecast Metrics:")
    for metric, value in sma_metrics.items():
        print(f"  {metric}: {value:.2f}")
        
except Exception as e:
    print(f"✗ SMA model error: {e}")

# LSTM Model (if TensorFlow is available)
print("\n5.3 LSTM Neural Network Model...")
try:
    # Prepare data for LSTM
    def prepare_lstm_data(data, look_back=12):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    # Prepare training data
    X_train, y_train, scaler = prepare_lstm_data(train_data['Total'])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # Build LSTM model
    def build_lstm_model(input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    # Train LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    print("✓ LSTM Model architecture created")
    
    # Train the model
    history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                            validation_split=0.2, verbose=0)
    print("✓ LSTM Model trained successfully")
    
    # Make LSTM predictions
    def make_lstm_predictions(model, scaler, train_data, test_data, look_back=12):
        # Prepare test data
        last_sequence = train_data[-look_back:].values.reshape(1, look_back, 1)
        
        predictions = []
        current_sequence = last_sequence
        
        for _ in range(len(test_data)):
            pred = model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    # Generate LSTM predictions
    lstm_predictions = make_lstm_predictions(lstm_model, scaler, train_data['Total'], test_data['Total'])
    
    # Evaluate LSTM
    lstm_metrics = evaluate_forecast(test_data['Total'], lstm_predictions)
    forecast_results['LSTM'] = {
        'predictions': lstm_predictions,
        'metrics': lstm_metrics
    }
    
    print("LSTM Forecast Metrics:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.2f}")
        
except Exception as e:
    print(f"✗ LSTM model error: {e}")

# Model Comparison
print("\n6. Model Comparison...")
if forecast_results:
    # Compare all models
    comparison_df = pd.DataFrame({
        model: results['metrics'] for model, results in forecast_results.items()
    }).T
    
    print("Model Performance Comparison:")
    print(comparison_df.round(2))
    
    # Find best model
    best_model = comparison_df['MAPE'].idxmin()
    print(f"\n✓ Best performing model: {best_model}")
    print(f"✓ MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%")
    
    # Visualize forecasts
    plt.figure(figsize=(15, 8))
    
    # Plot forecasts
    plt.plot(train_data.index, train_data['Total'], label='Training Data', linewidth=2, color='blue')
    plt.plot(test_data.index, test_data['Total'], label='Actual', linewidth=2, color='green')
    
    for model, results in forecast_results.items():
        plt.plot(test_data.index, results['predictions'], label=f'{model}', linewidth=2, linestyle='--')
    
    plt.title('Gas Import Forecasts Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Gas Imports')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Forecast comparison visualization saved as 'forecast_comparison.png'")
    
    # Future Forecasting
    print("\n7. Future Forecasting...")
    future_months = 12  # Forecast next 12 months
    
    if best_model == 'ARIMA':
        # ARIMA future forecast
        future_forecast = arima_model.forecast(steps=future_months)
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                                    periods=future_months, freq='MS')
        
    elif best_model == 'LSTM':
        # LSTM future forecast
        future_forecast = make_lstm_predictions(lstm_model, scaler, df['Total'], 
                                              pd.Series([0]*future_months), look_back=12)
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                                    periods=future_months, freq='MS')
        
    else:  # SMA
        # SMA future forecast (using last known SMA value)
        last_sma = simple_moving_average(df['Total'], window=12).iloc[-1]
        future_forecast = [last_sma] * future_months
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                                    periods=future_months, freq='MS')
    
    # Create future forecast DataFrame
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_forecast
    }).set_index('Date')
    
    # Plot historical data and future forecast
    plt.figure(figsize=(15, 8))
    
    # Historical data
    plt.plot(df.index, df['Total'], label='Historical Data', linewidth=2, color='blue')
    
    # Training and test periods
    plt.axvline(x=train_data.index[-1], color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    # Future forecast
    plt.plot(future_df.index, future_df['Forecast'], label=f'Future Forecast ({best_model})', 
             linewidth=2, color='red', linestyle='--')
    
    # Add confidence interval (simplified)
    forecast_std = np.std(df['Total'])
    plt.fill_between(future_df.index, 
                     future_df['Forecast'] - 1.96*forecast_std,
                     future_df['Forecast'] + 1.96*forecast_std,
                     alpha=0.2, color='red', label='95% Confidence Interval')
    
    plt.title('Gas Import Forecast with Future Projections', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Gas Imports')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('future_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Future forecast visualization saved as 'future_forecast.png'")
    print("\nFuture Forecast (Next 12 Months):")
    print(future_df.round(0))
    
else:
    print("✗ No models were successfully trained")

# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\nDataset Overview:")
print(f"- Time Period: {df.index[0].strftime('%B %Y')} to {df.index[-1].strftime('%B %Y')}")
print(f"- Total Observations: {len(df)} months")
print(f"- Countries Analyzed: {', '.join(countries)}")
print(f"- Missing Values Handled: Yes (Iran column)")

if forecast_results:
    print(f"\nModel Performance:")
    for model, results in forecast_results.items():
        print(f"- {model}: MAPE = {results['metrics']['MAPE']:.2f}%, RMSE = {results['metrics']['RMSE']:.2f}")
    
    print(f"\nBest Model: {best_model}")
    print(f"- MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%")
    print(f"- RMSE: {comparison_df.loc[best_model, 'RMSE']:.2f}")
    
    print(f"\nFuture Forecast (Next 12 Months):")
    print(f"- Average Monthly Total: {future_df['Forecast'].mean():.0f}")
    print(f"- Forecast Range: {future_df['Forecast'].min():.0f} - {future_df['Forecast'].max():.0f}")

print(f"\nKey Insights:")
print(f"- Total gas imports show {'increasing' if df['Total'].iloc[-12:].mean() > df['Total'].iloc[:12].mean() else 'decreasing'} trend")
print(f"- Norway contributes the most to total imports ({df['Norway'].mean():.0f} average)")
print(f"- Russia shows {'increasing' if df['Russia'].iloc[-12:].mean() > df['Russia'].iloc[:12].mean() else 'decreasing'} trend")

print(f"\nRecommendations:")
if forecast_results:
    print(f"1. Use {best_model} model for future forecasting")
print(f"2. Monitor Norway and Russia imports closely as they are major contributors")
print(f"3. Consider seasonal patterns in forecasting")
print(f"4. Update model regularly with new data")
print(f"5. Consider external factors (geopolitical, economic) in forecasting")

print(f"\nFiles Generated:")
print(f"- gas_import_analysis.png")
print(f"- time_series_decomposition.png")
print(f"- forecast_comparison.png")
print(f"- future_forecast.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

