"""
Data processing and collection utilities for Japan Gas Demand Forecasting
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class JapanGasDataCollector:
    """
    Class for collecting and processing Japanese gas demand data
    """
    
    def __init__(self):
        self.data_sources = {
            'meti_energy': 'https://www.enecho.meti.go.jp/statistics/',
            'jma_weather': 'https://www.jma.go.jp/jma/indexe.html',
            'boj_economics': 'https://www.boj.or.jp/en/',
            'jga_industry': 'https://www.gas.or.jp/en/'
        }
        
    def generate_synthetic_data(self, start_date='2018-01-01', end_date='2024-08-31'):
        """
        Generate synthetic but realistic Japanese gas demand data for demonstration
        
        Parameters:
        -----------
        start_date : str
            Start date for data generation
        end_date : str
            End date for data generation
            
        Returns:
        --------
        pandas.DataFrame : Synthetic gas demand dataset
        """
        dates = pd.date_range(start_date, end_date, freq='MS')
        np.random.seed(42)  # For reproducibility
        
        n_months = len(dates)
        base_monthly = 37000 / 12  # Million cubic meters per month (Japan's approx annual consumption)
        
        # 1. Trend component (gradual efficiency improvements)
        trend = np.linspace(1.0, 0.95, n_months)
        
        # 2. Seasonal component (winter heating dominance)
        months = np.array([d.month for d in dates])
        seasonal = 1.0 + 0.4 * np.cos(2 * np.pi * (months - 1) / 12)
        
        # 3. Economic cycle
        years = np.array([(d - pd.Timestamp(start_date)).days / 365.25 for d in dates])
        economic_cycle = 1.0 + 0.08 * np.sin(2 * np.pi * years / 5)
        
        # 4. COVID-19 impact
        covid_impact = np.ones(n_months)
        for i, date in enumerate(dates):
            if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-08-01'):
                covid_impact[i] = 0.85 - 0.05 * np.random.random()
        
        # 5. Random noise
        noise = 1.0 + np.random.normal(0, 0.04, n_months)
        
        # Combine components
        total_demand = base_monthly * trend * seasonal * economic_cycle * covid_impact * noise
        
        # Sector breakdown
        residential_share = 0.30 + 0.08 * np.cos(2 * np.pi * months / 12)
        commercial_share = 0.16 + 0.03 * np.cos(2 * np.pi * months / 12)
        industrial_share = 0.38 - 0.04 * np.cos(2 * np.pi * months / 12)
        power_share = 1.0 - residential_share - commercial_share - industrial_share
        
        # Weather data
        temperature = np.array([
            14 + 11 * np.cos(2 * np.pi * (d.month - 7) / 12) + np.random.normal(0, 2.5)
            for d in dates
        ])
        
        hdd = np.maximum(0, 18 - temperature) * 30  # Monthly heating degree days
        cdd = np.maximum(0, temperature - 24) * 30  # Monthly cooling degree days
        
        # Economic indicators
        gdp_growth = np.random.normal(0.7, 1.1, n_months)
        industrial_production = 100 + np.cumsum(np.random.normal(0.15, 1.8, n_months))
        
        # Energy prices (simplified)
        gas_price = 50 + 12 * np.sin(2 * np.pi * np.arange(n_months) / 12) + np.random.normal(0, 3, n_months)
        oil_price = 70 + 25 * np.sin(2 * np.pi * np.arange(n_months) / 18) + np.random.normal(0, 5, n_months)
        
        # Create dataset
        data = pd.DataFrame({
            'date': dates,
            'total_gas_demand_mcm': total_demand,
            'residential_demand_mcm': total_demand * residential_share,
            'commercial_demand_mcm': total_demand * commercial_share,
            'industrial_demand_mcm': total_demand * industrial_share,
            'power_generation_demand_mcm': total_demand * power_share,
            'avg_temperature_celsius': temperature,
            'heating_degree_days': hdd,
            'cooling_degree_days': cdd,
            'gdp_growth_rate_pct': gdp_growth,
            'industrial_production_index': industrial_production,
            'gas_price_jpy_per_mcm': gas_price,
            'crude_oil_price_usd_per_barrel': oil_price
        })
        
        data.set_index('date', inplace=True)
        
        return data
    
    def add_calendar_features(self, df):
        """
        Add calendar-based features to the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with added calendar features
        """
        df = df.copy()
        
        # Basic calendar features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Seasonal indicators
        df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)
        df['is_spring'] = df.index.month.isin([3, 4, 5]).astype(int)
        df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['is_autumn'] = df.index.month.isin([9, 10, 11]).astype(int)
        
        # Heating/cooling season indicators
        df['heating_season'] = df.index.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        df['cooling_season'] = df.index.month.isin([6, 7, 8, 9]).astype(int)
        
        # Cyclical encoding of month (preserves seasonal relationships)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        return df
    
    def create_lagged_features(self, df, target_col, max_lag=12):
        """
        Create lagged versions of target variable and other features
        """
        df = df.copy()
        
        # Target variable lags
        for lag in [1, 2, 3, 6, 12]:
            if lag <= max_lag:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
        
        # Year-over-year change
        df[f'{target_col}_yoy_change'] = df[target_col].pct_change(12) * 100
        
        return df
    
    def clean_and_validate_data(self, df, target_col='total_gas_demand_mcm'):
        """
        Clean and validate the dataset
        """
        df = df.copy()
        
        print("🧹 DATA CLEANING AND VALIDATION")
        print("=" * 40)
        
        initial_shape = df.shape
        print(f"Initial dataset shape: {initial_shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Remove negative demand values (data quality issue)
        if target_col in df.columns:
            negative_mask = df[target_col] < 0
            if negative_mask.sum() > 0:
                print(f"Removing {negative_mask.sum()} negative demand observations")
                df = df[~negative_mask]
        
        # Check for outliers using IQR method
        if target_col in df.columns:
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[target_col] < Q1 - 3*IQR) | (df[target_col] > Q3 + 3*IQR)
            
            if outlier_mask.sum() > 0:
                print(f"Potential outliers detected: {outlier_mask.sum()} observations")
                print(f"Outlier values: {df.loc[outlier_mask, target_col].tolist()}")
        
        # Sort by date
        df = df.sort_index()
        
        final_shape = df.shape
        print(f"Final dataset shape: {final_shape}")
        print(f"Data cleaned successfully ✓")
        
        return df


def load_sample_data():
    """
    Quick function to load sample data for analysis
    """
    collector = JapanGasDataCollector()
    
    # Generate base data
    data = collector.generate_synthetic_data()
    
    # Add calendar features
    data = collector.add_calendar_features(data)
    
    # Create lagged features
    data = collector.create_lagged_features(data, 'total_gas_demand_mcm')
    
    # Clean and validate
    data = collector.clean_and_validate_data(data)
    
    return data


def calculate_seasonal_indices(data, target_col, period=12):
    """
    Calculate seasonal indices for the target variable
    """
    # Group by month and calculate average
    seasonal_averages = data.groupby(data.index.month)[target_col].mean()
    overall_average = data[target_col].mean()
    
    # Calculate seasonal indices
    seasonal_indices = seasonal_averages / overall_average
    
    return seasonal_indices


def detect_structural_breaks(series, method='chow_test'):
    """
    Detect structural breaks in time series
    (Simplified implementation for demonstration)
    """
    from scipy import stats
    
    n = len(series)
    mid_point = n // 2
    
    # Split series in half
    first_half = series.iloc[:mid_point]
    second_half = series.iloc[mid_point:]
    
    # Perform t-test for difference in means
    t_stat, p_value = stats.ttest_ind(first_half, second_half)
    
    has_break = p_value < 0.05
    
    return {
        'has_structural_break': has_break,
        'break_point_approx': series.index[mid_point] if has_break else None,
        'p_value': p_value,
        't_statistic': t_stat,
        'first_half_mean': first_half.mean(),
        'second_half_mean': second_half.mean()
    }


def weather_demand_correlation_analysis(data):
    """
    Analyze correlation between weather variables and gas demand
    """
    weather_cols = ['avg_temperature_celsius', 'heating_degree_days', 'cooling_degree_days']
    demand_cols = [col for col in data.columns if 'demand' in col and 'mcm' in col]
    
    correlations = {}
    
    for weather_var in weather_cols:
        if weather_var in data.columns:
            correlations[weather_var] = {}
            for demand_var in demand_cols:
                if demand_var in data.columns:
                    corr = data[weather_var].corr(data[demand_var])
                    correlations[weather_var][demand_var] = corr
    
    return correlations


# Constants for Japan gas market
JAPAN_GAS_CONSTANTS = {
    'annual_consumption_bcm': 37.0,  # Billion cubic meters
    'peak_winter_multiplier': 1.4,
    'base_temperature_celsius': 18,
    'major_consuming_regions': ['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka'],
    'typical_seasonal_variation': 0.35,  # 35% coefficient of variation
    'industrial_share': 0.38,
    'residential_share': 0.30,
    'commercial_share': 0.16,
    'power_generation_share': 0.16
}