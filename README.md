# German Gas Demand Forecasting Analysis

A comprehensive forecasting analysis for German gas demand with separate models for SLP (Standardized Load Profile) and RLM (Registered Load Management) consumers.

## Overview

This project provides forecasting analysis for German gas demand with separate models for:
- **SLP (Standardized Load Profile)**: Small consumers with standardized consumption patterns
- **RLM (Registered Load Management)**: Large industrial consumers with registered metering

## Data Sources

- **Primary Source**: [TradingHub.eu Aggregated Consumption Data](https://www.tradinghub.eu/en-gb/Publications/Transparency/Aggregated-consumption-data)
- Historical gas consumption patterns for SLP and RLM segments
- Weather data and temperature correlations (Heating Degree Days)
- Economic factors affecting industrial demand

## Key Features

- **Multi-Segment Analysis**: Separate forecasting for SLP and RLM consumers
- **Weather Integration**: Temperature correlations and Heating Degree Days
- **Seasonal Decomposition**: Trend and seasonal pattern analysis
- **Multiple Models**: Various forecasting algorithms with 2-year horizon
- **Interactive Visualizations**: Plotly-based charts and dashboards

## German Gas Market Characteristics

- **SLP Customers**: Primarily residential and small commercial
- **RLM Customers**: Industrial, large commercial, power generation
- **Seasonal Patterns**: Strong winter heating demand
- **Industrial Cycles**: Weekly and monthly production patterns

## Project Structure

```
├── German_Gas_Demand_Forecast_SLP_RLM.ipynb  # Main analysis notebook
├── README.md                                  # This file
├── requirements.txt                          # Python dependencies
└── data/                                     # Data files (if any)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- plotly
- scikit-learn (for forecasting models)
- matplotlib/seaborn (for additional visualizations)

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook to run the analysis

## Usage

1. Ensure you have the required data files
2. Open `German_Gas_Demand_Forecast_SLP_RLM.ipynb`
3. Run the cells sequentially to perform the analysis
4. Modify parameters as needed for your specific use case

## Contributing

This is a personal analysis project. Feel free to fork and modify for your own use.

## License

This project is for educational and research purposes.

## Contact

- GitHub: [@timcrumpfge](https://github.com/timcrumpfge)
- Analysis Date: October 2025
