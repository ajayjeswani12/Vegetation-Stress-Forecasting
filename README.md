# Forecasting Vegetation Stress in UK Farmlands

A comprehensive system for predicting vegetation stress in farmland areas by combining satellite indices (NDVI, MSI) with weather data. This project uses Sentinel-2 satellite imagery and meteorological data to forecast potential crop stress or drought conditions.

## 🎯 Project Overview

This project implements a complete workflow for:
- Acquiring Sentinel-2 satellite data (NDVI and MSI indices)
- Fetching weather data (temperature, precipitation, solar radiation)
- Feature engineering with lag features and rolling averages
- Training machine learning models (baseline and advanced time series)
- Evaluating and visualizing predictions

## 📋 Features

- **Satellite Data Acquisition**: Support for Google Earth Engine and OpenEO
- **Weather Data Integration**: Open-Meteo API integration
- **Feature Engineering**: Lag features, rolling averages, temporal features, interactions
- **Multiple Models**: 
  - Baseline: Random Forest, XGBoost, LightGBM
  - Advanced: LSTM, Temporal Convolutional Networks (TCN)
- **Comprehensive Evaluation**: RMSE, MAE, R² metrics
- **Rich Visualizations**: Time series plots, stress maps, feature importance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Earth Engine account (for satellite data)
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Authenticate Google Earth Engine** (if using GEE):
```bash
earthengine authenticate
```

4. **Configure the project**:
   - Edit `config.yaml` to set your study area, time period, and model preferences
   - Default configuration uses East Anglia region (2020-2024)

### Running the Workflow

**Run all steps**:
```bash
python main.py
```

**Run specific steps**:
```bash
python main.py --steps 1 2 3 4 5 7
```

**Train both baseline and advanced models**:
```bash
python main.py --model both
```

**Force rerun (overwrite existing data)**:
```bash
python main.py --force
```

## 📁 Project Structure

```
.
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── main.py                  # Main workflow script
├── README.md                # This file
├── src/
│   ├── data_acquisition/
│   │   ├── satellite_data.py  # Satellite data acquisition
│   │   └── weather_data.py    # Weather data acquisition
│   ├── data_processing/
│   │   └── merge_data.py      # Data merging and cleaning
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering
│   ├── models/
│   │   ├── baseline_model.py       # Baseline models (RF, XGBoost, LightGBM)
│   │   └── advanced_model.py       # Advanced models (LSTM, TCN)
│   └── evaluation/
│       └── visualization.py        # Evaluation and visualization
├── data/                    # Data storage (created automatically)
├── models/                  # Saved models (created automatically)
├── results/                 # Results (created automatically)
└── figures/                 # Visualizations (created automatically)
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Study Area**: Geographic bounds and grid size
- **Time Period**: Start and end dates
- **Satellite Source**: Google Earth Engine or OpenEO
- **Weather Source**: Open-Meteo, ERA5, or Met Office
- **Model Settings**: Model type, hyperparameters
- **Feature Engineering**: Lag periods, rolling windows

### Example Configuration

```yaml
study_area:
  name: "East Anglia"
  bounds:
    min_lon: 0.0
    max_lon: 1.5
    min_lat: 52.0
    max_lat: 53.0
  grid_size_km: 1.0

time_period:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  frequency: "monthly"
```

## 📊 Workflow Steps

### Step 1: Satellite Data Acquisition
- Creates grid cells over study area
- Queries Sentinel-2 images
- Computes NDVI and MSI indices
- Aggregates to monthly averages

### Step 2: Weather Data Acquisition
- Fetches weather data for each grid cell
- Variables: temperature, precipitation, solar radiation
- Aggregates to monthly averages

### Step 3: Data Merging
- Combines satellite and weather data
- Cleans and validates data
- Handles missing values

### Step 4: Feature Engineering
- Creates lag features (previous months)
- Computes rolling averages
- Adds temporal features (month, season)
- Creates interaction features
- Normalizes features per grid cell

### Step 5: Baseline Model Training
- Trains Random Forest, XGBoost, or LightGBM
- Evaluates on train/validation/test sets
- Saves model and metrics

### Step 6: Advanced Model Training (Optional)
- Trains LSTM or TCN for time series prediction
- Uses sequence of previous months as input
- Better for capturing temporal dependencies

### Step 7: Evaluation and Visualization
- Generates prediction vs actual plots
- Creates time series plots per grid cell
- Generates stress maps
- Saves all visualizations

## 📈 Model Performance

Models are evaluated using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

Results are saved in the `results/` directory and visualized in `figures/`.

## 🗺️ Visualizations

The project generates several types of visualizations:

1. **Predictions vs Actuals**: Scatter plots and time series comparisons
2. **Metrics Comparison**: Bar charts of RMSE, MAE, R² across datasets
3. **Feature Importance**: Top features for tree-based models
4. **Time Series by Cell**: NDVI/MSI trends for specific grid cells
5. **Stress Maps**: Geographic visualization of vegetation stress

## 🔧 Customization

### Adding New Features

Edit `src/features/feature_engineering.py` to add custom features:
- Additional lag periods
- New rolling statistics
- Domain-specific interactions

### Using Different Models

Modify `config.yaml` to switch between:
- Model types (random_forest, xgboost, lightgbm, lstm, tcn)
- Hyperparameters
- Training settings

### Alternative Data Sources

The codebase is designed to be extensible:
- Add new satellite sources in `src/data_acquisition/satellite_data.py`
- Add new weather sources in `src/data_acquisition/weather_data.py`

## 📝 Notes

### Data Acquisition

- **Google Earth Engine**: Requires authentication and may have usage limits
- **Open-Meteo**: Free API with rate limits (built-in delays)
- **Processing Time**: Satellite data processing can be time-consuming for large areas

### Model Selection

- **Baseline Models**: Fast training, good for initial experiments
- **Advanced Models**: Better for capturing temporal patterns, requires more data

### Limitations

- Current implementation uses simplified GEE data extraction (may need refinement for production)
- Weather data aggregation assumes monthly frequency
- Grid cell size affects resolution vs processing time trade-off

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Report issues
- Suggest improvements
- Add new features
- Extend to other regions

## 📄 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- Sentinel-2 data via Google Earth Engine / OpenEO
- Weather data via Open-Meteo API
- Built with Python, scikit-learn, TensorFlow, and other open-source tools

## 📚 References

- NDVI: Normalized Difference Vegetation Index
- MSI: Moisture Stress Index (SWIR1/NIR ratio)
- Sentinel-2: ESA's Earth observation satellite
- ERA5: ECMWF's reanalysis dataset

---

**For questions or issues**, please check the code comments or configuration file for detailed explanations.

