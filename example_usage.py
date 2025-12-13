"""
Example Usage Script
Demonstrates how to use the vegetation stress forecasting system
"""

import yaml
from pathlib import Path
from src.data_acquisition import SatelliteDataAcquisition, WeatherDataAcquisition
from src.data_processing import DataMerger
from src.features import FeatureEngineer
from src.models import BaselineModel
from src.evaluation import Visualizer

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Example 1: Quick data acquisition for a small area
print("Example 1: Acquiring data for a small test area")
print("-" * 60)

# Modify bounds for a smaller test area
test_bounds = {
    'min_lon': 0.5,
    'max_lon': 0.6,
    'min_lat': 52.2,
    'max_lat': 52.3
}

# Initialize satellite acquisition
satellite_acq = SatelliteDataAcquisition(config)
grid_gdf = satellite_acq.create_grid(test_bounds, grid_size_km=1.0)
print(f"Created grid with {len(grid_gdf)} cells")

# Example 2: Feature engineering on sample data
print("\nExample 2: Feature engineering")
print("-" * 60)

import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2020-01-01', '2021-12-31', freq='MS')
sample_data = []
for cell_id in ['cell_0_0', 'cell_0_1']:
    for date in dates:
        sample_data.append({
            'date': date,
            'cell_id': cell_id,
            'lon': 0.5 + np.random.random() * 0.1,
            'lat': 52.2 + np.random.random() * 0.1,
            'NDVI': 0.3 + np.random.random() * 0.4,
            'MSI': 1.0 + np.random.random() * 1.5,
            'temperature_2m': 10 + np.random.random() * 10,
            'precipitation': np.random.random() * 50
        })

sample_df = pd.DataFrame(sample_data)

# Engineer features
engineer = FeatureEngineer(config)
features_df = engineer.engineer_features(sample_df, target_column='NDVI')
print(f"Original columns: {len(sample_df.columns)}")
print(f"Features after engineering: {len(features_df.columns)}")
print(f"Sample feature columns: {[c for c in features_df.columns if 'lag' in c or 'rolling' in c][:5]}")

# Example 3: Training a simple model
print("\nExample 3: Training a baseline model")
print("-" * 60)

# Prepare features
model = BaselineModel(config)
X, y = model.prepare_features(features_df, target_column='NDVI_next')

if len(X) > 0 and len(y) > 0:
    print(f"Training on {len(X)} samples with {len(X.columns)} features")
    
    # Train model
    metrics = model.train(X, y, test_size=0.2, validation_split=0.2)
    
    print("\nModel Performance:")
    for dataset in ['train', 'validation', 'test']:
        if dataset in metrics:
            print(f"\n{dataset.upper()}:")
            for metric in ['rmse', 'mae', 'r2']:
                if metric in metrics[dataset]:
                    print(f"  {metric.upper()}: {metrics[dataset][metric]:.4f}")
else:
    print("Not enough data for training (need sequences)")

# Example 4: Visualization
print("\nExample 4: Creating visualizations")
print("-" * 60)

visualizer = Visualizer(config)

# Plot time series for sample cells
if len(features_df) > 0:
    cell_ids = features_df['cell_id'].unique()[:2]
    visualizer.plot_time_series_by_cell(
        features_df, 
        list(cell_ids),
        target_col='NDVI'
    )
    print(f"Created time series plots for {len(cell_ids)} cells")

print("\n" + "=" * 60)
print("Examples completed!")
print("=" * 60)
print("\nTo run the full workflow, use:")
print("  python main.py")
print("\nOr run specific steps:")
print("  python main.py --steps 1 2 3 4 5 7")

