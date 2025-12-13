"""
Main Workflow Script
Forecasting Vegetation Stress in UK Farmlands
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict

from src.data_acquisition import SatelliteDataAcquisition, WeatherDataAcquisition
from src.data_processing import DataMerger
from src.features import FeatureEngineer
from src.models import BaselineModel, AdvancedTimeSeriesModel
from src.evaluation import Visualizer


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def step1_acquire_satellite_data(config: Dict, force_rerun: bool = False) -> pd.DataFrame:
    """Step 1: Acquire satellite data (NDVI, MSI)"""
    print("\n" + "="*60)
    print("STEP 1: Acquiring Satellite Data")
    print("="*60)
    
    data_dir = Path(config['output']['data_dir'])
    data_dir.mkdir(exist_ok=True)
    satellite_file = data_dir / 'satellite_data.csv'
    
    if satellite_file.exists() and not force_rerun:
        print(f"Loading existing satellite data from {satellite_file}")
        return pd.read_csv(satellite_file, parse_dates=['date'])
    
    study_area = config['study_area']
    time_period = config['time_period']
    
    print("Initializing satellite data acquisition...")
    satellite_acq = SatelliteDataAcquisition(config)
    
    print(f"Study area: {study_area['name']}")
    print(f"Time period: {time_period['start_date']} to {time_period['end_date']}")
    
    satellite_df = satellite_acq.get_satellite_data(
        start_date=time_period['start_date'],
        end_date=time_period['end_date'],
        bounds=study_area['bounds'],
        grid_size_km=study_area['grid_size_km']
    )
    
    satellite_df.to_csv(satellite_file, index=False)
    print(f"Satellite data saved to {satellite_file}")
    print(f"Total records: {len(satellite_df)}")
    
    return satellite_df


def step2_acquire_weather_data(config: Dict, satellite_df: pd.DataFrame,
                               force_rerun: bool = False) -> pd.DataFrame:
    """Step 2: Acquire weather data"""
    print("\n" + "="*60)
    print("STEP 2: Acquiring Weather Data")
    print("="*60)
    
    data_dir = Path(config['output']['data_dir'])
    weather_file = data_dir / 'weather_data.csv'
    
    if weather_file.exists() and not force_rerun:
        print(f"Loading existing weather data from {weather_file}")
        return pd.read_csv(weather_file, parse_dates=['date'])
    
    time_period = config['time_period']
    
    print("Initializing weather data acquisition...")
    weather_acq = WeatherDataAcquisition(config)
    
    # Get unique grid cells from satellite data
    grid_df = satellite_df[['cell_id', 'lon', 'lat']].drop_duplicates()
    
    weather_df = weather_acq.get_weather_data(
        grid_df=grid_df,
        start_date=time_period['start_date'],
        end_date=time_period['end_date']
    )
    
    weather_df.to_csv(weather_file, index=False)
    print(f"Weather data saved to {weather_file}")
    print(f"Total records: {len(weather_df)}")
    
    return weather_df


def step3_merge_data(config: Dict, satellite_df: pd.DataFrame,
                    weather_df: pd.DataFrame, force_rerun: bool = False) -> pd.DataFrame:
    """Step 3: Merge satellite and weather data"""
    print("\n" + "="*60)
    print("STEP 3: Merging Data")
    print("="*60)
    
    data_dir = Path(config['output']['data_dir'])
    merged_file = data_dir / 'merged_data.csv'
    
    if merged_file.exists() and not force_rerun:
        print(f"Loading existing merged data from {merged_file}")
        return pd.read_csv(merged_file, parse_dates=['date'])
    
    merger = DataMerger(config)
    
    merged_df = merger.merge_satellite_weather(satellite_df, weather_df)
    merged_df = merger.clean_data(merged_df)
    merged_df = merger.aggregate_to_monthly(merged_df)
    
    merged_df.to_csv(merged_file, index=False)
    print(f"Merged data saved to {merged_file}")
    print(f"Total records: {len(merged_df)}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    return merged_df


def step4_feature_engineering(config: Dict, merged_df: pd.DataFrame,
                             force_rerun: bool = False) -> pd.DataFrame:
    """Step 4: Feature engineering"""
    print("\n" + "="*60)
    print("STEP 4: Feature Engineering")
    print("="*60)
    
    data_dir = Path(config['output']['data_dir'])
    features_file = data_dir / 'features_data.csv'
    
    if features_file.exists() and not force_rerun:
        print(f"Loading existing features from {features_file}")
        return pd.read_csv(features_file, parse_dates=['date'])
    
    engineer = FeatureEngineer(config)
    
    features_df = engineer.engineer_features(merged_df, target_column='NDVI')
    
    features_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_file, index=False)
    print(f"Features saved to {features_file}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"Feature columns: {len([c for c in features_df.columns if c not in ['date', 'cell_id', 'lon', 'lat', 'NDVI', 'MSI', 'NDVI_next', 'MSI_next']])}")
    
    return features_df


def step5_train_baseline_model(config: Dict, features_df: pd.DataFrame,
                              force_rerun: bool = False):
    """Step 5: Train baseline model"""
    print("\n" + "="*60)
    print("STEP 5: Training Baseline Model")
    print("="*60)
    
    models_dir = Path(config['output']['models_dir'])
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / 'baseline_model.pkl'
    
    if model_file.exists() and not force_rerun:
        print(f"Baseline model already exists at {model_file}")
        print("Use --force to retrain")
        return
    
    model = BaselineModel(config)
    
    X, y = model.prepare_features(features_df, target_column='NDVI_next')
    
    eval_config = config.get('evaluation', {})
    metrics = model.train(
        X, y,
        test_size=eval_config.get('test_split', 0.2),
        validation_split=eval_config.get('validation_split', 0.2)
    )
    
    # Print metrics
    print("\nModel Performance:")
    for dataset in ['train', 'validation', 'test']:
        print(f"\n{dataset.upper()}:")
        for metric in ['rmse', 'mae', 'r2']:
            print(f"  {metric.upper()}: {metrics[dataset][metric]:.4f}")
    
    # Save model
    model.save(str(model_file))
    print(f"\nModel saved to {model_file}")
    
    # Visualize
    visualizer = Visualizer(config)
    figures_dir = Path(config['output']['figures_dir'])
    visualizer.plot_metrics(
        metrics,
        save_path=str(figures_dir / 'metrics_baseline.png')
    )
    
    if 'feature_importance' in metrics:
        visualizer.plot_feature_importance(
            metrics['feature_importance'],
            save_path=str(figures_dir / 'feature_importance_baseline.png')
        )
    
    return model, metrics


def step6_train_advanced_model(config: Dict, features_df: pd.DataFrame,
                               force_rerun: bool = False):
    """Step 6: Train advanced time series model"""
    print("\n" + "="*60)
    print("STEP 6: Training Advanced Time Series Model")
    print("="*60)
    
    models_dir = Path(config['output']['models_dir'])
    model_file = models_dir / 'advanced_model.h5'
    # Check for actual saved files (save method creates _model.h5 and _metadata.pkl)
    actual_model_file = models_dir / 'advanced_model_model.h5'
    metadata_file = models_dir / 'advanced_model_metadata.pkl'
    
    if actual_model_file.exists() and metadata_file.exists() and not force_rerun:
        print(f"Advanced model already exists:")
        print(f"  Model: {actual_model_file}")
        print(f"  Metadata: {metadata_file}")
        print("Use --force to retrain")
        return
    
    try:
        model = AdvancedTimeSeriesModel(config)
        
        eval_config = config.get('evaluation', {})
        metrics = model.train(
            features_df,
            target_column='NDVI_next',
            test_size=eval_config.get('test_split', 0.2),
            validation_split=eval_config.get('validation_split', 0.2)
        )
        
        # Print metrics
        print("\nModel Performance:")
        for dataset in ['train', 'validation', 'test']:
            print(f"\n{dataset.upper()}:")
            for metric in ['rmse', 'mae', 'r2']:
                print(f"  {metric.upper()}: {metrics[dataset][metric]:.4f}")
        
        # Save model
        model.save(str(model_file))
        print(f"\nModel saved to {model_file}")
        
        # Visualize
        visualizer = Visualizer(config)
        figures_dir = Path(config['output']['figures_dir'])
        visualizer.plot_metrics(
            metrics,
            save_path=str(figures_dir / 'metrics_advanced.png')
        )
        
        # Save metrics to JSON file
        import json
        metrics_file = models_dir / 'advanced_model_metrics.json'
        metrics_serializable = {}
        for dataset, data in metrics.items():
            if dataset != 'history':
                metrics_serializable[dataset] = {
                    k: float(v) for k, v in data.items()
                }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
        
        return model, metrics
    except ImportError as e:
        print(f"Advanced model training skipped: {e}")
        return None, None


def step7_evaluation_and_visualization(config: Dict, features_df: pd.DataFrame,
                                      model, model_type: str = 'baseline'):
    """Step 7: Evaluation and visualization"""
    print("\n" + "="*60)
    print(f"STEP 7: Evaluation and Visualization - {model_type.upper()} Model")
    print("="*60)
    
    visualizer = Visualizer(config)
    
    # Load and display saved metrics if available (for advanced models)
    if model_type == 'advanced':
        models_dir = Path(config['output']['models_dir'])
        metrics_file = models_dir / 'advanced_model_metrics.json'
        if metrics_file.exists():
            import json
            try:
                with open(metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                print("\nSaved Model Performance Metrics:")
                for dataset in ['train', 'validation', 'test']:
                    if dataset in saved_metrics:
                        print(f"\n{dataset.upper()}:")
                        for metric in ['rmse', 'mae', 'r2']:
                            if metric in saved_metrics[dataset]:
                                print(f"  {metric.upper()}: {saved_metrics[dataset][metric]:.4f}")
            except Exception as e:
                print(f"Could not load saved metrics: {e}")
    
    # Get predictions
    if model_type == 'baseline':
        X, y = model.prepare_features(features_df, target_column='NDVI_next')
        y_pred = model.predict(X)
        y_true = y.values
    else:
        # For advanced models, create sequences and get predictions
        try:
            print(f"Evaluating {model_type} model...")
            
            # Prepare features and create sequences
            df_cleaned, feature_cols = model.prepare_features(features_df, target_column='NDVI_next')
            
            if model.feature_columns is None:
                model.feature_columns = feature_cols
            
            print(f"Creating sequences with {len(feature_cols)} features...")
            X, y = model.create_sequences(df_cleaned, feature_cols, 'NDVI_next')
            
            if len(X) == 0:
                print("ERROR: No valid sequences created. Check data and sequence_length.")
                return
            
            print(f"Created {len(X)} sequences for evaluation")
            
            # Get predictions
            y_pred = model.predict(X)
            y_true = y
            
            print(f"Predictions shape: {y_pred.shape}, True values shape: {y_true.shape}")
            
        except Exception as e:
            print(f"Error evaluating advanced model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Get figures directory
    figures_dir = Path(config['output']['figures_dir'])
    
    # Plot predictions vs actuals (with model-specific filename)
    visualizer.plot_predictions_vs_actuals(
        y_true, y_pred,
        title=f"{model_type.capitalize()} Model Predictions",
        save_path=str(figures_dir / f"predictions_vs_actuals_{model_type}.png")
    )
    
    # Plot time series for sample cells (with model-specific filename)
    sample_cells = features_df['cell_id'].unique()[:6]
    visualizer.plot_time_series_by_cell(
        features_df, list(sample_cells),
        target_col='NDVI',
        save_path=str(figures_dir / f"time_series_by_cell_{model_type}.png")
    )
    
    # Create stress maps for recent dates (with model-specific filename)
    recent_dates = features_df['date'].unique()[-3:]
    for date in recent_dates:
        date_str = str(date.date()).replace('-', '_')
        # Use HTML for folium maps (or PNG if matplotlib is used)
        visualizer.create_stress_map(
            features_df, str(date.date()),
            stress_col='NDVI',
            save_path=str(figures_dir / f"stress_map_{date_str}_{model_type}.html")
        )
    
    print(f"\nVisualizations saved to {figures_dir}/ directory:")
    print(f"  - predictions_vs_actuals_{model_type}.png")
    print(f"  - time_series_by_cell_{model_type}.png")
    print(f"  - stress_map_*_{model_type}.html (or .png if using matplotlib)")


def main():
    """Main workflow"""
    parser = argparse.ArgumentParser(description='Vegetation Stress Forecasting')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--steps', type=str, nargs='+',
                       choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
                       default=['all'],
                       help='Steps to run (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun even if output exists')
    parser.add_argument('--model', type=str, choices=['baseline', 'advanced', 'both'],
                       default='both',
                       help='Which model to train (default: both)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run steps
    steps_to_run = args.steps
    if 'all' in steps_to_run:
        steps_to_run = ['1', '2', '3', '4', '5', '6', '7']
    
    satellite_df = None
    weather_df = None
    merged_df = None
    features_df = None
    
    if '1' in steps_to_run:
        satellite_df = step1_acquire_satellite_data(config, args.force)
    
    if '2' in steps_to_run:
        if satellite_df is None:
            satellite_df = step1_acquire_satellite_data(config, False)
        weather_df = step2_acquire_weather_data(config, satellite_df, args.force)
    
    if '3' in steps_to_run:
        if satellite_df is None:
            satellite_df = step1_acquire_satellite_data(config, False)
        if weather_df is None:
            weather_df = step2_acquire_weather_data(config, satellite_df, False)
        merged_df = step3_merge_data(config, satellite_df, weather_df, args.force)
    
    if '4' in steps_to_run:
        if merged_df is None:
            if satellite_df is None:
                satellite_df = step1_acquire_satellite_data(config, False)
            if weather_df is None:
                weather_df = step2_acquire_weather_data(config, satellite_df, False)
            merged_df = step3_merge_data(config, satellite_df, weather_df, False)
        features_df = step4_feature_engineering(config, merged_df, args.force)
    
    if '5' in steps_to_run and args.model in ['baseline', 'both']:
        if features_df is None:
            if merged_df is None:
                if satellite_df is None:
                    satellite_df = step1_acquire_satellite_data(config, False)
                if weather_df is None:
                    weather_df = step2_acquire_weather_data(config, satellite_df, False)
                merged_df = step3_merge_data(config, satellite_df, weather_df, False)
            features_df = step4_feature_engineering(config, merged_df, False)
        model, metrics = step5_train_baseline_model(config, features_df, args.force)
    
    if '6' in steps_to_run and args.model in ['advanced', 'both']:
        if features_df is None:
            if merged_df is None:
                if satellite_df is None:
                    satellite_df = step1_acquire_satellite_data(config, False)
                if weather_df is None:
                    weather_df = step2_acquire_weather_data(config, satellite_df, False)
                merged_df = step3_merge_data(config, satellite_df, weather_df, False)
            features_df = step4_feature_engineering(config, merged_df, False)
        model, metrics = step6_train_advanced_model(config, features_df, args.force)
    
    if '7' in steps_to_run:
        if features_df is None:
            if merged_df is None:
                if satellite_df is None:
                    satellite_df = step1_acquire_satellite_data(config, False)
                if weather_df is None:
                    weather_df = step2_acquire_weather_data(config, satellite_df, False)
                merged_df = step3_merge_data(config, satellite_df, weather_df, False)
            features_df = step4_feature_engineering(config, merged_df, False)
        
        models_dir = Path(config['output']['models_dir'])
        models_evaluated = False
        
        # Try to evaluate baseline model
        from src.models import BaselineModel
        baseline_model = BaselineModel(config)
        baseline_file = models_dir / 'baseline_model.pkl'
        
        if baseline_file.exists():
            print("\n" + "="*60)
            print("Evaluating Baseline Model")
            print("="*60)
            baseline_model.load(str(baseline_file))
            step7_evaluation_and_visualization(config, features_df, baseline_model, 'baseline')
            models_evaluated = True
        
        # Try to evaluate advanced model
        from src.models import AdvancedTimeSeriesModel
        # The save method creates advanced_model_model.h5, so check for that
        advanced_model_file = models_dir / 'advanced_model_model.h5'
        advanced_metadata_file = models_dir / 'advanced_model_metadata.pkl'
        
        # Check if either the model file or metadata exists (both are needed)
        if advanced_model_file.exists() and advanced_metadata_file.exists():
            print("\n" + "="*60)
            print("Evaluating Advanced Model")
            print("="*60)
            try:
                advanced_model = AdvancedTimeSeriesModel(config)
                # Load using the base filename (load method will find _model.h5 and _metadata.pkl)
                advanced_model.load(str(models_dir / 'advanced_model.h5'))
                step7_evaluation_and_visualization(config, features_df, advanced_model, 'advanced')
                models_evaluated = True
            except Exception as e:
                print(f"Error loading advanced model: {e}")
                import traceback
                traceback.print_exc()
        elif advanced_model_file.exists() or advanced_metadata_file.exists():
            print("\nWARNING: Advanced model files incomplete:")
            print(f"  Model file exists: {advanced_model_file.exists()}")
            print(f"  Metadata file exists: {advanced_metadata_file.exists()}")
            print("  Both files are required. Please retrain the model.")
        
        if not models_evaluated:
            print("\nNo models found. Train models first:")
            print("  Baseline: python main.py --steps 5")
            print("  Advanced: python main.py --steps 6 --model advanced")
    
    print("\n" + "="*60)
    print("Workflow completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

