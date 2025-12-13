"""
Evaluation and Visualization Module
Creates plots and maps for model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import os

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """Create visualizations for model evaluation and results"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.figures_dir = config.get('output', {}).get('figures_dir', 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_predictions_vs_actuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Predictions vs Actuals",
                                   save_path: Optional[str] = None):
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'{title} - Scatter Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Time series plot (if indices available)
        axes[1].plot(y_true[:100], label='Actual', alpha=0.7)
        axes[1].plot(y_pred[:100], label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Value')
        axes[1].set_title(f'{title} - Time Series (First 100 samples)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.figures_dir}/predictions_vs_actuals.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_metrics(self, metrics: Dict, save_path: Optional[str] = None):
        """
        Plot evaluation metrics
        
        Args:
            metrics: Dictionary with train/val/test metrics
            save_path: Path to save figure
        """
        datasets = ['train', 'validation', 'test']
        metric_names = ['rmse', 'mae', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metric_names):
            values = [metrics.get(ds, {}).get(metric, 0) for ds in datasets]
            axes[idx].bar(datasets, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            axes[idx].set_title(f'{metric.upper()}')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.figures_dir}/metrics.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.figures_dir}/feature_importance.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_time_series_by_cell(self, df: pd.DataFrame, cell_ids: List[str],
                                 target_col: str = 'NDVI', 
                                 pred_col: Optional[str] = None,
                                 save_path: Optional[str] = None):
        """
        Plot time series for specific grid cells
        
        Args:
            df: DataFrame with date, cell_id, and target columns
            cell_ids: List of cell IDs to plot
            target_col: Target column name
            pred_col: Prediction column name (optional)
            save_path: Path to save figure
        """
        n_cells = len(cell_ids)
        n_cols = min(3, n_cells)
        n_rows = (n_cells + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_cells == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, cell_id in enumerate(cell_ids):
            cell_data = df[df['cell_id'] == cell_id].sort_values('date')
            
            axes[idx].plot(cell_data['date'], cell_data[target_col], 
                          label='Actual', marker='o', markersize=3)
            
            if pred_col and pred_col in cell_data.columns:
                axes[idx].plot(cell_data['date'], cell_data[pred_col], 
                              label='Predicted', marker='s', markersize=3)
            
            axes[idx].set_title(f'Cell: {cell_id}')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel(target_col)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide extra subplots
        for idx in range(n_cells, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.figures_dir}/time_series_by_cell.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_stress_map(self, df: pd.DataFrame, date: str,
                         stress_col: str = 'NDVI',
                         save_path: Optional[str] = None):
        """
        Create a map showing vegetation stress
        
        Args:
            df: DataFrame with lon, lat, and stress column
            date: Date to visualize
            stress_col: Column name for stress indicator
            save_path: Path to save figure
        """
        if not FOLIUM_AVAILABLE:
            print("Folium not available. Creating matplotlib map instead.")
            self._create_stress_map_matplotlib(df, date, stress_col, save_path)
            return
        
        # Filter data for specific date
        date_data = df[df['date'] == pd.to_datetime(date)]
        
        if len(date_data) == 0:
            print(f"No data for date {date}")
            return
        
        # Create map centered on study area
        center_lat = date_data['lat'].mean()
        center_lon = date_data['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        # Add markers
        for idx, row in date_data.iterrows():
            color = self._get_stress_color(row[stress_col], stress_col)
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                popup=f"{stress_col}: {row[stress_col]:.3f}",
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
        
        if save_path:
            # If save_path is PNG but we're using folium, change to HTML
            if save_path.endswith('.png'):
                save_path = save_path.replace('.png', '.html')
            m.save(save_path)
        else:
            m.save(f"{self.figures_dir}/stress_map_{date}.html")
    
    def _create_stress_map_matplotlib(self, df: pd.DataFrame, date: str,
                                     stress_col: str, save_path: Optional[str]):
        """Create stress map using matplotlib"""
        date_data = df[df['date'] == pd.to_datetime(date)]
        
        if len(date_data) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(date_data['lon'], date_data['lat'], 
                            c=date_data[stress_col], 
                            cmap='RdYlGn', s=50, alpha=0.6)
        plt.colorbar(scatter, label=stress_col)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Vegetation Stress Map - {date}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.figures_dir}/stress_map_{date}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _get_stress_color(self, value: float, stress_col: str) -> str:
        """Get color based on stress value"""
        if stress_col == 'NDVI':
            # NDVI: higher is better (green), lower is stress (red)
            # Adjusted for Pakistan's desert climate (Sindh has lower baseline vegetation)
            if value > 0.5:
                return 'green'  # Healthy irrigated crops
            elif value > 0.3:
                return 'yellow'  # Moderate vegetation/stressed crops
            else:
                return 'red'  # Desert/barren/severe stress
        elif stress_col == 'MSI':
            # MSI: lower is better, higher is stress
            if value < 1.0:
                return 'green'
            elif value < 2.0:
                return 'yellow'
            else:
                return 'red'
        else:
            return 'blue'

