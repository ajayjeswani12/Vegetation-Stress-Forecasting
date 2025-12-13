"""
Feature Engineering Module
Creates lag features, rolling averages, and other derived features
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    """Create features for vegetation stress prediction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_config = config.get('features', {})
        self.lag_months = self.feature_config.get('lag_months', [1, 2, 3])
        self.rolling_windows = self.feature_config.get('rolling_windows', [3, 6])
        self.should_normalize_per_cell = self.feature_config.get('normalize_per_cell', True)
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for specified columns
        
        Args:
            df: DataFrame sorted by cell_id and date
            columns: List of column names to create lags for
            lags: List of lag periods (in months)
        
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        df = df.sort_values(['cell_id', 'date']).reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                lag_col = f'{col}_lag_{lag}'
                df[lag_col] = df.groupby('cell_id')[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Create rolling average features
        
        Args:
            df: DataFrame sorted by cell_id and date
            columns: List of column names
            windows: List of window sizes (in months)
        
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        df = df.sort_values(['cell_id', 'date']).reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                roll_col = f'{col}_rolling_{window}'
                df[roll_col] = df.groupby('cell_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features (month, season, etc.)
        
        Args:
            df: DataFrame with date column
        
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        # Season (Northern Hemisphere)
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,   # Winter
            3: 1, 4: 1, 5: 1,    # Spring
            6: 2, 7: 2, 8: 2,    # Summer
            9: 3, 10: 3, 11: 3   # Fall
        })
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def normalize_per_cell(self, df: pd.DataFrame, 
                          columns: List[str]) -> pd.DataFrame:
        """
        Normalize features per grid cell (z-score normalization)
        
        Args:
            df: DataFrame with cell_id column
            columns: List of columns to normalize
        
        Returns:
            DataFrame with normalized columns
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Calculate mean and std per cell
            cell_stats = df.groupby('cell_id')[col].agg(['mean', 'std'])
            
            # Merge back
            df = df.merge(cell_stats, left_on='cell_id', right_index=True, how='left')
            
            # Normalize
            norm_col = f'{col}_normalized'
            df[norm_col] = (df[col] - df['mean']) / (df['std'] + 1e-8)
            
            # Drop intermediate columns
            df = df.drop(['mean', 'std'], axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # NDVI * Temperature (stress indicator)
        if 'NDVI' in df.columns and 'temperature_2m' in df.columns:
            df['NDVI_temp_interaction'] = df['NDVI'] * df['temperature_2m']
        
        # MSI * Precipitation (moisture availability)
        if 'MSI' in df.columns and 'precipitation' in df.columns:
            df['MSI_precip_interaction'] = df['MSI'] * df['precipitation']
        
        # NDVI / MSI (vegetation health vs stress)
        if 'NDVI' in df.columns and 'MSI' in df.columns:
            df['NDVI_MSI_ratio'] = df['NDVI'] / (df['MSI'] + 1e-8)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, 
                         target_column: str = 'NDVI') -> pd.DataFrame:
        """
        Main method to engineer all features
        
        Args:
            df: Input DataFrame with date, cell_id, NDVI, MSI, weather vars
            target_column: Target variable for prediction (NDVI or MSI)
        
        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()
        df = df.sort_values(['cell_id', 'date']).reset_index(drop=True)
        
        # Columns to create features for
        feature_columns = ['NDVI', 'MSI', 'temperature_2m', 'precipitation']
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Create lag features
        print("Creating lag features...")
        df = self.create_lag_features(df, feature_columns, self.lag_months)
        
        # Create rolling features
        print("Creating rolling features...")
        df = self.create_rolling_features(df, feature_columns, self.rolling_windows)
        
        # Create temporal features
        print("Creating temporal features...")
        df = self.create_temporal_features(df)
        
        # Create interaction features
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Normalize per cell if requested
        if self.should_normalize_per_cell:
            print("Normalizing features per cell...")
            normalize_cols = [col for col in feature_columns 
                            if col not in ['NDVI', 'MSI']]  # Don't normalize targets
            df = self.normalize_per_cell(df, normalize_cols)
        
        # Create target variable (next month's value)
        if target_column in df.columns:
            df[f'{target_column}_next'] = df.groupby('cell_id')[target_column].shift(-1)
        
        return df

