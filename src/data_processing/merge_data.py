"""
Data Merging and Processing Module
Combines satellite and weather data into a unified dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class DataMerger:
    """Merge satellite and weather data"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def merge_satellite_weather(self, satellite_df: pd.DataFrame,
                               weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge satellite indices with weather data
        
        Args:
            satellite_df: DataFrame with columns: date, cell_id, NDVI, MSI, lon, lat
            weather_df: DataFrame with columns: date, cell_id, temperature_2m, precipitation, etc.
        
        Returns:
            Merged DataFrame
        """
        # Ensure date columns are datetime
        satellite_df['date'] = pd.to_datetime(satellite_df['date'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Merge on date and cell_id
        merged_df = pd.merge(
            satellite_df,
            weather_df,
            on=['date', 'cell_id'],
            how='outer',
            suffixes=('', '_weather')
        )
        
        # Handle duplicate lon/lat columns
        if 'lon_weather' in merged_df.columns:
            merged_df['lon'] = merged_df['lon'].fillna(merged_df['lon_weather'])
            merged_df['lat'] = merged_df['lat'].fillna(merged_df['lat_weather'])
            merged_df = merged_df.drop(['lon_weather', 'lat_weather'], axis=1)
        
        # Sort by cell_id and date
        merged_df = merged_df.sort_values(['cell_id', 'date']).reset_index(drop=True)
        
        return merged_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the merged dataset
        
        Args:
            df: Merged DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with all NaN values
        df = df.dropna(subset=['NDVI', 'MSI'], how='all')
        
        # Clip NDVI to valid range [-1, 1]
        if 'NDVI' in df.columns:
            df['NDVI'] = df['NDVI'].clip(-1, 1)
        
        # Clip MSI to reasonable range [0, 10]
        if 'MSI' in df.columns:
            df['MSI'] = df['MSI'].clip(0, 10)
        
        # Remove extreme outliers (optional)
        for col in ['NDVI', 'MSI', 'temperature_2m', 'precipitation']:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df.loc[df[col] < q1, col] = np.nan
                df.loc[df[col] > q3, col] = np.nan
        
        return df
    
    def aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure data is aggregated to monthly level
        
        Args:
            df: DataFrame with date column
        
        Returns:
            Monthly aggregated DataFrame
        """
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Group by cell_id and year_month
        agg_dict = {
            'NDVI': 'mean',
            'MSI': 'mean',
            'temperature_2m': 'mean',
            'precipitation': 'sum',
            'shortwave_radiation': 'sum' if 'shortwave_radiation' in df.columns else 'mean',
            'lon': 'first',
            'lat': 'first'
        }
        
        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        monthly_df = df.groupby(['cell_id', 'year_month']).agg(agg_dict).reset_index()
        monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
        monthly_df = monthly_df.drop('year_month', axis=1)
        
        return monthly_df

