"""
Satellite Data Acquisition Module
Supports both Google Earth Engine and OpenEO for Sentinel-2 data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("Warning: Google Earth Engine not available. Install with: pip install earthengine-api")

try:
    import openeo
    OPENEO_AVAILABLE = True
except ImportError:
    OPENEO_AVAILABLE = False
    print("Warning: OpenEO not available. Install with: pip install openeo")


class SatelliteDataAcquisition:
    """Acquire NDVI and MSI from Sentinel-2 satellite data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sat_config = config.get('satellite', {})
        self.use_gee = self.sat_config.get('use_gee', True)
        self.use_openeo = self.sat_config.get('use_openeo', False)
        
        if self.use_gee and GEE_AVAILABLE:
            self._initialize_gee()
        elif self.use_openeo and OPENEO_AVAILABLE:
            self._initialize_openeo()
        else:
            raise ValueError("Either GEE or OpenEO must be available and configured")
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            ee.Initialize()
            print("Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"GEE initialization error: {e}")
            print("Please authenticate: earthengine authenticate")
            raise
    
    def _initialize_openeo(self):
        """Initialize OpenEO connection"""
        backend_url = self.sat_config.get('openeo', {}).get('backend', 'openeo-earthengine')
        # This would need actual OpenEO backend connection
        print(f"OpenEO backend: {backend_url}")
    
    def create_grid(self, bounds: Dict, grid_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """
        Create a grid of cells over the study area
        
        Args:
            bounds: Dictionary with min_lon, max_lon, min_lat, max_lat
            grid_size_km: Size of grid cells in kilometers
        
        Returns:
            GeoDataFrame with grid cells
        """
        min_lon, max_lon = bounds['min_lon'], bounds['max_lon']
        min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
        
        # Convert km to degrees based on latitude
        # At equator: 1 degree ≈ 111 km
        # Latitude conversion: 1 km ≈ 0.009 degrees (constant)
        # Longitude conversion: 1 km ≈ 0.009 / cos(latitude) degrees
        lat_center = (min_lat + max_lat) / 2
        grid_size_deg_lat = grid_size_km * 0.009  # Constant for latitude
        grid_size_deg_lon = grid_size_km * 0.009 / np.cos(np.radians(lat_center))  # Varies by latitude
        
        # Create grid with proper lat/lon spacing
        lons = np.arange(min_lon, max_lon, grid_size_deg_lon)
        lats = np.arange(min_lat, max_lat, grid_size_deg_lat)
        
        grid_cells = []
        cell_ids = []
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                cell_id = f"cell_{i}_{j}"
                cell_geom = box(lon, lat, lon + grid_size_deg_lon, lat + grid_size_deg_lat)
                grid_cells.append(cell_geom)
                cell_ids.append(cell_id)
        
        grid_gdf = gpd.GeoDataFrame(
            {'cell_id': cell_ids, 'geometry': grid_cells},
            crs='EPSG:4326'
        )
        
        return grid_gdf
    
    def calculate_ndvi(self, image: 'ee.Image') -> 'ee.Image':
        """Calculate NDVI from Sentinel-2 image"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi
    
    def calculate_msi(self, image: 'ee.Image') -> 'ee.Image':
        """Calculate MSI (Moisture Stress Index) from Sentinel-2 image"""
        # MSI = SWIR1 / NIR = B11 / B8
        msi = image.select('B11').divide(image.select('B8')).rename('MSI')
        return msi
    
    def get_sentinel2_collection(self, start_date: str, end_date: str, 
                                 bounds: Dict, cloud_cover_max: int = 20) -> 'ee.ImageCollection':
        """Get Sentinel-2 image collection filtered by date, bounds, and cloud cover"""
        if not GEE_AVAILABLE:
            raise ImportError("Google Earth Engine is required")
        
        # Define area of interest
        aoi = ee.Geometry.Rectangle([
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat']
        ])
        
        # Get Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                     .select(['B4', 'B8', 'B11', 'QA60']))  # Red, NIR, SWIR1, Quality
        
        return collection
    
    def process_monthly_averages(self, start_date: str, end_date: str,
                                bounds: Dict, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Process Sentinel-2 data to get monthly averages of NDVI and MSI per grid cell
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            bounds: Study area bounds
            grid_gdf: Grid cells GeoDataFrame
        
        Returns:
            DataFrame with columns: date, cell_id, NDVI, MSI
        """
        if not self.use_gee or not GEE_AVAILABLE:
            raise NotImplementedError("GEE processing not available")
        
        # Get image collection
        collection = self.get_sentinel2_collection(
            start_date, end_date, bounds,
            self.sat_config.get('cloud_cover_max', 20)
        )
        
        # Generate monthly date ranges
        date_ranges = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        results = []
        
        for date in date_ranges:
            month_start = date.strftime('%Y-%m-%d')
            month_end = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
            
            # Filter collection for this month
            monthly_collection = collection.filterDate(month_start, month_end)
            
            # Calculate NDVI and MSI for each image
            def add_indices(img):
                ndvi = self.calculate_ndvi(img)
                msi = self.calculate_msi(img)
                return img.addBands([ndvi, msi])
            
            monthly_collection = monthly_collection.map(add_indices)
            
            # Compute monthly median (reduces cloud effects)
            monthly_median = monthly_collection.select(['NDVI', 'MSI']).median()
            
            # Extract values for each grid cell
            for idx, row in grid_gdf.iterrows():
                cell_id = row['cell_id']
                cell_geom = row['geometry']
                
                # Convert to EE geometry
                coords = list(cell_geom.exterior.coords)
                ee_geom = ee.Geometry.Polygon([[[lon, lat] for lon, lat in coords]])
                
                # Sample the image at the grid cell center
                center = cell_geom.centroid
                point = ee.Geometry.Point([center.x, center.y])
                
                # Get pixel values using actual GEE data extraction
                try:
                    sample = monthly_median.sample(
                        region=point,
                        scale=10,  # Sentinel-2 resolution
                        numPixels=1
                    )
                    
                    # Extract actual values from GEE sample
                    sample_info = sample.getInfo()
                    
                    if sample_info and 'features' in sample_info and len(sample_info['features']) > 0:
                        properties = sample_info['features'][0]['properties']
                        ndvi_value = properties.get('NDVI', np.nan)
                        msi_value = properties.get('MSI', np.nan)
                    else:
                        ndvi_value = np.nan
                        msi_value = np.nan
                    
                    results.append({
                        'date': month_start,
                        'cell_id': cell_id,
                        'lon': center.x,
                        'lat': center.y,
                        'NDVI': ndvi_value,
                        'MSI': msi_value
                    })
                except Exception as e:
                    print(f"Error processing cell {cell_id} for {month_start}: {e}")
                    results.append({
                        'date': month_start,
                        'cell_id': cell_id,
                        'lon': center.x,
                        'lat': center.y,
                        'NDVI': np.nan,
                        'MSI': np.nan
                    })
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_satellite_data(self, start_date: str, end_date: str,
                          bounds: Dict, grid_size_km: float = 1.0) -> pd.DataFrame:
        """
        Main method to get satellite data
        
        Returns:
            DataFrame with monthly NDVI and MSI per grid cell
        """
        # Create grid
        grid_gdf = self.create_grid(bounds, grid_size_km)
        print(f"Created grid with {len(grid_gdf)} cells")
        
        # Process monthly averages
        df = self.process_monthly_averages(start_date, end_date, bounds, grid_gdf)
        
        return df

