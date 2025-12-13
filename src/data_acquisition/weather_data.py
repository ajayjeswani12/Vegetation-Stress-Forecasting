"""
Weather Data Acquisition Module
Supports ERA5 CDS API, Open-Meteo API, and Met Office data
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import os
import zipfile
from pathlib import Path

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False
    print("Warning: cdsapi not available. Install with: pip install cdsapi")
    print("For ERA5 data, you also need to set up ~/.cdsapirc file with your credentials")


class WeatherDataAcquisition:
    """Acquire weather data from various sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weather_config = config.get('weather', {})
        self.source = self.weather_config.get('source', 'open-meteo')
        self.variables = self.weather_config.get('variables', [])
        
        if self.source == 'open-meteo':
            self.base_url = self.weather_config.get('open_meteo', {}).get(
                'base_url', 'https://archive-api.open-meteo.com/v1/archive'
            )
        elif self.source == 'era5':
            if not CDSAPI_AVAILABLE:
                raise ImportError("cdsapi is required for ERA5 data. Install with: pip install cdsapi")
            # Initialize CDS client (credentials should be in ~/.cdsapirc)
            try:
                self.cds_client = cdsapi.Client()
                print("ERA5 CDS API client initialized successfully")
            except Exception as e:
                print(f"Error initializing CDS API client: {e}")
                print("Please ensure ~/.cdsapirc file exists with your CDS API credentials")
                print("See: https://cds.climate.copernicus.eu/api-how-to")
                raise
    
    def get_era5_data(self, bounds: Dict, start_date: str, end_date: str,
                     max_retries: int = 3, retry_delay: int = 60) -> pd.DataFrame:
        """
        Get ERA5-Land hourly data from CDS API and aggregate to monthly
        
        Args:
            bounds: Dictionary with min_lon, max_lon, min_lat, max_lat
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            DataFrame with monthly aggregated weather variables
        """
        if not CDSAPI_AVAILABLE:
            raise ImportError("cdsapi is required for ERA5 data")
        
        # Check if xarray and netcdf4 are available
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for ERA5 data. Install with: pip install xarray")
        
        # Verify netcdf4 is actually available
        try:
            import netCDF4
            # Verify it's actually working
            try:
                # Check if we can access the Dataset class
                _ = netCDF4.Dataset
            except AttributeError:
                raise ImportError("netcdf4 is installed but not working properly. "
                                "Try: pip install --upgrade netcdf4")
        except ImportError:
            raise ImportError("netcdf4 is required for ERA5 data (NetCDF 4 format). "
                            "Install with: pip install netcdf4")
        
        # Verify xarray can use netcdf4 engine
        try:
            # Check if netcdf4 backend is available in xarray
            backends = xr.backends.list_engines()
            if 'netcdf4' not in backends:
                print("Warning: xarray cannot find netcdf4 engine.")
                print("  This may still work, but if you get errors, try:")
                print("  pip install --upgrade xarray netcdf4")
        except Exception:
            # list_engines might not be available in older xarray versions
            pass
        
        # ERA5-Land dataset parameters
        # Map our variable names to ERA5 variable names
        # Note: CDS API uses long names in requests, but NetCDF files use short names
        era5_variable_map = {
            'temperature_2m': '2m_temperature',  # CDS API name
            'precipitation': 'total_precipitation',  # CDS API name
            'shortwave_radiation': 'surface_solar_radiation_downwards',  # CDS API name
            'soil_moisture_0_to_7cm': 'volumetric_soil_water_layer_1'  # CDS API name
        }
        
        # Map CDS API names to NetCDF short names (what's actually in the files)
        era5_netcdf_name_map = {
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
            'surface_solar_radiation_downwards': 'ssrd',
            'volumetric_soil_water_layer_1': 'swvl1'
        }
        
        # Get ERA5 variables for requested variables
        era5_variables = []
        for var in self.variables:
            if var in era5_variable_map:
                era5_variables.append(era5_variable_map[var])
        
        if not era5_variables:
            era5_variables = ['2m_temperature', 'total_precipitation', 
                            'surface_solar_radiation_downwards', 
                            'volumetric_soil_water_layer_1']
        
        # Limit variables if too many (to reduce request size)
        # Start with essential variables only
        essential_vars = ['2m_temperature', 'total_precipitation']
        if len(era5_variables) > 2:
            print(f"Warning: Requesting {len(era5_variables)} variables may exceed cost limits.")
            print(f"  Consider reducing to essential variables only in config.yaml")
            print(f"  Essential: temperature_2m, precipitation")
            print(f"  Optional: shortwave_radiation, soil_moisture_0_to_7cm")
        
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create output directory for ERA5 downloads
        output_dir = Path(self.config.get('output', {}).get('data_dir', 'data')) / 'era5_downloads'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download data month by month to avoid cost limit errors
        # ERA5 has strict cost limits, so we need to make smaller requests
        # Process incrementally to avoid memory issues
        monthly_dataframes = []  # Store aggregated monthly DataFrames instead of raw datasets
        
        # Generate monthly date ranges
        date_ranges = pd.date_range(start=start_dt, end=end_dt, freq='MS')
        
        for month_start in date_ranges:
            month_end = month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            month_end = min(month_end, end_dt)
            
            year = month_start.year
            month = month_start.month
            
            # Get actual number of days in this month
            days_in_month = (month_end - month_start).days + 1
            days_list = [f'{d:02d}' for d in range(1, days_in_month + 1)]
            
            output_file = output_dir / f'era5_{year}_{month:02d}.nc'
            
            # Check if file already exists (or extracted NetCDF from ZIP)
            actual_nc_file = None
            extract_dir = output_file.parent / f"{output_file.stem}_extracted"
            
            # Check for extracted NetCDF file first
            if extract_dir.exists():
                nc_files = list(extract_dir.glob('*.nc'))
                if not nc_files:
                    nc_files = list(extract_dir.rglob('*.nc'))
                if nc_files:
                    actual_nc_file = nc_files[0]
            
            # If no extracted file, check if original file exists
            if actual_nc_file is None and output_file.exists():
                # Check if it's a ZIP file
                try:
                    with open(output_file, 'rb') as f:
                        first_bytes = f.read(4)
                        if first_bytes[:2] == b'PK':
                            # It's a ZIP, extract it
                            extract_dir.mkdir(exist_ok=True)
                            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            nc_files = list(extract_dir.glob('*.nc'))
                            if not nc_files:
                                nc_files = list(extract_dir.rglob('*.nc'))
                            if nc_files:
                                actual_nc_file = nc_files[0]
                except:
                    # Not a ZIP, use as NetCDF
                    actual_nc_file = output_file
            
            if actual_nc_file and actual_nc_file.exists():
                print(f"ERA5 data for {year}-{month:02d} already exists, processing...")
                try:
                    # Validate file is not empty
                    file_size = actual_nc_file.stat().st_size
                    if file_size == 0:
                        print(f"  File is empty, will re-download...")
                        if actual_nc_file != output_file:
                            actual_nc_file.unlink()
                        output_file.unlink()
                    else:
                        # Process this month's data immediately (don't store in memory)
                        monthly_df = self._process_single_month_era5(
                            actual_nc_file, era5_variable_map, month_start
                        )
                        if monthly_df is not None and not monthly_df.empty:
                            monthly_dataframes.append(monthly_df)
                            print(f"  Successfully processed cached file")
                            continue
                        else:
                            print(f"  Failed to process, will re-download...")
                except Exception as e:
                    error_msg = str(e)
                    if 'unknown file format' in error_msg.lower() or 'not a valid' in error_msg.lower():
                        print(f"  File appears corrupted, deleting and will re-download...")
                        try:
                            output_file.unlink()
                        except:
                            pass
                    elif 'netcdf4' in error_msg.lower():
                        print(f"Error: Cannot read NetCDF 4 file. netcdf4 library may not be properly installed.")
                        print(f"  Error: {e}")
                        print(f"  Try: pip install --upgrade netcdf4")
                        print(f"  Will attempt to re-download...")
                    else:
                        print(f"Error loading existing file, will re-download: {e}")
            
            # CDS API request parameters - monthly request (much smaller)
            request_params = {
                'product_type': 'reanalysis',
                'variable': era5_variables,
                'year': str(year),
                'month': f'{month:02d}',
                'day': days_list,  # Only actual days in this month
                'time': [f'{h:02d}:00' for h in range(24)],  # All hours (needed for aggregation)
                'area': [
                    bounds['max_lat'],  # North
                    bounds['min_lon'],  # West
                    bounds['min_lat'],  # South
                    bounds['max_lon']   # East
                ],
                'format': 'netcdf',
            }
            
            # Retry logic for CDS API (can be slow and timeout)
            for attempt in range(max_retries):
                try:
                    print(f"Requesting ERA5 data for {year}-{month:02d} (attempt {attempt + 1}/{max_retries})...")
                    print(f"  Days: {days_in_month}, Hours: 24, Variables: {len(era5_variables)}")
                    print("  Note: CDS API requests can take 5-15 minutes to process")
                    
                    self.cds_client.retrieve(
                        'reanalysis-era5-land',
                        request_params,
                        str(output_file)
                    )
                    
                    # Verify file was downloaded and wait if needed
                    import time
                    max_wait = 30  # Wait up to 30 seconds for file
                    wait_count = 0
                    while not output_file.exists() and wait_count < max_wait:
                        time.sleep(1)
                        wait_count += 1
                    
                    if not output_file.exists():
                        raise FileNotFoundError(f"Downloaded file not found: {output_file}")
                    
                    # Wait a moment for file to be fully written and check file size
                    time.sleep(2)
                    
                    # Verify file is not empty and check if it might be an error message
                    file_size = output_file.stat().st_size
                    if file_size == 0:
                        raise ValueError(f"Downloaded file is empty: {output_file}")
                    
                    # Check if downloaded file is a ZIP file (CDS API sometimes returns ZIP)
                    actual_nc_file = output_file
                    is_zip = False
                    
                    try:
                        with open(output_file, 'rb') as f:
                            first_bytes = f.read(4)
                            # ZIP files start with PK\x03\x04 or PK\x05\x06
                            if first_bytes[:2] == b'PK':
                                is_zip = True
                                print(f"  Detected ZIP file, extracting...")
                    except:
                        pass
                    
                    # If it's a ZIP file, extract it
                    if is_zip:
                        try:
                            extract_dir = output_file.parent / f"{output_file.stem}_extracted"
                            extract_dir.mkdir(exist_ok=True)
                            
                            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                                print(f"  Extracted ZIP to: {extract_dir}")
                            
                            # Find the NetCDF file in the extracted directory
                            nc_files = list(extract_dir.glob('*.nc'))
                            if not nc_files:
                                # Try subdirectories
                                nc_files = list(extract_dir.rglob('*.nc'))
                            
                            if nc_files:
                                actual_nc_file = nc_files[0]
                                print(f"  Found NetCDF file: {actual_nc_file.name}")
                                # Optionally move it to the expected location
                                if actual_nc_file != output_file:
                                    # Keep the extracted file, use it directly
                                    pass
                            else:
                                raise ValueError(f"No NetCDF file found in ZIP archive: {output_file}")
                        except zipfile.BadZipFile:
                            # Not a valid ZIP, might be corrupted or actually NetCDF
                            print(f"  File is not a valid ZIP, treating as NetCDF...")
                            is_zip = False
                        except Exception as zip_error:
                            raise ValueError(f"Error extracting ZIP file: {zip_error}")
                    
                    # Check if file might be an error message (HTML/text) instead of NetCDF
                    if not is_zip and file_size < 1000:  # Very small file might be an error message
                        try:
                            with open(actual_nc_file, 'rb') as f:
                                first_bytes = f.read(100)
                                # Check for HTML/error message indicators
                                if b'<html' in first_bytes.lower() or b'error' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
                                    with open(actual_nc_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read(500)
                                    raise ValueError(f"Downloaded file appears to be an error message, not NetCDF:\n{content[:200]}")
                        except:
                            pass
                    
                    # Load the downloaded file with explicit engine
                    import xarray as xr
                    try:
                        # ERA5 files are NetCDF 4 format, must use netcdf4 engine
                        # Try to validate it's a NetCDF file first
                        try:
                            import netCDF4
                            # Quick validation: try to open with netCDF4 directly
                            test_nc = netCDF4.Dataset(str(actual_nc_file), 'r')
                            test_nc.close()
                        except Exception as nc_error:
                            # If netCDF4 can't open it, it's likely corrupted or wrong format
                            raise ValueError(f"File is not a valid NetCDF file. "
                                           f"netCDF4 error: {nc_error}. "
                                           f"File might be corrupted. Try deleting and re-downloading: {actual_nc_file}")
                        
                        # Process this month's data immediately (don't store raw dataset in memory)
                        monthly_df = self._process_single_month_era5(
                            actual_nc_file, era5_variable_map, month_start
                        )
                        if monthly_df is not None and not monthly_df.empty:
                            monthly_dataframes.append(monthly_df)
                            print(f"Successfully downloaded and processed ERA5 data for {year}-{month:02d}")
                            break
                        else:
                            raise ValueError(f"Failed to process downloaded file: {actual_nc_file}. "
                                           f"Check error messages above for details.")
                    except Exception as e:
                        error_msg = str(e)
                        if 'unknown file format' in error_msg.lower() or 'not a valid' in error_msg.lower():
                            # File might be corrupted - suggest deleting and retrying
                            print(f"WARNING: File appears corrupted or invalid format: {output_file}")
                            print(f"  File size: {file_size} bytes")
                            print(f"  Error: {error_msg}")
                            print(f"  Attempting to delete corrupted file and re-download...")
                            try:
                                output_file.unlink()  # Delete corrupted file
                            except:
                                pass
                            # Re-raise to trigger retry
                            raise ValueError(f"Corrupted NetCDF file detected. Will retry download. "
                                           f"Original error: {error_msg}")
                        elif 'netcdf4' in error_msg.lower():
                            raise ImportError(
                                f"Failed to open NetCDF 4 file. The file requires netcdf4 library.\n"
                                f"Error: {e}\n"
                                f"Please install/upgrade netcdf4: pip install --upgrade netcdf4\n"
                                f"Or reinstall: pip uninstall netcdf4 && pip install netcdf4"
                            )
                        else:
                            raise ValueError(f"Failed to process NetCDF file: {e}")
                    
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a cost limit error
                    if 'cost limits exceeded' in error_str.lower() or 'too large' in error_str.lower():
                        print(f"ERROR: Request too large for CDS API cost limits")
                        print(f"  Try reducing: time period, area size, or number of variables")
                        print(f"  Current area: {bounds['max_lat']:.2f}N, {bounds['min_lon']:.2f}E to {bounds['min_lat']:.2f}N, {bounds['max_lon']:.2f}E")
                        if attempt < max_retries - 1:
                            print(f"  Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            raise ValueError(f"ERA5 request too large. Try reducing study area or time period.")
                    elif attempt < max_retries - 1:
                        print(f"Error downloading ERA5 data for {year}-{month:02d}: {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to download ERA5 data for {year}-{month:02d} after {max_retries} attempts: {e}")
                        raise
        
        # Combine all monthly DataFrames (much smaller than raw datasets)
        if not monthly_dataframes:
            raise ValueError("No ERA5 data was successfully processed")
        
        # Concatenate all monthly DataFrames
        combined_df = pd.concat(monthly_dataframes, ignore_index=True)
        
        return combined_df
    
    def _process_single_month_era5(self, nc_file: Path, era5_variable_map: Dict,
                                   month_start: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Process a single month's ERA5 NetCDF file and return aggregated monthly DataFrame
        
        Args:
            nc_file: Path to NetCDF file
            era5_variable_map: Mapping of our variable names to ERA5 names
            month_start: Start date of the month
        
        Returns:
            DataFrame with monthly aggregated data for this month
        """
        try:
            import xarray as xr
            import netCDF4
            
            # Open dataset
            print(f"  Opening NetCDF file: {nc_file}")
            ds = xr.open_dataset(nc_file, engine='netcdf4')
            
            # Debug: Print available variables
            print(f"  Available variables in file: {list(ds.data_vars.keys())}")
            print(f"  Available coordinates: {list(ds.coords.keys())}")
            
            # Map CDS API names to NetCDF short names (what's actually in the files)
            era5_netcdf_name_map = {
                '2m_temperature': 't2m',
                'total_precipitation': 'tp',
                'surface_solar_radiation_downwards': 'ssrd',
                'volumetric_soil_water_layer_1': 'swvl1'
            }
            
            # Check if we have any of the expected variables (using short NetCDF names)
            reverse_map = {v: k for k, v in era5_variable_map.items()}
            
            # First, try to find variables by their NetCDF short names
            available_netcdf_vars = []
            var_mapping = {}  # Maps NetCDF name -> our name
            
            for cds_name, netcdf_name in era5_netcdf_name_map.items():
                if netcdf_name in ds.data_vars:
                    available_netcdf_vars.append(netcdf_name)
                    # Map NetCDF name to our variable name
                    if cds_name in reverse_map:
                        var_mapping[netcdf_name] = reverse_map[cds_name]
            
            # Also check for long names (in case some files use them)
            for cds_name in reverse_map.keys():
                if cds_name in ds.data_vars and cds_name not in [v for v in var_mapping.values()]:
                    available_netcdf_vars.append(cds_name)
                    var_mapping[cds_name] = reverse_map[cds_name]
            
            if not available_netcdf_vars:
                print(f"  WARNING: None of the expected ERA5 variables found in file!")
                print(f"  Expected (CDS API): {list(reverse_map.keys())}")
                print(f"  Expected (NetCDF short): {list(era5_netcdf_name_map.values())}")
                print(f"  Found: {list(ds.data_vars.keys())}")
                ds.close()
                return None
            
            print(f"  Processing variables: {available_netcdf_vars}")
            print(f"  Variable mapping: {var_mapping}")
            
            # Convert to DataFrame (this will be large for hourly data, but we aggregate immediately)
            # Select only the variables we need to reduce memory
            vars_to_select = list(ds.coords.keys()) + available_netcdf_vars
            df = ds[vars_to_select].to_dataframe().reset_index()
            
            # Close dataset to free memory
            ds.close()
            del ds
            
            # Rename columns from NetCDF names to our names
            rename_dict = {}
            for netcdf_name, our_var in var_mapping.items():
                if netcdf_name in df.columns:
                    rename_dict[netcdf_name] = our_var
            
            df = df.rename(columns=rename_dict)
            
            # Convert temperature from Kelvin to Celsius
            if 'temperature_2m' in df.columns:
                df['temperature_2m'] = df['temperature_2m'] - 273.15
            
            # Convert precipitation from m to mm
            if 'precipitation' in df.columns:
                df['precipitation'] = df['precipitation'] * 1000
            
            # Aggregate to monthly (mean for temperature/radiation, sum for precipitation)
            # Check for time coordinate (might be 'time' or other names)
            time_col = None
            for col in ['time', 'valid_time', 'datetime']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                df['date'] = pd.to_datetime(df[time_col])
                df['year_month'] = df['date'].dt.to_period('M')
                
                # Monthly aggregation
                agg_dict = {}
                if 'temperature_2m' in df.columns:
                    agg_dict['temperature_2m'] = 'mean'
                if 'precipitation' in df.columns:
                    agg_dict['precipitation'] = 'sum'
                if 'shortwave_radiation' in df.columns:
                    agg_dict['shortwave_radiation'] = 'mean'
                if 'soil_moisture_0_to_7cm' in df.columns:
                    agg_dict['soil_moisture_0_to_7cm'] = 'mean'
                
                if not agg_dict:
                    print(f"  WARNING: No variables to aggregate after renaming")
                    print(f"  Available columns: {list(df.columns)}")
                    return None
                
                # Group by year_month and spatial coordinates
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    monthly_df = df.groupby(['year_month', 'latitude', 'longitude']).agg(agg_dict).reset_index()
                    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
                    monthly_df = monthly_df.drop('year_month', axis=1)
                    monthly_df = monthly_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
                elif 'lat' in df.columns and 'lon' in df.columns:
                    monthly_df = df.groupby(['year_month', 'lat', 'lon']).agg(agg_dict).reset_index()
                    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
                    monthly_df = monthly_df.drop('year_month', axis=1)
                else:
                    # If no spatial coordinates, aggregate globally
                    monthly_df = df.groupby('year_month').agg(agg_dict).reset_index()
                    monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
                    monthly_df = monthly_df.drop('year_month', axis=1)
                    monthly_df['lat'] = np.nan
                    monthly_df['lon'] = np.nan
                
                print(f"  Aggregated to {len(monthly_df)} monthly records")
                
                # Free memory
                del df
                
                return monthly_df
            else:
                print(f"  ERROR: No time coordinate found in file")
                print(f"  Available columns: {list(df.columns)}")
                return None
                
        except Exception as e:
            import traceback
            print(f"  ERROR processing {nc_file}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            return None
    
    def get_era5_data_for_grid(self, grid_df: pd.DataFrame, bounds: Dict,
                               start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get ERA5 data and match to grid cells
        
        Args:
            grid_df: DataFrame with cell_id, lon, lat columns
            bounds: Study area bounds
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather data per cell and date
        """
        # Get ERA5 data for the entire region
        era5_df = self.get_era5_data(bounds, start_date, end_date)
        
        if era5_df.empty:
            return pd.DataFrame()
        
        # Match ERA5 grid points to our grid cells
        results = []
        
        for _, cell_row in grid_df.iterrows():
            cell_id = cell_row['cell_id']
            cell_lon = cell_row['lon']
            cell_lat = cell_row['lat']
            
            # Find nearest ERA5 grid point for this cell
            if 'lat' in era5_df.columns and 'lon' in era5_df.columns:
                # Calculate distances and find nearest
                era5_df['distance'] = np.sqrt(
                    (era5_df['lat'] - cell_lat)**2 + 
                    (era5_df['lon'] - cell_lon)**2
                )
                
                # For each date, get the nearest point
                for date in era5_df['date'].unique():
                    date_data = era5_df[era5_df['date'] == date]
                    nearest_idx = date_data['distance'].idxmin()
                    nearest_row = date_data.loc[nearest_idx]
                    
                    result_row = {
                        'date': date,
                        'cell_id': cell_id,
                        'lon': cell_lon,
                        'lat': cell_lat,
                        'temperature_2m': nearest_row.get('temperature_2m', np.nan),
                        'precipitation': nearest_row.get('precipitation', np.nan),
                        'shortwave_radiation': nearest_row.get('shortwave_radiation', np.nan),
                        'soil_moisture_0_to_7cm': nearest_row.get('soil_moisture_0_to_7cm', np.nan)
                    }
                    results.append(result_row)
            else:
                # If no spatial coordinates, assign same values to all cells
                for _, era5_row in era5_df.iterrows():
                    result_row = {
                        'date': era5_row['date'],
                        'cell_id': cell_id,
                        'lon': cell_lon,
                        'lat': cell_lat,
                        'temperature_2m': era5_row.get('temperature_2m', np.nan),
                        'precipitation': era5_row.get('precipitation', np.nan),
                        'shortwave_radiation': era5_row.get('shortwave_radiation', np.nan),
                        'soil_moisture_0_to_7cm': era5_row.get('soil_moisture_0_to_7cm', np.nan)
                    }
                    results.append(result_row)
        
        return pd.DataFrame(results)
    
    def get_open_meteo_data(self, lat: float, lon: float, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get weather data from Open-Meteo API
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather variables
        """
        # Map variable names to Open-Meteo parameters
        variable_map = {
            'temperature_2m': 'temperature_2m',
            'precipitation': 'precipitation',
            'soil_moisture_0_to_7cm': 'soil_moisture_0_to_7cm',
            'shortwave_radiation': 'shortwave_radiation_sum'
        }
        
        # Build parameters
        daily_params = []
        
        for var in self.variables:
            if var in variable_map:
                mapped_var = variable_map[var]
                daily_params.append(mapped_var)
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(daily_params) if daily_params else 'temperature_2m_max,precipitation_sum',
            'timezone': self.weather_config.get('open_meteo', {}).get('timezone', 'Asia/Karachi')
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if 'daily' in data:
                daily_data = data['daily']
                dates = pd.to_datetime(daily_data['time'])
                
                df = pd.DataFrame({
                    'date': dates,
                    'temperature_2m': daily_data.get('temperature_2m_max', [np.nan] * len(dates)),
                    'precipitation': daily_data.get('precipitation_sum', [np.nan] * len(dates)),
                    'shortwave_radiation': daily_data.get('shortwave_radiation_sum', [np.nan] * len(dates))
                })
                
                # Aggregate to monthly
                df['year_month'] = df['date'].dt.to_period('M')
                monthly_df = df.groupby('year_month').agg({
                    'temperature_2m': 'mean',
                    'precipitation': 'sum',
                    'shortwave_radiation': 'sum'
                }).reset_index()
                
                monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
                monthly_df = monthly_df.drop('year_month', axis=1)
                
                return monthly_df
            else:
                print(f"Warning: No daily data in response for {lat}, {lon}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Open-Meteo data for {lat}, {lon}: {e}")
            return pd.DataFrame()
    
    def get_weather_for_grid(self, grid_df: pd.DataFrame, 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get weather data for all grid cells
        
        Args:
            grid_df: DataFrame with columns: cell_id, lon, lat
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather data per cell and date
        """
        if self.source == 'era5':
            # Get bounds from grid
            bounds = {
                'min_lon': grid_df['lon'].min(),
                'max_lon': grid_df['lon'].max(),
                'min_lat': grid_df['lat'].min(),
                'max_lat': grid_df['lat'].max()
            }
            return self.get_era5_data_for_grid(grid_df, bounds, start_date, end_date)
        
        results = []
        unique_cells = grid_df[['cell_id', 'lon', 'lat']].drop_duplicates()
        
        print(f"Fetching weather data for {len(unique_cells)} grid cells...")
        
        for idx, row in unique_cells.iterrows():
            cell_id = row['cell_id']
            lon = row['lon']
            lat = row['lat']
            
            if self.source == 'open-meteo':
                weather_df = self.get_open_meteo_data(lat, lon, start_date, end_date)
            else:
                raise NotImplementedError(f"Weather source {self.source} not implemented")
            
            if not weather_df.empty:
                weather_df['cell_id'] = cell_id
                weather_df['lon'] = lon
                weather_df['lat'] = lat
                results.append(weather_df)
            
            # Rate limiting
            time.sleep(0.1)
        
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_weather_data(self, grid_df: pd.DataFrame,
                        start_date: str, end_date: str) -> pd.DataFrame:
        """
        Main method to get weather data
        
        Args:
            grid_df: DataFrame with grid cell information
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather variables per cell and date
        """
        return self.get_weather_for_grid(grid_df, start_date, end_date)
