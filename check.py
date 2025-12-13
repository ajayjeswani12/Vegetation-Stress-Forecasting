import zipfile
import os
import xarray as xr

zip_path = "data/era5_downloads/era5_2023_01.nc"  # the ZIP file
extract_dir = os.path.dirname(zip_path)  # same folder as the ZIP

# Extract the ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Usually ERA5 ZIP contains a single .nc file
    for file in zip_ref.namelist():
        if file.endswith('.nc'):
            extracted_path = os.path.join(extract_dir, file)
            zip_ref.extract(file, extract_dir)
            break

print("Extracted NetCDF file to:", extracted_path)

# Open with xarray
ds = xr.open_dataset(extracted_path)
print(ds)
